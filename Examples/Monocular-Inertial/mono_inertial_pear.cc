/**
 * @file mono_inertial_pear.cc
 * @brief Monocular-Inertial VIO with PearAPI + MAVLink/ArduPilot Integration
 *
 * Combines:
 * - Camera/IMU/trigger logic from mono_inertial_custom_cam.cc (PearAPI)
 * - Full MAVLink/VIOBridge infrastructure from stereo_inertial_realsense_D455_VIO_v2.cc
 *
 * Hardware:
 * - OV9281 global shutter camera (via libcamera / PearAPI)
 * - BMI160 IMU via Raspberry Pi Pico (hardware-triggered camera)
 * - UART connection to ArduPilot flight controller
 *
 * Usage: ./mono_inertial_pear path_to_vocabulary path_to_settings
 *            [imu_serial_port] [mavlink_serial_port] [mavlink_baud] [mode] [visualization]
 *
 *   Defaults: /dev/ttyACM0, /dev/ttyAMA0, 1500000, 1 (VISION_POSITION_ESTIMATE), 0 (OFF)
 */

#include <signal.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <ctime>
#include <sstream>
#include <thread>
#include <atomic>
#include <iomanip>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <memory>
#include <set>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <cstdlib>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <Eigen/Dense>

// PearAPI for camera and IMU access
#include <PearAPI/PearAPI.h>

#include <System.h>

// I2C includes for OV9281 trigger mode control
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>
#include <errno.h>
#include <termios.h>

// MAVLink headers
#include <common/mavlink.h>

using namespace std;

//=============================================================================
// Global state
//=============================================================================

atomic<bool> b_continue_session{true};

void exit_loop_handler(int s) {
    cout << "\nFinishing session..." << endl;
    b_continue_session = false;
}

//=============================================================================
// Configuration Constants
//=============================================================================

// Map change detection threshold (meters)
// Position jumps larger than this after tracking recovery indicate a new map
constexpr double MAP_CHANGE_THRESHOLD = 2.0;

// EKF feedback timeout (milliseconds)
constexpr uint32_t EKF_FEEDBACK_TIMEOUT_MS = 1000;

// Covariance values
constexpr float NORMAL_POSITION_COVARIANCE = 0.1f;      // Normal tracking: ~10cm uncertainty
constexpr float RECENTLY_LOST_COVARIANCE = 10.0f;       // Recently lost: high uncertainty
constexpr float NORMAL_ANGLE_COVARIANCE = 0.01f;        // ~0.5 degree uncertainty

// Quality values (0-100 scale)
constexpr int8_t QUALITY_GOOD = 100;
constexpr int8_t QUALITY_RECENTLY_LOST = 10;

//=============================================================================
// Utility Functions
//=============================================================================

// Get current time in milliseconds
uint32_t getCurrentTimeMs() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();
}

//=============================================================================
// MAVLink Mode Enumeration
//=============================================================================

enum class MAVLinkMode {
    ODOMETRY,                    // Send ODOMETRY messages
    VISION_POSITION_ESTIMATE,    // Send VISION_POSITION_ESTIMATE messages
    VISION_POSITION_AND_SPEED    // Send both VISION_POSITION_ESTIMATE and VISION_SPEED_ESTIMATE
};

//=============================================================================
// Odometry Data Structure
//=============================================================================

struct OdometryData {
    uint64_t time_usec;
    float x, y, z;                    // Position
    float q[4];                       // Quaternion (w, x, y, z)
    float vx, vy, vz;                 // Linear velocity
    float rollspeed, pitchspeed, yawspeed;  // Angular velocity
    float pose_covariance[21];        // Position covariance
    float velocity_covariance[21];    // Velocity covariance
    uint8_t reset_counter;
    uint8_t estimator_type;
    int8_t quality;
};

//=============================================================================
// Thread-Safe Odometry Queue
//=============================================================================

class OdometryQueue {
private:
    std::queue<OdometryData> queue;
    std::mutex mutex;
    std::condition_variable cv;
    const size_t max_size = 100;

public:
    void push(const OdometryData& data) {
        std::unique_lock<std::mutex> lock(mutex);
        if (queue.size() >= max_size) {
            queue.pop();
        }
        queue.push(data);
        cv.notify_one();
    }

    bool pop(OdometryData& data, int timeout_ms = 100) {
        std::unique_lock<std::mutex> lock(mutex);
        if (cv.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                        [this] { return !queue.empty(); })) {
            data = queue.front();
            queue.pop();
            return true;
        }
        return false;
    }

    size_t size() {
        std::lock_guard<std::mutex> lock(mutex);
        return queue.size();
    }
};

//=============================================================================
// EKF Position Feedback Structure
//=============================================================================

struct EKFPositionFeedback {
    Eigen::Vector3d position;
    Eigen::Vector3d velocity;
    uint32_t timestamp_ms;
    bool valid;

    EKFPositionFeedback()
        : position(Eigen::Vector3d::Zero())
        , velocity(Eigen::Vector3d::Zero())
        , timestamp_ms(0)
        , valid(false) {}
};

//=============================================================================
// MAVLink Interface (Extended with EKF Feedback)
//=============================================================================

class MAVLinkInterface {
private:
    int serial_fd;
    uint8_t system_id;
    uint8_t component_id;
    MAVLinkMode tx_mode;

    std::atomic<bool> running;
    std::thread mavlink_thread;
    OdometryQueue odometry_queue;
    std::atomic<bool> connected;
    std::mutex connected_mutex;
    std::condition_variable connected_cv;

    // EKF position feedback (thread-safe)
    mutable std::mutex ekf_mutex;
    EKFPositionFeedback ekf_feedback;

public:
    MAVLinkInterface(const char* serial_port, int baud_rate = 57600, MAVLinkMode mode = MAVLinkMode::ODOMETRY)
        : system_id(255), component_id(197), running(false), connected(false), tx_mode(mode) {

        std::cout << "[MAVLink] Initializing with mode = " << static_cast<int>(mode)
                  << " (0=ODOMETRY, 1=VISION_POS, 2=VISION_POS+SPEED)" << std::endl;

        // Open serial port
        serial_fd = open(serial_port, O_RDWR | O_NOCTTY);
        if (serial_fd < 0) {
            std::cerr << "[MAVLink] Failed to open serial port " << serial_port
                     << ": " << strerror(errno) << std::endl;
            exit(1);
        }

        // Configure serial port
        struct termios tty;
        memset(&tty, 0, sizeof(tty));

        if (tcgetattr(serial_fd, &tty) != 0) {
            std::cerr << "[MAVLink] Error getting serial port attributes: "
                     << strerror(errno) << std::endl;
            close(serial_fd);
            exit(1);
        }

        // Set baud rate
        speed_t speed = B57600;
        switch (baud_rate) {
            case 9600:    speed = B9600; break;
            case 19200:   speed = B19200; break;
            case 38400:   speed = B38400; break;
            case 57600:   speed = B57600; break;
            case 115200:  speed = B115200; break;
            case 230400:  speed = B230400; break;
            case 460800:  speed = B460800; break;
            case 921600:  speed = B921600; break;
            case 1000000: speed = B1000000; break;
            case 1152000: speed = B1152000; break;
            case 1500000: speed = B1500000; break;
            case 2000000: speed = B2000000; break;
            case 2500000: speed = B2500000; break;
            case 3000000: speed = B3000000; break;
            default:
                std::cerr << "[MAVLink] Unsupported baud rate: " << baud_rate
                         << ", using 57600" << std::endl;
                speed = B57600;
        }

        cfsetospeed(&tty, speed);
        cfsetispeed(&tty, speed);

        // 8N1 mode
        tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
        tty.c_cflag |= (CLOCAL | CREAD);
        tty.c_cflag &= ~(PARENB | PARODD);
        tty.c_cflag &= ~CSTOPB;
        tty.c_cflag &= ~CRTSCTS;

        tty.c_lflag = 0;
        tty.c_iflag &= ~(IXON | IXOFF | IXANY);
        tty.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL);
        tty.c_oflag = 0;

        tty.c_cc[VMIN]  = 0;
        tty.c_cc[VTIME] = 1;

        if (tcsetattr(serial_fd, TCSANOW, &tty) != 0) {
            std::cerr << "[MAVLink] Error setting serial port attributes: "
                     << strerror(errno) << std::endl;
            close(serial_fd);
            exit(1);
        }

        tcflush(serial_fd, TCIOFLUSH);

        std::cout << "[MAVLink] Interface initialized on " << serial_port
                  << " @ " << baud_rate << " baud" << std::endl;
    }

    ~MAVLinkInterface() {
        stop();
        close(serial_fd);
    }

    void start() {
        if (!running) {
            running = true;
            mavlink_thread = std::thread(&MAVLinkInterface::run, this);
            std::cout << "[MAVLink] Thread started" << std::endl;
        }
    }

    void stop() {
        if (running) {
            running = false;
            if (mavlink_thread.joinable()) {
                mavlink_thread.join();
            }
            std::cout << "[MAVLink] Thread stopped" << std::endl;
        }
    }

    void queueOdometry(uint64_t time_usec,
                       float x, float y, float z,
                       const float q[4],
                       float vx, float vy, float vz,
                       float rollspeed, float pitchspeed, float yawspeed,
                       const float pose_covariance[21],
                       const float velocity_covariance[21],
                       uint8_t reset_counter,
                       uint8_t estimator_type,
                       int8_t quality) {
        OdometryData data;
        data.time_usec = time_usec;
        data.x = x;
        data.y = y;
        data.z = z;
        memcpy(data.q, q, sizeof(data.q));
        data.vx = vx;
        data.vy = vy;
        data.vz = vz;
        data.rollspeed = rollspeed;
        data.pitchspeed = pitchspeed;
        data.yawspeed = yawspeed;
        memcpy(data.pose_covariance, pose_covariance, sizeof(data.pose_covariance));
        memcpy(data.velocity_covariance, velocity_covariance, sizeof(data.velocity_covariance));
        data.reset_counter = reset_counter;
        data.estimator_type = estimator_type;
        data.quality = quality;

        odometry_queue.push(data);
    }

    size_t getQueueSize() const {
        return const_cast<OdometryQueue&>(odometry_queue).size();
    }

    bool isConnected() const {
        return connected;
    }

    bool waitForConnection(int timeout_seconds = 10) {
        std::cout << "[MAVLink] Waiting for connection";
        std::cout.flush();

        std::unique_lock<std::mutex> lock(connected_mutex);

        for (int i = 0; i < timeout_seconds; i++) {
            if (connected_cv.wait_for(lock, std::chrono::seconds(1),
                                     [this] { return connected.load(); })) {
                std::cout << " Connected!" << std::endl;
                return true;
            }
            std::cout << ".";
            std::cout.flush();
        }

        std::cout << " Timeout!" << std::endl;
        return false;
    }

    // Get EKF position feedback (thread-safe)
    EKFPositionFeedback getEKFPosition() const {
        std::lock_guard<std::mutex> lock(ekf_mutex);
        return ekf_feedback;
    }

    // Check if EKF feedback is recent
    bool hasRecentEKFFeedback(uint32_t timeout_ms = EKF_FEEDBACK_TIMEOUT_MS) const {
        std::lock_guard<std::mutex> lock(ekf_mutex);
        if (!ekf_feedback.valid) return false;
        uint32_t age = getCurrentTimeMs() - ekf_feedback.timestamp_ms;
        return age < timeout_ms;
    }

private:
    void run() {
        requestDataStream();

        int heartbeat_counter = 0;

        while (running) {
            // Send heartbeat every second (~100 iterations * 10ms)
            if (heartbeat_counter % 100 == 0) {
                sendHeartbeat();
            }

            // Check for odometry data in queue and send
            OdometryData odom_data;
            if (odometry_queue.pop(odom_data, 10)) {
                sendData(odom_data);
            }

            // Receive and process messages
            receiveMessages();

            usleep(50);
            heartbeat_counter++;
        }
    }

    void sendHeartbeat() {
        mavlink_message_t msg;
        uint8_t buf[MAVLINK_MAX_PACKET_LEN];

        mavlink_msg_heartbeat_pack(system_id, component_id, &msg,
                                   MAV_TYPE_ONBOARD_CONTROLLER,
                                   MAV_AUTOPILOT_INVALID,
                                   0, 0, MAV_STATE_ACTIVE);

        uint16_t len = mavlink_msg_to_send_buffer(buf, &msg);
        ssize_t sent = write(serial_fd, buf, len);

        if (sent < 0) {
            std::cerr << "[MAVLink] Failed to send heartbeat: " << strerror(errno) << std::endl;
        }
    }

    void requestDataStream() {
        mavlink_message_t msg;
        uint8_t buf[MAVLINK_MAX_PACKET_LEN];

        // Request LOCAL_POSITION_NED stream
        mavlink_msg_request_data_stream_pack(system_id, component_id, &msg,
                                            1, 1,
                                            MAV_DATA_STREAM_POSITION,
                                            10,  // 10 Hz
                                            1);

        uint16_t len = mavlink_msg_to_send_buffer(buf, &msg);
        write(serial_fd, buf, len);

        std::cout << "[MAVLink] Requested LOCAL_POSITION_NED stream" << std::endl;
    }

    void sendOdometry(const OdometryData& data) {
        mavlink_message_t msg;
        uint8_t buf[MAVLINK_MAX_PACKET_LEN];

        mavlink_msg_odometry_pack(
            system_id, component_id, &msg,
            data.time_usec,
            MAV_FRAME_LOCAL_NED,
            MAV_FRAME_BODY_FRD,
            data.x, data.y, data.z,
            data.q,
            data.vx, data.vy, data.vz,
            data.rollspeed, data.pitchspeed, data.yawspeed,
            data.pose_covariance,
            data.velocity_covariance,
            data.reset_counter,
            data.estimator_type,
            data.quality
        );

        uint16_t len = mavlink_msg_to_send_buffer(buf, &msg);
        ssize_t sent = write(serial_fd, buf, len);

        if (sent < 0) {
            std::cerr << "[MAVLink] Failed to send odometry: " << strerror(errno) << std::endl;
        }
    }

    void sendVisionPositionEstimate(const OdometryData& data) {
        mavlink_message_t msg;
        memset(&msg, 0, sizeof(msg));
        uint8_t buf[MAVLINK_MAX_PACKET_LEN];
        memset(buf, 0, sizeof(buf));

        // Validate inputs
        if (!std::isfinite(data.q[0]) || !std::isfinite(data.q[1]) ||
            !std::isfinite(data.q[2]) || !std::isfinite(data.q[3])) {
            std::cerr << "[MAVLink] Invalid quaternion values" << std::endl;
            return;
        }

        if (!std::isfinite(data.x) || !std::isfinite(data.y) || !std::isfinite(data.z)) {
            std::cerr << "[MAVLink] Invalid position values" << std::endl;
            return;
        }

        // Calculate yaw from quaternion
        float qw = data.q[0], qx = data.q[1], qy = data.q[2], qz = data.q[3];
        float yaw = atan2(2.0f * (qw * qz + qx * qy), 1.0f - 2.0f * (qy * qy + qz * qz));

        mavlink_msg_vision_position_estimate_pack(
            system_id, component_id, &msg,
            data.time_usec,
            data.x, data.y, data.z,
            0.0f, 0.0f, yaw,
            data.pose_covariance,
            data.reset_counter
        );

        uint16_t len = mavlink_msg_to_send_buffer(buf, &msg);
        ssize_t sent = write(serial_fd, buf, len);

        if (sent < 0) {
            std::cerr << "[MAVLink] Failed to send vision position estimate: " << strerror(errno) << std::endl;
        }
    }

    void sendVisionSpeedEstimate(const OdometryData& data) {
        mavlink_message_t msg;
        memset(&msg, 0, sizeof(msg));
        uint8_t buf[MAVLINK_MAX_PACKET_LEN];
        memset(buf, 0, sizeof(buf));

        if (!std::isfinite(data.vx) || !std::isfinite(data.vy) || !std::isfinite(data.vz)) {
            std::cerr << "[MAVLink] Invalid velocity values" << std::endl;
            return;
        }

        mavlink_msg_vision_speed_estimate_pack(
            system_id, component_id, &msg,
            data.time_usec,
            data.vx, data.vy, data.vz,
            nullptr,
            data.reset_counter
        );

        uint16_t len = mavlink_msg_to_send_buffer(buf, &msg);
        ssize_t sent = write(serial_fd, buf, len);

        if (sent < 0) {
            std::cerr << "[MAVLink] Failed to send vision speed estimate: " << strerror(errno) << std::endl;
        }
    }

    void sendData(const OdometryData& data) {
        switch (tx_mode) {
            case MAVLinkMode::ODOMETRY:
                sendOdometry(data);
                break;
            case MAVLinkMode::VISION_POSITION_ESTIMATE:
                sendVisionPositionEstimate(data);
                break;
            case MAVLinkMode::VISION_POSITION_AND_SPEED:
                sendVisionPositionEstimate(data);
                sendVisionSpeedEstimate(data);
                break;
        }
    }

    void receiveMessages() {
        uint8_t buf[MAVLINK_MAX_PACKET_LEN];
        mavlink_message_t msg;
        mavlink_status_t status;

        ssize_t recsize = read(serial_fd, buf, sizeof(buf));

        if (recsize > 0) {
            for (int i = 0; i < recsize; i++) {
                if (mavlink_parse_char(MAVLINK_COMM_0, buf[i], &msg, &status)) {
                    handleMessage(msg);
                }
            }
        }
    }

    void handleMessage(const mavlink_message_t& msg) {
        switch (msg.msgid) {
            case MAVLINK_MSG_ID_HEARTBEAT: {
                mavlink_heartbeat_t heartbeat;
                mavlink_msg_heartbeat_decode(&msg, &heartbeat);

                if (msg.sysid != system_id && heartbeat.autopilot != MAV_AUTOPILOT_INVALID) {
                    if (!connected) {
                        system_id = msg.sysid;
                        component_id = MAV_COMP_ID_VISUAL_INERTIAL_ODOMETRY;

                        std::lock_guard<std::mutex> lock(connected_mutex);
                        connected = true;
                        connected_cv.notify_all();

                        std::cout << "[MAVLink] Connected to system " << (int)msg.sysid << std::endl;
                    }
                }
                break;
            }

            // EKF Position Feedback - key addition for coordinate bridging
            case MAVLINK_MSG_ID_LOCAL_POSITION_NED: {
                mavlink_local_position_ned_t pos;
                mavlink_msg_local_position_ned_decode(&msg, &pos);

                {
                    std::lock_guard<std::mutex> lock(ekf_mutex);
                    ekf_feedback.position = Eigen::Vector3d(pos.x, pos.y, pos.z);
                    ekf_feedback.velocity = Eigen::Vector3d(pos.vx, pos.vy, pos.vz);
                    ekf_feedback.timestamp_ms = getCurrentTimeMs();
                    ekf_feedback.valid = true;
                }

                // Log occasionally
                static int log_counter = 0;
                if (++log_counter % 100 == 0) {
                    std::cout << "[EKF Feedback] Position: (" << std::fixed << std::setprecision(2)
                              << pos.x << ", " << pos.y << ", " << pos.z << ")" << std::endl;
                }
                break;
            }

            case MAVLINK_MSG_ID_ATTITUDE: {
                // Logged for debugging
                break;
            }

            default:
                break;
        }
    }
};

//=============================================================================
// VIO Bridge State Structure
//=============================================================================

struct VIOBridgeState {
    // Coordinate transformation
    Eigen::Vector3d vio_to_ned_offset;
    Eigen::Quaterniond vio_to_ned_rotation;

    // Map tracking (heuristic-based)
    int tracking_state_current;
    int tracking_state_previous;
    std::set<int> map_history;

    // EKF feedback
    Eigen::Vector3d last_ekf_position;
    uint32_t last_ekf_update_ms;

    // Last valid pose (for RECENTLY_LOST handling)
    Eigen::Vector3d last_valid_ned_position;
    Eigen::Quaterniond last_valid_ned_orientation;
    Eigen::Vector3d last_valid_vio_position;  // For map change detection

    // Reset counter management
    uint8_t reset_counter;

    // Safety flags
    bool first_pose_received;
    bool vio_initialized;
    uint32_t last_pose_time_ms;

    // Health metrics
    uint32_t messages_sent;
    uint32_t map_transitions;
    uint32_t tracking_loss_events;

    VIOBridgeState()
        : vio_to_ned_offset(Eigen::Vector3d::Zero())
        , vio_to_ned_rotation(Eigen::Quaterniond::Identity())
        , tracking_state_current(-1)
        , tracking_state_previous(-1)
        , last_ekf_position(Eigen::Vector3d::Zero())
        , last_ekf_update_ms(0)
        , last_valid_ned_position(Eigen::Vector3d::Zero())
        , last_valid_ned_orientation(Eigen::Quaterniond::Identity())
        , last_valid_vio_position(Eigen::Vector3d::Zero())
        , reset_counter(0)
        , first_pose_received(false)
        , vio_initialized(false)
        , last_pose_time_ms(0)
        , messages_sent(0)
        , map_transitions(0)
        , tracking_loss_events(0)
    {}
};

//=============================================================================
// VIO Bridge Class - Coordinate Transform Layer
//=============================================================================

class VIOBridge {
private:
    VIOBridgeState state;
    std::shared_ptr<MAVLinkInterface> mavlink;
    uint64_t start_time_us;

    // IMU data for angular velocity logging
    struct IMUData {
        double timestamp;
        Eigen::Vector3f accel;
        Eigen::Vector3f gyro;
        bool valid;

        IMUData() : timestamp(0), accel(0,0,0), gyro(0,0,0), valid(false) {}
    } latest_imu;

public:
    VIOBridge(const std::string& serial_port, int baud_rate,
              MAVLinkMode mode = MAVLinkMode::VISION_POSITION_ESTIMATE) {
        // Get system start time
        auto now = std::chrono::high_resolution_clock::now();
        start_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
            now.time_since_epoch()).count();

        std::cout << "==================================================" << std::endl;
        std::cout << "[VIOBridge v2] Initializing with coordinate bridging" << std::endl;
        std::cout << "==================================================" << std::endl;

        std::cout << "[VIOBridge] MAVLink mode: ";
        switch(mode) {
            case MAVLinkMode::ODOMETRY:
                std::cout << "ODOMETRY" << std::endl;
                break;
            case MAVLinkMode::VISION_POSITION_ESTIMATE:
                std::cout << "VISION_POSITION_ESTIMATE" << std::endl;
                break;
            case MAVLinkMode::VISION_POSITION_AND_SPEED:
                std::cout << "VISION_POSITION_ESTIMATE + VISION_SPEED_ESTIMATE" << std::endl;
                break;
        }

        // Initialize MAVLink with parameterized serial port and baud rate
        mavlink = std::make_shared<MAVLinkInterface>(serial_port.c_str(), baud_rate, mode);
        mavlink->start();
    }

    ~VIOBridge() {
        if (mavlink) {
            mavlink->stop();
        }

        std::cout << "==================================================" << std::endl;
        std::cout << "[VIOBridge] Shutdown statistics:" << std::endl;
        std::cout << "  Messages sent: " << state.messages_sent << std::endl;
        std::cout << "  Map transitions: " << state.map_transitions << std::endl;
        std::cout << "  Tracking loss events: " << state.tracking_loss_events << std::endl;
        std::cout << "==================================================" << std::endl;
    }

    bool waitForConnection(int timeout_seconds = 30) {
        if (!mavlink->waitForConnection(timeout_seconds)) {
            std::cerr << "[VIOBridge] Failed to connect to ArduPilot!" << std::endl;
            mavlink->stop();
            return false;
        }
        return true;
    }

    void updateIMUData(double timestamp, const Eigen::Vector3f& accel, const Eigen::Vector3f& gyro) {
        latest_imu.timestamp = timestamp;
        latest_imu.accel = accel;
        latest_imu.gyro = gyro;
        latest_imu.valid = true;
    }

    /**
     * Main processing function - called for each ORB-SLAM3 pose
     */
    void processORBSLAMPose(const Sophus::SE3f& Tcw,
                           int tracking_state,
                           const Eigen::Vector3f& velocity) {

        uint32_t current_time_ms = getCurrentTimeMs();

        // Update tracking state history
        state.tracking_state_previous = state.tracking_state_current;
        state.tracking_state_current = tracking_state;

        // Extract VIO pose (inverse of Tcw gives camera position in world frame)
        Sophus::SE3f Twc = Tcw.inverse();
        Eigen::Vector3f position_f = Twc.translation();
        Eigen::Quaternionf orientation_f = Twc.unit_quaternion();

        // Convert to double precision for transformations
        Eigen::Vector3d vio_position = position_f.cast<double>();
        Eigen::Quaterniond vio_orientation = orientation_f.cast<double>();

        // Convert velocity to double
        Eigen::Vector3d vio_velocity = velocity.cast<double>();

        // Handle based on tracking state
        switch (tracking_state) {
            case ORB_SLAM3::Tracking::OK:
            case ORB_SLAM3::Tracking::OK_KLT:
                handleTrackingOK(vio_position, vio_orientation, vio_velocity, current_time_ms);
                break;

            case ORB_SLAM3::Tracking::RECENTLY_LOST:
                handleRecentlyLost(current_time_ms);
                break;

            case ORB_SLAM3::Tracking::LOST:
            case ORB_SLAM3::Tracking::NOT_INITIALIZED:
            case ORB_SLAM3::Tracking::SYSTEM_NOT_READY:
            case ORB_SLAM3::Tracking::NO_IMAGES_YET:
                handleTrackingLost();
                return;  // Don't send any messages
        }
    }

    /**
     * Health check function
     */
    bool performHealthCheck() {
        bool healthy = true;
        uint32_t current_time = getCurrentTimeMs();

        // Check 1: Recent pose data
        if (state.last_pose_time_ms > 0) {
            uint32_t pose_age = current_time - state.last_pose_time_ms;
            if (pose_age > 500) {
                std::cerr << "[VIOBridge] WARNING: No pose data for " << pose_age << "ms" << std::endl;
                healthy = false;
            }
        }

        // Check 2: EKF feedback (warning only, not critical)
        if (!mavlink->hasRecentEKFFeedback()) {
            static int warning_count = 0;
            if (++warning_count % 100 == 0) {
                std::cout << "[VIOBridge] INFO: No recent EKF feedback" << std::endl;
            }
        }

        // Check 3: Position error (if we have EKF feedback)
        if (mavlink->hasRecentEKFFeedback() && state.first_pose_received) {
            EKFPositionFeedback ekf = mavlink->getEKFPosition();
            double position_error = (state.last_valid_ned_position - ekf.position).norm();
            if (position_error > 5.0) {
                std::cerr << "[VIOBridge] WARNING: Large position error: "
                          << std::fixed << std::setprecision(2) << position_error << "m" << std::endl;
            }
        }

        return healthy;
    }

    /**
     * Get statistics
     */
    uint32_t getMessagesSent() const { return state.messages_sent; }
    uint32_t getMapTransitions() const { return state.map_transitions; }
    uint32_t getTrackingLossEvents() const { return state.tracking_loss_events; }

private:
    /**
     * Handle normal tracking (OK or OK_KLT)
     */
    void handleTrackingOK(const Eigen::Vector3d& vio_position,
                         const Eigen::Quaterniond& vio_orientation,
                         const Eigen::Vector3d& vio_velocity,
                         uint32_t current_time_ms) {

        // Check for map change using heuristics
        bool map_changed = detectMapChange(vio_position);

        if (map_changed) {
            handleMapTransition(vio_position, vio_orientation);
        }

        // Apply coordinate transformation
        Eigen::Vector3d ned_position = applyCoordinateTransform(vio_position);
        Eigen::Quaterniond ned_orientation = applyRotationTransform(vio_orientation);
        Eigen::Vector3d ned_velocity = applyVelocityTransform(vio_velocity);

        // Save as last valid pose
        state.last_valid_ned_position = ned_position;
        state.last_valid_ned_orientation = ned_orientation;
        state.last_valid_vio_position = vio_position;

        // Update EKF feedback position
        updateEKFPosition();

        // Send to ArduPilot with normal covariance and quality
        sendVisionPositionEstimate(ned_position, ned_orientation, ned_velocity,
                                   current_time_ms, NORMAL_POSITION_COVARIANCE, QUALITY_GOOD);

        state.last_pose_time_ms = current_time_ms;
        state.messages_sent++;
    }

    /**
     * Handle recently lost tracking
     * Continue sending last valid position with high covariance
     */
    void handleRecentlyLost(uint32_t current_time_ms) {
        // Log tracking loss event (once per loss sequence)
        static bool was_tracking = true;
        if (was_tracking) {
            std::cout << "[VIOBridge] RECENTLY_LOST - sending high-uncertainty estimate" << std::endl;
            state.tracking_loss_events++;
            was_tracking = false;
        }

        // Send last valid position with inflated covariance
        sendVisionPositionEstimate(state.last_valid_ned_position,
                                   state.last_valid_ned_orientation,
                                   Eigen::Vector3d::Zero(),  // Zero velocity when lost
                                   current_time_ms,
                                   RECENTLY_LOST_COVARIANCE,
                                   QUALITY_RECENTLY_LOST);

        state.messages_sent++;
    }

    /**
     * Handle complete tracking loss
     * Stop sending messages - let EKF timeout trigger failsafe
     */
    void handleTrackingLost() {
        static bool logged = false;
        static int last_state = -1;

        if (!logged || last_state != state.tracking_state_current) {
            std::cout << "[VIOBridge] TRACKING LOST (state=" << state.tracking_state_current
                      << ") - suspending vision messages. EKF will dead-reckon." << std::endl;
            logged = true;
            last_state = state.tracking_state_current;
        }

        // Do NOT send any position messages
        // Do NOT increment reset_counter
        // Let EKF failsafe handle it
    }

    /**
     * Detect map change using heuristic (position discontinuity after tracking recovery)
     */
    bool detectMapChange(const Eigen::Vector3d& new_vio_position) {
        // Case 1: First pose ever - not a map change
        if (!state.first_pose_received) {
            state.first_pose_received = true;
            state.vio_initialized = true;
            std::cout << "[VIOBridge] VIO initialized with first pose" << std::endl;
            return false;
        }

        // Case 2: Transition from non-tracking to tracking
        bool was_lost = (state.tracking_state_previous == ORB_SLAM3::Tracking::LOST ||
                        state.tracking_state_previous == ORB_SLAM3::Tracking::NOT_INITIALIZED ||
                        state.tracking_state_previous == ORB_SLAM3::Tracking::RECENTLY_LOST ||
                        state.tracking_state_previous == -1);

        bool now_tracking = (state.tracking_state_current == ORB_SLAM3::Tracking::OK ||
                            state.tracking_state_current == ORB_SLAM3::Tracking::OK_KLT);

        if (was_lost && now_tracking) {
            // Check for position discontinuity
            double pos_jump = (new_vio_position - state.last_valid_vio_position).norm();

            if (pos_jump > MAP_CHANGE_THRESHOLD) {
                std::cout << "[VIOBridge] Map change detected! Position jump: "
                          << std::fixed << std::setprecision(2) << pos_jump << "m" << std::endl;
                return true;
            } else {
                std::cout << "[VIOBridge] Tracking recovered in same map (jump: "
                          << std::fixed << std::setprecision(2) << pos_jump << "m)" << std::endl;
            }
        }

        return false;
    }

    /**
     * Handle map transition - calculate new offset to maintain position continuity
     */
    void handleMapTransition(const Eigen::Vector3d& new_vio_position,
                            const Eigen::Quaterniond& new_vio_orientation) {

        state.map_transitions++;

        // Get current EKF position (where ArduPilot thinks we are)
        Eigen::Vector3d current_ekf_position = state.last_ekf_position;

        // Check if we have recent EKF feedback
        if (mavlink->hasRecentEKFFeedback()) {
            EKFPositionFeedback ekf = mavlink->getEKFPosition();
            current_ekf_position = ekf.position;

            std::cout << "[VIOBridge] Using EKF feedback for offset: ("
                      << std::fixed << std::setprecision(2)
                      << current_ekf_position.x() << ", "
                      << current_ekf_position.y() << ", "
                      << current_ekf_position.z() << ")" << std::endl;
        } else {
            // Fall back to last valid NED position
            current_ekf_position = state.last_valid_ned_position;

            std::cout << "[VIOBridge] No recent EKF feedback, using last NED position: ("
                      << std::fixed << std::setprecision(2)
                      << current_ekf_position.x() << ", "
                      << current_ekf_position.y() << ", "
                      << current_ekf_position.z() << ")" << std::endl;
        }

        // Calculate new offset: EKF_position - VIO_position (in NED frame)
        // First transform VIO position to NED frame (without current offset)
        Eigen::Vector3d vio_in_ned = transformVIOtoNED(new_vio_position);

        // New offset ensures transformed VIO matches EKF position
        state.vio_to_ned_offset = current_ekf_position - vio_in_ned;

        std::cout << "[VIOBridge] *** MAP TRANSITION #" << state.map_transitions << " ***" << std::endl;
        std::cout << "  New VIO position (raw): (" << std::fixed << std::setprecision(2)
                  << new_vio_position.x() << ", " << new_vio_position.y() << ", "
                  << new_vio_position.z() << ")" << std::endl;
        std::cout << "  New offset: (" << state.vio_to_ned_offset.x() << ", "
                  << state.vio_to_ned_offset.y() << ", " << state.vio_to_ned_offset.z() << ")" << std::endl;

        // CRITICAL: Do NOT increment reset_counter for new map creation
        // Only increment for valid relocalization to a merged map
        // Since we're using heuristic detection, we don't know if maps are merged
        // So we never increment reset_counter (conservative approach)

        std::cout << "  reset_counter unchanged at: " << (int)state.reset_counter << std::endl;
    }

    /**
     * Transform VIO position to NED frame (camera frame to NED)
     * ORB-SLAM3: X-right, Y-down, Z-forward
     * NED: X-north, Y-east, Z-down
     */
    Eigen::Vector3d transformVIOtoNED(const Eigen::Vector3d& vio_position) {
        // Apply frame rotation
        // Forward (Z) -> North (X)
        // Right (X) -> East (Y), negated
        // Down (Y) -> Down (Z)
        return Eigen::Vector3d(
            vio_position.z(),   // Z -> X (forward -> north)
            -vio_position.x(),  // -X -> Y (right -> west, negate for east)
            vio_position.y()    // Y -> Z (down -> down)
        );
    }

    /**
     * Apply full coordinate transformation (frame rotation + offset)
     */
    Eigen::Vector3d applyCoordinateTransform(const Eigen::Vector3d& vio_position) {
        // First apply frame rotation
        Eigen::Vector3d ned_position = transformVIOtoNED(vio_position);

        // Then apply offset
        return ned_position + state.vio_to_ned_offset;
    }

    /**
     * Apply rotation transformation
     */
    Eigen::Quaterniond applyRotationTransform(const Eigen::Quaterniond& vio_orientation) {
        // Apply vio_to_ned_rotation if needed
        return state.vio_to_ned_rotation * vio_orientation;
    }

    /**
     * Transform velocity to NED frame
     */
    Eigen::Vector3d applyVelocityTransform(const Eigen::Vector3d& vio_velocity) {
        // Same frame rotation as position
        return Eigen::Vector3d(
            vio_velocity.z(),
            -vio_velocity.x(),
            vio_velocity.y()
        );
    }

    /**
     * Update EKF position from MAVLink feedback
     */
    void updateEKFPosition() {
        if (mavlink->hasRecentEKFFeedback()) {
            EKFPositionFeedback ekf = mavlink->getEKFPosition();
            state.last_ekf_position = ekf.position;
            state.last_ekf_update_ms = ekf.timestamp_ms;
        }
    }

    /**
     * Send VISION_POSITION_ESTIMATE message via MAVLink
     */
    void sendVisionPositionEstimate(const Eigen::Vector3d& ned_position,
                                   const Eigen::Quaterniond& ned_orientation,
                                   const Eigen::Vector3d& ned_velocity,
                                   uint32_t current_time_ms,
                                   float position_covariance,
                                   int8_t quality) {

        // Convert timestamp to microseconds
        uint64_t timestamp_us = start_time_us + static_cast<uint64_t>(current_time_ms * 1000);

        // Build quaternion array
        float q[4] = {
            static_cast<float>(ned_orientation.w()),
            static_cast<float>(ned_orientation.x()),
            static_cast<float>(ned_orientation.y()),
            static_cast<float>(ned_orientation.z())
        };

        // Build covariance arrays
        float pose_covariance[21] = {0};
        pose_covariance[0] = position_covariance * position_covariance;   // x variance
        pose_covariance[6] = position_covariance * position_covariance;   // y variance
        pose_covariance[11] = position_covariance * position_covariance;  // z variance
        pose_covariance[15] = NORMAL_ANGLE_COVARIANCE * NORMAL_ANGLE_COVARIANCE;  // roll
        pose_covariance[18] = NORMAL_ANGLE_COVARIANCE * NORMAL_ANGLE_COVARIANCE;  // pitch
        pose_covariance[20] = NORMAL_ANGLE_COVARIANCE * NORMAL_ANGLE_COVARIANCE;  // yaw

        float velocity_covariance[21] = {0};
        velocity_covariance[0] = 0.01f;
        velocity_covariance[6] = 0.01f;
        velocity_covariance[11] = 0.01f;

        // Queue odometry data
        mavlink->queueOdometry(
            timestamp_us,
            static_cast<float>(ned_position.x()),
            static_cast<float>(ned_position.y()),
            static_cast<float>(ned_position.z()),
            q,
            static_cast<float>(ned_velocity.x()),
            static_cast<float>(ned_velocity.y()),
            static_cast<float>(ned_velocity.z()),
            latest_imu.valid ? latest_imu.gyro.x() : 0.0f,
            latest_imu.valid ? latest_imu.gyro.y() : 0.0f,
            latest_imu.valid ? latest_imu.gyro.z() : 0.0f,
            pose_covariance,
            velocity_covariance,
            state.reset_counter,
            MAV_ESTIMATOR_TYPE_VIO,
            quality
        );

        // Log periodically
        static int log_counter = 0;
        if (++log_counter % 30 == 0) {
            std::cout << "[VIO->NED] pos=(" << std::fixed << std::setprecision(2)
                      << ned_position.x() << "," << ned_position.y() << "," << ned_position.z()
                      << ") cov=" << position_covariance << " q=" << (int)quality
                      << " rst=" << (int)state.reset_counter << std::endl;
        }
    }
};

//=============================================================================
// Main Function
//=============================================================================

int main(int argc, char** argv) {
    if (argc < 3 || argc > 8) {
        cerr << endl
             << "Usage: ./mono_inertial_pear path_to_vocabulary path_to_settings"
             << endl
             << "           [imu_serial_port] [mavlink_serial_port] [mavlink_baud] [mode] [visualization]"
             << endl
             << endl
             << "  Defaults:"
             << endl
             << "    imu_serial_port:     /dev/ttyACM0"
             << endl
             << "    mavlink_serial_port: /dev/ttyAMA0"
             << endl
             << "    mavlink_baud:        1500000"
             << endl
             << "    mode: 0 = ODOMETRY"
             << endl
             << "          1 = VISION_POSITION_ESTIMATE (default)"
             << endl
             << "          2 = VISION_POSITION_ESTIMATE + VISION_SPEED_ESTIMATE"
             << endl
             << "    visualization: 0 = OFF (default), 1 = ON (Pangolin viewer)"
             << endl;
        return 1;
    }

    string vocabularyPath = argv[1];
    string settingsPath = argv[2];
    string imuSerialPort = (argc >= 4) ? argv[3] : "/dev/ttyACM0";
    string mavlinkSerialPort = (argc >= 5) ? argv[4] : "/dev/ttyAMA0";
    int mavlinkBaud = (argc >= 6) ? atoi(argv[5]) : 1500000;

    // Parse MAVLink mode
    MAVLinkMode mavlink_mode = MAVLinkMode::VISION_POSITION_ESTIMATE;
    if (argc >= 7) {
        int mode_val = atoi(argv[6]);
        if (mode_val == 0) {
            mavlink_mode = MAVLinkMode::ODOMETRY;
        } else if (mode_val == 1) {
            mavlink_mode = MAVLinkMode::VISION_POSITION_ESTIMATE;
        } else if (mode_val == 2) {
            mavlink_mode = MAVLinkMode::VISION_POSITION_AND_SPEED;
        } else {
            cerr << "Invalid mode: " << mode_val << ". Using VISION_POSITION_ESTIMATE." << endl;
        }
    }

    // Parse visualization option
    bool enable_visualization = false;
    if (argc >= 8) {
        int vis_val = atoi(argv[7]);
        enable_visualization = (vis_val != 0);
    }

    // Setup signal handler
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = exit_loop_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);

    cout << "========================================" << endl;
    cout << "Monocular-Inertial VIO (PearAPI + MAVLink)" << endl;
    cout << "========================================" << endl;
    cout << "Vocabulary:      " << vocabularyPath << endl;
    cout << "Settings:        " << settingsPath << endl;
    cout << "IMU Port:        " << imuSerialPort << endl;
    cout << "MAVLink Port:    " << mavlinkSerialPort << endl;
    cout << "MAVLink Baud:    " << mavlinkBaud << endl;
    cout << "MAVLink Mode:    " << static_cast<int>(mavlink_mode)
         << " (0=ODOM, 1=VISION_POS, 2=VISION_POS+SPEED)" << endl;
    cout << "Visualization:   " << (enable_visualization ? "ON" : "OFF") << endl;
    cout << "========================================" << endl;

    // ---- Initialize IMU reader using PearAPI ----
    pearvio::IMUReader imuReader;
    if (!imuReader.open(imuSerialPort)) {
        cerr << "Failed to open IMU serial port: " << imuSerialPort << endl;
        cerr << "Check that:" << endl;
        cerr << "  - Pico is connected and running" << endl;
        cerr << "  - User has dialout group access: sudo usermod -aG dialout $USER" << endl;
        return 1;
    }

    // ---- Initialize camera using PearAPI ----
    auto camera = pearvio::CameraBackend::create();
    if (!camera) {
        cerr << "Failed to create camera backend" << endl;
        return 1;
    }

    pearvio::CameraConfig camConfig;
    camConfig.cameraIndex = 0;
    camConfig.width = 640;
    camConfig.height = 400;
    camConfig.fps = 20;  // 20 FPS for triggered mode

    // Load camera settings from PearCameraApp config (exposure, gain, etc.)
    if (camConfig.loadFromIniFile()) {
        cout << "Loaded camera settings from config file" << endl;
    } else {
        cout << "No config file found, using default camera settings" << endl;
    }

    // VIO always needs trigger mode for hardware sync
    camConfig.triggerMode = true;

    if (!camera->initialize(camConfig)) {
        cerr << "Failed to initialize camera" << endl;
        return 1;
    }

    cout << "Camera initialized: " << camera->frameWidth() << "x" << camera->frameHeight() << endl;

    if (!camera->start()) {
        cerr << "Failed to start camera" << endl;
        return 1;
    }

    // ---- Wait for first IMU data to establish time base ----
    cout << "Waiting for IMU data..." << endl;
    uint64_t firstImuTimestamp = 0;
    while (b_continue_session && firstImuTimestamp == 0) {
        auto imuData = imuReader.getIMUData();
        if (!imuData.empty()) {
            firstImuTimestamp = imuData.front().timestamp_ms;
            cout << "First IMU timestamp: " << firstImuTimestamp << " ms" << endl;

            const auto& first = imuData.front();
            cout << "First IMU reading (raw):" << endl;
            cout << "  Accel: [" << first.accel_x() << ", " << first.accel_y() << ", " << first.accel_z() << "] g" << endl;
            cout << "  Gyro:  [" << first.gyro_x() << ", " << first.gyro_y() << ", " << first.gyro_z() << "] deg/s" << endl;

            float ax, ay, az;
            first.getAccelSI(ax, ay, az);
            cout << "  Accel (SI): [" << ax << ", " << ay << ", " << az << "] m/s^2" << endl;
        }
        this_thread::sleep_for(chrono::milliseconds(10));
    }

    if (!b_continue_session) {
        cout << "Interrupted before initialization" << endl;
        return 0;
    }

    // ---- Create SLAM system ----
    cout << "Creating ORB-SLAM3 system..." << endl;
    ORB_SLAM3::System SLAM(vocabularyPath, settingsPath, ORB_SLAM3::System::IMU_MONOCULAR, enable_visualization);
    float imageScale = SLAM.GetImageScale();

    // Calculate half exposure time for frame timestamp offset
    double halfExposureTimeSec = camera->exposureTime() / 2.0 / 1e6;
    cout << "Exposure time: " << camera->exposureTime() << " us" << endl;
    cout << "Half exposure time offset: " << halfExposureTimeSec * 1000.0 << " ms" << endl;

    // Clear IMU buffer accumulated during SLAM initialization (loading vocabulary takes time)
    cout << "Clearing IMU buffer accumulated during initialization..." << endl;
    imuReader.getIMUData();  // Discard accumulated IMU data
    imuReader.clearTriggerQueue();  // Discard ALL pending camera triggers

    // Wait for fresh IMU data to establish new time base
    firstImuTimestamp = 0;
    while (b_continue_session && firstImuTimestamp == 0) {
        auto imuData = imuReader.getIMUData();
        if (!imuData.empty()) {
            firstImuTimestamp = imuData.front().timestamp_ms;
            cout << "Reset time base to: " << firstImuTimestamp << " ms" << endl;
        }
        this_thread::sleep_for(chrono::milliseconds(5));
    }

    // Discard first few camera frames to ensure they have timestamps after our time base
    cout << "Waiting for camera frames with valid timestamps..." << endl;
    for (int i = 0; i < 5 && b_continue_session; i++) {
        cv::Mat frame;
        double ts;
        camera->getFrame(frame, ts);
        imuReader.getIMUData();  // Discard IMU
    }

    // Final cleanup: clear any remaining old triggers that accumulated during init
    imuReader.clearTriggerQueue();
    imuReader.getIMUData();  // Discard any remaining IMU data

    // ---- Create VIO Bridge for MAVLink/ArduPilot integration ----
    VIOBridge vio_bridge(mavlinkSerialPort, mavlinkBaud, mavlink_mode);

    // Wait for MAVLink connection (non-fatal timeout)
    if (!vio_bridge.waitForConnection(30)) {
        cout << "[Main] MAVLink connection timeout - continuing without ArduPilot." << endl;
        cout << "[Main] MAVLink will auto-connect when ArduPilot becomes available." << endl;
    } else {
        cout << "[Main] MAVLink connected!" << endl;
    }

    cout << "VIO system ready. Press Ctrl+C to exit." << endl;
    cout << "========================================" << endl;

    // Unit conversion constants
    constexpr float DEG_TO_RAD = 0.0174532925f;  // pi/180
    constexpr float G_TO_MS2 = 9.80665f;

    // IMU measurements for current frame
    vector<ORB_SLAM3::IMU::Point> vImuMeas;

    // Statistics
    uint64_t frameCount = 0;
    uint64_t imuCount = 0;
    auto startTime = chrono::steady_clock::now();

    // Track previous IMU timestamp for gap detection
    double lastImuTimestamp = 0;
    int imuGapWarnings = 0;

    // Health check timer
    uint32_t last_health_check = getCurrentTimeMs();
    const uint32_t HEALTH_CHECK_INTERVAL_MS = 5000;

    // ---- Main loop ----
    while (!SLAM.isShutDown() && b_continue_session) {
        // Get camera frame FIRST (this blocks until frame is ready)
        cv::Mat frame;
        double cameraTimestamp;

        if (!camera->getFrame(frame, cameraTimestamp)) {
            continue;
        }

        // Get Pico trigger timestamp
        // When processing is slower than camera rate, triggers accumulate in queue.
        // Drain ALL triggers and use only the LAST one to match the current frame.
        uint64_t triggerTimestamp = 0;
        int droppedTriggers = 0;
        while (imuReader.hasTriggerTimestamp()) {
            uint64_t ts = imuReader.getCameraTriggerTimestamp();
            if (triggerTimestamp > 0) {
                droppedTriggers++;
            }
            triggerTimestamp = ts;
        }
        if (droppedTriggers > 0 && frameCount < 50) {
            cout << "Note: Skipped " << droppedTriggers << " triggers (dropped frames during processing)" << endl;
        }

        // Check for invalid trigger (before our time base)
        if (triggerTimestamp > 0 && triggerTimestamp < firstImuTimestamp) {
            if (frameCount < 20) {
                cout << "Skipping frame with old trigger timestamp: " << triggerTimestamp
                     << " < " << firstImuTimestamp << endl;
            }
            continue;
        }

        // NOW get all IMU data (after camera frame, so we have data up to trigger time)
        auto imuData = imuReader.getIMUData();

        // Latest IMU sample for VIOBridge angular velocity (in SI units)
        Eigen::Vector3f latestAccelSI(0, 0, 0);
        Eigen::Vector3f latestGyroSI(0, 0, 0);

        for (const auto& imu : imuData) {
            // Convert to SI units and create IMU::Point
            double t = (imu.timestamp_ms - firstImuTimestamp) / 1000.0;

            // Check for gaps in IMU stream
            if (lastImuTimestamp > 0 && t > lastImuTimestamp) {
                double gap = t - lastImuTimestamp;
                if (gap > 0.005 && imuGapWarnings < 10) {  // More than 5ms gap
                    cout << "WARNING: IMU gap detected: " << fixed << setprecision(1)
                         << gap * 1000 << "ms at t=" << setprecision(3) << t << endl;
                    imuGapWarnings++;
                }
            }
            lastImuTimestamp = t;

            // Convert: gyro from deg/s to rad/s, accel from g to m/s^2
            float ax = imu.accel_x() * G_TO_MS2;
            float ay = imu.accel_y() * G_TO_MS2;
            float az = imu.accel_z() * G_TO_MS2;
            float gx = imu.gyro_x() * DEG_TO_RAD;
            float gy = imu.gyro_y() * DEG_TO_RAD;
            float gz = imu.gyro_z() * DEG_TO_RAD;

            ORB_SLAM3::IMU::Point pt(ax, ay, az, gx, gy, gz, t);
            vImuMeas.push_back(pt);
            imuCount++;

            // Track latest for VIOBridge
            latestAccelSI = Eigen::Vector3f(ax, ay, az);
            latestGyroSI = Eigen::Vector3f(gx, gy, gz);
        }

        // Determine frame time
        double frameTime;
        if (triggerTimestamp > 0 && triggerTimestamp >= firstImuTimestamp) {
            frameTime = (triggerTimestamp - firstImuTimestamp) / 1000.0 + halfExposureTimeSec;
        } else {
            if (!vImuMeas.empty()) {
                frameTime = vImuMeas.back().t;
                if (frameCount < 20) {
                    cout << "No trigger, using last IMU time: " << fixed << setprecision(3) << frameTime << endl;
                }
            } else {
                if (frameCount < 20) {
                    cout << "Skipping frame: no trigger and no IMU data" << endl;
                }
                continue;
            }
        }

        // Filter IMU measurements: only keep those with timestamp <= frameTime
        vector<ORB_SLAM3::IMU::Point> vImuForFrame;
        vector<ORB_SLAM3::IMU::Point> vImuForNext;

        for (const auto& pt : vImuMeas) {
            if (pt.t <= frameTime) {
                vImuForFrame.push_back(pt);
            } else {
                vImuForNext.push_back(pt);
            }
        }

        // Debug: Print timing info for first 20 frames
        if (frameCount < 20) {
            cout << "Frame " << frameCount << ": t=" << fixed << setprecision(3) << frameTime
                 << " IMU=" << vImuForFrame.size() << "/" << vImuMeas.size();
            if (!vImuForFrame.empty()) {
                cout << " range=[" << vImuForFrame.front().t << "," << vImuForFrame.back().t << "]";
                double imuFrameGap = frameTime - vImuForFrame.back().t;
                if (imuFrameGap > 0.005) {
                    cout << " GAP=" << setprecision(0) << imuFrameGap * 1000 << "ms";
                }
            }
            cout << " trigger=" << (triggerTimestamp > 0 ? "yes" : "NO");
            if (!vImuForNext.empty()) {
                cout << " kept=" << vImuForNext.size();
            }
            cout << endl;
        }

        // Keep IMU data after frame time for next iteration
        vImuMeas = std::move(vImuForNext);

        // Need at least some IMU data to track
        if (vImuForFrame.empty()) {
            if (frameCount < 20) {
                cout << "  -> Skipping frame: no IMU data" << endl;
            }
            continue;
        }

        // Resize if needed
        if (imageScale != 1.0f) {
            int newWidth = static_cast<int>(frame.cols * imageScale);
            int newHeight = static_cast<int>(frame.rows * imageScale);
            cv::resize(frame, frame, cv::Size(newWidth, newHeight));
        }

        // Track with ORB-SLAM3 monocular-inertial
        Sophus::SE3f Tcw = SLAM.TrackMonocular(frame, frameTime, vImuForFrame);

        // Get velocity and tracking state from ORB-SLAM3
        Eigen::Vector3f velocity = SLAM.GetVelocity();
        auto tracking_state = SLAM.GetTrackingState();

        // Update VIOBridge with latest IMU data (SI units: rad/s, m/s^2)
        if (!imuData.empty()) {
            vio_bridge.updateIMUData(lastImuTimestamp, latestAccelSI, latestGyroSI);
        }

        // Process pose through VIO Bridge (coordinate bridging + MAVLink happens here)
        vio_bridge.processORBSLAMPose(Tcw, tracking_state, velocity);

        frameCount++;

        // Periodic health check
        uint32_t current_time = getCurrentTimeMs();
        if (current_time - last_health_check > HEALTH_CHECK_INTERVAL_MS) {
            vio_bridge.performHealthCheck();
            last_health_check = current_time;
        }

        // Print IMU diagnostics every 50 frames
        if (frameCount % 50 == 0 && !vImuForFrame.empty()) {
            float maxGyro = 0;
            float avgAccelMag = 0;
            for (const auto& pt : vImuForFrame) {
                float gyroMag = pt.w.norm();
                if (gyroMag > maxGyro) maxGyro = gyroMag;
                avgAccelMag += pt.a.norm();
            }
            avgAccelMag /= vImuForFrame.size();
            cout << "IMU check: maxGyro=" << fixed << setprecision(2) << maxGyro * 57.3 << " deg/s"
                 << " | accelMag=" << setprecision(2) << avgAccelMag << " m/s^2"
                 << " (expect ~9.8 stationary)" << endl;
        }

        // Print statistics every 100 frames
        if (frameCount % 100 == 0) {
            auto now = chrono::steady_clock::now();
            double elapsed = chrono::duration<double>(now - startTime).count();
            double avgFps = frameCount / elapsed;

            // Instantaneous FPS (over the last 100 frames)
            static auto lastStatTime = startTime;
            static uint64_t lastStatFrameCount = 0;
            double windowElapsed = chrono::duration<double>(now - lastStatTime).count();
            double instantFps = (windowElapsed > 0) ? (frameCount - lastStatFrameCount) / windowElapsed : avgFps;
            lastStatTime = now;
            lastStatFrameCount = frameCount;

            const char* stateStr = "UNKNOWN";
            switch (tracking_state) {
                case -1: stateStr = "NOT_READY"; break;
                case 0: stateStr = "NO_IMAGES"; break;
                case 1: stateStr = "INIT"; break;
                case 2: stateStr = "OK"; break;
                case 3: stateStr = "RECENTLY_LOST"; break;
                case 4: stateStr = "LOST"; break;
                case 5: stateStr = "OK_KLT"; break;
            }

            cout << "Frames: " << frameCount
                 << " | IMU: " << imuCount
                 << " | FPS: " << fixed << setprecision(1) << instantFps
                 << " (avg: " << avgFps << ")"
                 << " | State: " << stateStr
                 << " | MAVLink sent: " << vio_bridge.getMessagesSent()
                 << " | Maps: " << vio_bridge.getMapTransitions()
                 << " | Time: " << fixed << setprecision(1) << elapsed << "s"
                 << endl;
        }
    }

    // ---- Cleanup ----
    cout << endl << "Shutting down..." << endl;

    // Disable trigger mode before stopping camera
    camera->setTriggerMode(false);

    camera->stop();
    imuReader.close();

    SLAM.Shutdown();

    // Final statistics
    auto endTime = chrono::steady_clock::now();
    double totalTime = chrono::duration<double>(endTime - startTime).count();

    cout << "========================================" << endl;
    cout << "Session complete" << endl;
    cout << "Total frames:      " << frameCount << endl;
    cout << "Total IMU:         " << imuCount << endl;
    cout << "Duration:          " << fixed << setprecision(1) << totalTime << " s" << endl;
    cout << "Average FPS:       " << fixed << setprecision(1) << (frameCount / totalTime) << endl;
    cout << "MAVLink sent:      " << vio_bridge.getMessagesSent() << endl;
    cout << "Map transitions:   " << vio_bridge.getMapTransitions() << endl;
    cout << "Tracking losses:   " << vio_bridge.getTrackingLossEvents() << endl;
    cout << "========================================" << endl;

    // Save trajectory
    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
    cout << "Trajectories saved to CameraTrajectory.txt and KeyFrameTrajectory.txt" << endl;

    return 0;
}
