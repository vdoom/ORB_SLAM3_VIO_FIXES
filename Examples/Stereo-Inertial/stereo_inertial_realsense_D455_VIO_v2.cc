/**
 * @file stereo_inertial_realsense_D455_VIO_v2.cc
 * @brief Refactored VIO implementation with coordinate frame bridging for ArduPilot integration
 *
 * This version implements:
 * - VIOBridgeState structure for centralized state management
 * - Coordinate transform layer for map transition handling
 * - EKF feedback loop via LOCAL_POSITION_NED
 * - Heuristic-based map change detection
 * - Proper tracking state handling (RECENTLY_LOST vs LOST)
 *
 * Key architectural change: Instead of passing coordinate frame discontinuities to ArduPilot,
 * we maintain a persistent offset between VIO coordinates and NED coordinates.
 *
 * NED_position = VIO_position + vio_to_ned_offset
 */

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include <iomanip>
#include <Eigen/Dense>
#include "System.h"

// System includes
#include <cstring>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <errno.h>
#include <ctime>
#include <random>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <memory>
#include <set>

// MAVLink headers
#include <common/mavlink.h>

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

// Interpolate IMU measurements to match gyro timestamps
rs2_vector interpolateMeasure(const double target_time,
                              const rs2_vector current_data, const double current_time,
                              const rs2_vector prev_data, const double prev_time)
{
    if(prev_time == 0) {
        return current_data;
    }

    rs2_vector value_interp;

    if(target_time > current_time) {
        value_interp = current_data;
    }
    else if(target_time > prev_time) {
        // Zero-order hold (use current data)
        value_interp = current_data;
    }
    else {
        value_interp = prev_data;
    }

    return value_interp;
}

//=============================================================================
// MAVLink Mode Enumeration
//=============================================================================

enum class MAVLinkMode {
    ODOMETRY,                    // Send ODOMETRY messages (default)
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

            // EKF Position Feedback - This is the key addition for coordinate bridging
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
    VIOBridge(MAVLinkMode mode = MAVLinkMode::VISION_POSITION_ESTIMATE) {
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

        // Initialize MAVLink
        const char* serial_port = "/dev/ttyTHS1";
        int baud_rate = 1500000;

        mavlink = std::make_shared<MAVLinkInterface>(serial_port, baud_rate, mode);
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

public:
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
};

//=============================================================================
// Main Function
//=============================================================================

int main(int argc, char **argv) {
    std::cout << "==================================================" << std::endl;
    std::cout << "ORB-SLAM3 Stereo-Inertial VIO v2 (ArduPilot Bridge)" << std::endl;
    std::cout << "==================================================" << std::endl;

    // Check for --test flag anywhere in arguments
    bool test_mode = false;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--test") {
            test_mode = true;
        }
    }

    if(argc < 3 || (argc > 5 && !test_mode) || (argc > 4 && test_mode)) {
        std::cerr << "Usage:" << std::endl;
        std::cerr << "  Normal:  ./stereo_inertial_realsense_D455_VIO_v2 path_to_vocabulary path_to_settings [mode] [visualization]" << std::endl;
        std::cerr << "  Test:    ./stereo_inertial_realsense_D455_VIO_v2 path_to_vocabulary path_to_settings --test" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  mode: 0 = ODOMETRY" << std::endl;
        std::cerr << "        1 = VISION_POSITION_ESTIMATE (default)" << std::endl;
        std::cerr << "        2 = VISION_POSITION_ESTIMATE + VISION_SPEED_ESTIMATE" << std::endl;
        std::cerr << "  visualization: 0 = OFF (default)" << std::endl;
        std::cerr << "                 1 = ON (Pangolin viewer)" << std::endl;
        std::cerr << std::endl;
        std::cerr << "  --test: Run without MAVLink/flight controller, visualization ON" << std::endl;
        return 1;
    }

    if (test_mode) {
        std::cout << "==================================================" << std::endl;
        std::cout << "  TEST MODE - No MAVLink, Visualization ON" << std::endl;
        std::cout << "==================================================" << std::endl;
    }

    // Parse MAVLink mode (only relevant in normal mode)
    MAVLinkMode mavlink_mode = MAVLinkMode::VISION_POSITION_ESTIMATE;  // v2 defaults to VISION_POSITION_ESTIMATE
    if (!test_mode && argc >= 4) {
        int mode_val = std::atoi(argv[3]);
        if (mode_val == 0) {
            mavlink_mode = MAVLinkMode::ODOMETRY;
        } else if (mode_val == 1) {
            mavlink_mode = MAVLinkMode::VISION_POSITION_ESTIMATE;
        } else if (mode_val == 2) {
            mavlink_mode = MAVLinkMode::VISION_POSITION_AND_SPEED;
        } else {
            std::cerr << "Invalid mode: " << mode_val << ". Using VISION_POSITION_ESTIMATE." << std::endl;
        }
    }
    if (!test_mode) {
        std::cout << "[Main] MAVLink mode = " << static_cast<int>(mavlink_mode) << std::endl;
    }

    // Parse visualization option (test mode forces ON)
    bool enable_visualization = test_mode;  // Default: OFF for headless, ON for test
    if (!test_mode && argc >= 5) {
        int vis_val = std::atoi(argv[4]);
        enable_visualization = (vis_val != 0);
    }
    std::cout << "[Main] Visualization = " << (enable_visualization ? "ON" : "OFF") << std::endl;

    // Create SLAM system
    std::cout << "[Main] Loading ORB-SLAM3..." << std::endl;
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::IMU_STEREO, enable_visualization);

    // Create VIO Bridge only in normal mode
    std::unique_ptr<VIOBridge> vio_bridge;
    if (!test_mode) {
        vio_bridge = std::make_unique<VIOBridge>(mavlink_mode);
    }

    // Configure RealSense
    std::cout << "[Main] Configuring RealSense D455..." << std::endl;
    rs2::pipeline pipe;
    rs2::config cfg;

    cfg.enable_stream(RS2_STREAM_INFRARED, 1, 640, 480, RS2_FORMAT_Y8, 30);
    cfg.enable_stream(RS2_STREAM_INFRARED, 2, 640, 480, RS2_FORMAT_Y8, 30);
    cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
    cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F);

    // IMU callback variables
    std::mutex imu_mutex;
    std::condition_variable cond_image_rec;

    std::vector<double> v_accel_timestamp;
    std::vector<rs2_vector> v_accel_data;
    std::vector<double> v_gyro_timestamp;
    std::vector<rs2_vector> v_gyro_data;

    double prev_accel_timestamp = 0;
    rs2_vector prev_accel_data;
    double current_accel_timestamp = 0;
    rs2_vector current_accel_data;
    std::vector<double> v_accel_timestamp_sync;
    std::vector<rs2_vector> v_accel_data_sync;

    cv::Mat imCV, imRightCV;
    int width_img = 640, height_img = 480;
    double timestamp_image = -1.0;
    bool image_ready = false;
    int count_im_buffer = 0;

    double offset = 0;

    // Variables to store latest IMU data for logging
    Eigen::Vector3f latest_accel(0, 0, 0);
    Eigen::Vector3f latest_gyro(0, 0, 0);
    double latest_imu_timestamp = 0;

    // IMU callback
    auto imu_callback = [&](const rs2::frame& frame)
    {
        std::unique_lock<std::mutex> lock(imu_mutex);

        if(rs2::frameset fs = frame.as<rs2::frameset>())
        {
            count_im_buffer++;

            double new_timestamp_image = fs.get_timestamp()*1e-3;
            if(std::abs(timestamp_image-new_timestamp_image)<0.001){
                count_im_buffer--;
                return;
            }

            rs2::video_frame ir_frameL = fs.get_infrared_frame(1);
            rs2::video_frame ir_frameR = fs.get_infrared_frame(2);

            imCV = cv::Mat(cv::Size(width_img, height_img), CV_8U, (void*)(ir_frameL.get_data()), cv::Mat::AUTO_STEP);
            imRightCV = cv::Mat(cv::Size(width_img, height_img), CV_8U, (void*)(ir_frameR.get_data()), cv::Mat::AUTO_STEP);

            timestamp_image = fs.get_timestamp()*1e-3;
            image_ready = true;

            while(v_gyro_timestamp.size() > v_accel_timestamp_sync.size())
            {
                int index = v_accel_timestamp_sync.size();
                double target_time = v_gyro_timestamp[index];

                v_accel_data_sync.push_back(current_accel_data);
                v_accel_timestamp_sync.push_back(target_time);
            }

            lock.unlock();
            cond_image_rec.notify_all();
        }
        else if (rs2::motion_frame m_frame = frame.as<rs2::motion_frame>())
        {
            if (m_frame.get_profile().stream_name() == "Gyro")
            {
                v_gyro_data.push_back(m_frame.get_motion_data());
                v_gyro_timestamp.push_back((m_frame.get_timestamp()+offset)*1e-3);
            }
            else if (m_frame.get_profile().stream_name() == "Accel")
            {
                prev_accel_timestamp = current_accel_timestamp;
                prev_accel_data = current_accel_data;

                current_accel_data = m_frame.get_motion_data();
                current_accel_timestamp = (m_frame.get_timestamp()+offset)*1e-3;

                while(v_gyro_timestamp.size() > v_accel_timestamp_sync.size())
                {
                    int index = v_accel_timestamp_sync.size();
                    double target_time = v_gyro_timestamp[index];

                    rs2_vector interp_data = interpolateMeasure(target_time, current_accel_data, current_accel_timestamp,
                                                                prev_accel_data, prev_accel_timestamp);

                    v_accel_data_sync.push_back(interp_data);
                    v_accel_timestamp_sync.push_back(target_time);
                }
            }
        }
    };

    std::cout << "[Main] Starting RealSense pipeline..." << std::endl;
    rs2::pipeline_profile profile = pipe.start(cfg, imu_callback);

    std::vector<ORB_SLAM3::IMU::Point> vImuMeas;
    int frame_count = 0;

    std::cout << "[Main] Starting VIO tracking with coordinate bridging..." << std::endl;
    std::cout << "[Main] Move the camera to initialize the system." << std::endl;

    // Wait for MAVLink connection (skip in test mode)
    if (!test_mode) {
        if(!vio_bridge->waitForConnection(30)) {
            std::cout << "[Main] MAVLink connection failed." << std::endl;
            return 1;
        }
        std::cout << "[Main] MAVLink connected!" << std::endl;
    }

    // Clear IMU vectors
    v_gyro_data.clear();
    v_gyro_timestamp.clear();
    v_accel_data_sync.clear();
    v_accel_timestamp_sync.clear();

    // Health check timer
    uint32_t last_health_check = getCurrentTimeMs();
    const uint32_t HEALTH_CHECK_INTERVAL_MS = 5000;

    std::cout << "[Main] Entering main loop..." << std::endl;

    while(true) {
        std::vector<rs2_vector> vGyro;
        std::vector<double> vGyro_times;
        std::vector<rs2_vector> vAccel;
        cv::Mat left, right;
        double timestamp;

        {
            std::unique_lock<std::mutex> lk(imu_mutex);
            if(!image_ready)
                cond_image_rec.wait(lk);

            if(count_im_buffer > 1)
                std::cout << "[Main] " << count_im_buffer - 1 << " dropped frames" << std::endl;
            count_im_buffer = 0;

            while(v_gyro_timestamp.size() > v_accel_timestamp_sync.size())
            {
                int index = v_accel_timestamp_sync.size();
                double target_time = v_gyro_timestamp[index];

                rs2_vector interp_data = interpolateMeasure(target_time, current_accel_data, current_accel_timestamp,
                                                            prev_accel_data, prev_accel_timestamp);

                v_accel_data_sync.push_back(interp_data);
                v_accel_timestamp_sync.push_back(target_time);
            }

            vGyro = v_gyro_data;
            vGyro_times = v_gyro_timestamp;
            vAccel = v_accel_data_sync;
            timestamp = timestamp_image;
            left = imCV.clone();
            right = imRightCV.clone();

            v_gyro_data.clear();
            v_gyro_timestamp.clear();
            v_accel_data_sync.clear();
            v_accel_timestamp_sync.clear();

            image_ready = false;
        }

        frame_count++;

        // Build IMU measurements
        for(size_t i = 0; i < vGyro.size(); ++i)
        {
            ORB_SLAM3::IMU::Point imu_point(vAccel[i].x, vAccel[i].y, vAccel[i].z,
                                          vGyro[i].x, vGyro[i].y, vGyro[i].z,
                                          vGyro_times[i]);
            vImuMeas.push_back(imu_point);
        }

        // Store latest IMU data
        if(!vGyro.empty()) {
            latest_gyro = Eigen::Vector3f(vGyro.back().x, vGyro.back().y, vGyro.back().z);
            latest_accel = Eigen::Vector3f(vAccel.back().x, vAccel.back().y, vAccel.back().z);
            latest_imu_timestamp = vGyro_times.back();
            if (vio_bridge) {
                vio_bridge->updateIMUData(latest_imu_timestamp, latest_accel, latest_gyro);
            }
        }

        // Track with stereo-inertial
        Sophus::SE3f Tcw = SLAM.TrackStereo(left, right, timestamp, vImuMeas);

        // Get velocity from ORB-SLAM3
        Eigen::Vector3f velocity = SLAM.GetVelocity();

        // Get tracking state
        auto tracking_state = SLAM.GetTrackingState();

        // Display tracking status (reduced frequency)
        static int status_counter = 0;
        if (++status_counter % 30 == 0) {
            switch(tracking_state) {
                case ORB_SLAM3::Tracking::SYSTEM_NOT_READY:
                    std::cout << "[Tracking] System not ready" << std::endl;
                    break;
                case ORB_SLAM3::Tracking::NO_IMAGES_YET:
                    std::cout << "[Tracking] No images yet" << std::endl;
                    break;
                case ORB_SLAM3::Tracking::NOT_INITIALIZED:
                    std::cout << "[Tracking] Not initialized - move camera with rotation!" << std::endl;
                    break;
                case ORB_SLAM3::Tracking::OK:
                    std::cout << "[Tracking] OK" << std::endl;
                    break;
                case ORB_SLAM3::Tracking::RECENTLY_LOST:
                    std::cout << "[Tracking] RECENTLY_LOST" << std::endl;
                    break;
                case ORB_SLAM3::Tracking::LOST:
                    std::cout << "[Tracking] LOST" << std::endl;
                    break;
                case ORB_SLAM3::Tracking::OK_KLT:
                    std::cout << "[Tracking] OK (KLT)" << std::endl;
                    break;
            }
        }

        if (test_mode) {
            // Test mode: print VIO data to console
            if (tracking_state == ORB_SLAM3::Tracking::OK ||
                tracking_state == ORB_SLAM3::Tracking::OK_KLT) {
                Sophus::SE3f Twc = Tcw.inverse();
                Eigen::Vector3f position = Twc.translation();
                Eigen::Quaternionf q = Twc.unit_quaternion();

                // Convert quaternion to Euler angles (roll, pitch, yaw) in degrees
                Eigen::Matrix3f R = q.toRotationMatrix();
                float roll  = atan2(R(2,1), R(2,2)) * 180.0f / M_PI;
                float pitch = atan2(-R(2,0), sqrt(R(2,1)*R(2,1) + R(2,2)*R(2,2))) * 180.0f / M_PI;
                float yaw   = atan2(R(1,0), R(0,0)) * 180.0f / M_PI;

                static int print_counter = 0;
                if (++print_counter % 10 == 0) {
                    std::cout << "[TEST] pos=(" << std::fixed << std::setprecision(3)
                              << position.x() << ", " << position.y() << ", " << position.z()
                              << ") rpy=(" << std::setprecision(1) << roll << ", " << pitch << ", " << yaw
                              << ") vel=(" << std::setprecision(2) << velocity.x() << ", " << velocity.y() << ", " << velocity.z()
                              << ") frame=" << frame_count << std::endl;
                }
            }
        } else {
            // Normal mode: process pose through VIO Bridge (coordinate bridging happens here)
            vio_bridge->processORBSLAMPose(Tcw, tracking_state, velocity);

            // Periodic health check
            uint32_t current_time = getCurrentTimeMs();
            if (current_time - last_health_check > HEALTH_CHECK_INTERVAL_MS) {
                vio_bridge->performHealthCheck();
                last_health_check = current_time;
            }
        }

        // Clear IMU measurements for next frame
        vImuMeas.clear();
    }

    std::cout << "[Main] Shutting down..." << std::endl;
    SLAM.Shutdown();
    return 0;
}
