/**
 * @file stereo_inertial_realsense_D455_VIO_GPS_v4.cc
 * @brief VIO to GPS output with coordinate frame bridging for ArduPilot integration
 *
 * v4 Improvements over v2:
 * - VIOBridgeState structure for centralized state management
 * - Coordinate transform layer for map transition handling (position continuity)
 * - EKF feedback loop via LOCAL_POSITION_NED for offset calculation
 * - Heuristic-based map change detection (position jump > 2m after recovery)
 * - Proper tracking state handling:
 *   - OK/OK_KLT: Normal operation, send GPS with fix_type=3
 *   - RECENTLY_LOST: Send high-uncertainty estimate (fix_type=2, EPH/EPV=500)
 *   - LOST: Send very degraded estimate (fix_type=1, EPH/EPV=2000)
 *   - NOT_INITIALIZED/NO_IMAGES: No GPS sent (no valid position available)
 * - Never increment reset_counter for new map creation (conservative approach)
 * - stderr for unbuffered real-time output
 * - --vis/--novis command line flag for visualization control
 *
 * Key architectural change: Instead of passing coordinate frame discontinuities
 * to ArduPilot via reset_counter, we maintain a persistent offset between VIO
 * coordinates and NED coordinates.
 *
 * NED_position = transformVIOtoNED(VIO_position) + vio_to_ned_offset
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

// GPS quality values for normal tracking
constexpr float NORMAL_EPH_EPV = 50;              // 0.5m accuracy (cm*100)
constexpr uint8_t NORMAL_SATELLITES = 12;
constexpr uint8_t NORMAL_FIX_TYPE = 3;            // 3D fix

// GPS quality values for RECENTLY_LOST (high uncertainty)
constexpr float RECENTLY_LOST_EPH_EPV = 500;      // 5m accuracy (cm*100)
constexpr uint8_t RECENTLY_LOST_SATELLITES = 6;
constexpr uint8_t RECENTLY_LOST_FIX_TYPE = 2;     // 2D fix (degraded)

// GPS quality values for LOST (very high uncertainty)
constexpr float LOST_EPH_EPV = 2000;              // 20m accuracy (cm*100)
constexpr uint8_t LOST_SATELLITES = 4;
constexpr uint8_t LOST_FIX_TYPE = 1;              // No fix (very degraded)

// Earth constants
constexpr double EARTH_RADIUS = 6378137.0;

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
// GPS Coordinate Structures and Functions
//=============================================================================

struct GPSCoord {
    double lat;
    double lon;
    double alt;
};

GPSCoord ned_to_gps(double north, double east, double down, const GPSCoord& origin) {
    GPSCoord result;

    double lat_rad = origin.lat * M_PI / 180.0;

    double dLat = north / EARTH_RADIUS;
    double dLon = east / (EARTH_RADIUS * cos(lat_rad));

    result.lat = origin.lat + (dLat * 180.0 / M_PI);
    result.lon = origin.lon + (dLon * 180.0 / M_PI);
    result.alt = origin.alt - down;

    return result;
}

//=============================================================================
// GPS Data Structure
//=============================================================================

struct GPSData {
    uint64_t time_usec;
    int32_t lat;
    int32_t lon;
    int32_t alt;
    float vn, ve, vd;
    uint16_t cog;
    uint8_t fix_type;
    uint8_t satellites_visible;
    uint16_t eph;
    uint16_t epv;
    uint16_t vel;
    int16_t vz;
    uint8_t reset_counter;
};

//=============================================================================
// Thread-Safe GPS Queue
//=============================================================================

class GPSQueue {
private:
    std::queue<GPSData> queue;
    std::mutex mutex;
    std::condition_variable cv;
    const size_t max_size = 100;

public:
    void push(const GPSData& data) {
        std::unique_lock<std::mutex> lock(mutex);
        if (queue.size() >= max_size) {
            queue.pop();
        }
        queue.push(data);
        cv.notify_one();
    }

    bool pop(GPSData& data, int timeout_ms = 100) {
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
// VIO Bridge State Structure
//=============================================================================

struct VIOBridgeState {
    // Coordinate transformation
    Eigen::Vector3d vio_to_ned_offset;
    Eigen::Quaterniond vio_to_ned_rotation;

    // Tracking state tracking
    int tracking_state_current;
    int tracking_state_previous;
    std::set<unsigned long> map_history;
    unsigned long current_map_id;

    // EKF feedback
    Eigen::Vector3d last_ekf_position;
    uint32_t last_ekf_update_ms;
    bool ekf_position_valid;

    // Last valid poses
    Eigen::Vector3d last_valid_ned_position;
    Eigen::Vector3d last_valid_vio_position;
    Eigen::Vector3d last_valid_ned_velocity;

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
        , current_map_id(0)
        , last_ekf_position(Eigen::Vector3d::Zero())
        , last_ekf_update_ms(0)
        , ekf_position_valid(false)
        , last_valid_ned_position(Eigen::Vector3d::Zero())
        , last_valid_vio_position(Eigen::Vector3d::Zero())
        , last_valid_ned_velocity(Eigen::Vector3d::Zero())
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
// MAVLink GPS Interface (Extended with EKF Feedback)
//=============================================================================

class MAVLinkGPSInterface {
private:
    int serial_fd;
    uint8_t system_id;
    uint8_t component_id;

    std::atomic<bool> running;
    std::thread mavlink_thread;
    GPSQueue gps_queue;
    std::atomic<bool> connected;
    std::mutex connected_mutex;
    std::condition_variable connected_cv;

    // GPS origin tracking
    std::atomic<bool> gps_origin_received;
    GPSCoord gps_origin;
    std::mutex gps_origin_mutex;
    std::condition_variable gps_origin_cv;

    // EKF position feedback (thread-safe)
    mutable std::mutex ekf_mutex;
    EKFPositionFeedback ekf_feedback;

    std::chrono::steady_clock::time_point last_waiting_msg_time;

public:
    MAVLinkGPSInterface(const char* serial_port, int baud_rate = 57600)
        : system_id(255), component_id(197), running(false), connected(false), gps_origin_received(false) {

        std::cerr << "[MAVLinkGPSInterface] Initializing GPS interface" << std::endl;

        serial_fd = open(serial_port, O_RDWR | O_NOCTTY);
        if (serial_fd < 0) {
            std::cerr << "Failed to open serial port " << serial_port
                     << ": " << strerror(errno) << std::endl;
            exit(1);
        }

        struct termios tty;
        memset(&tty, 0, sizeof(tty));

        if (tcgetattr(serial_fd, &tty) != 0) {
            std::cerr << "Error getting serial port attributes: "
                     << strerror(errno) << std::endl;
            close(serial_fd);
            exit(1);
        }

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
                std::cerr << "Unsupported baud rate: " << baud_rate
                         << ", using 57600" << std::endl;
                speed = B57600;
        }

        cfsetospeed(&tty, speed);
        cfsetispeed(&tty, speed);

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
            std::cerr << "Error setting serial port attributes: "
                     << strerror(errno) << std::endl;
            close(serial_fd);
            exit(1);
        }

        tcflush(serial_fd, TCIOFLUSH);

        last_waiting_msg_time = std::chrono::steady_clock::now();

        std::cerr << "[MAVLink] Interface initialized on " << serial_port
                  << " @ " << baud_rate << " baud" << std::endl;
    }

    ~MAVLinkGPSInterface() {
        stop();
        close(serial_fd);
    }

    void start() {
        if (!running) {
            running = true;
            mavlink_thread = std::thread(&MAVLinkGPSInterface::run, this);
            std::cerr << "[MAVLink] Thread started" << std::endl;
        }
    }

    void stop() {
        if (running) {
            running = false;
            if (mavlink_thread.joinable()) {
                mavlink_thread.join();
            }
            std::cerr << "[MAVLink] Thread stopped" << std::endl;
        }
    }

    void setGPSOrigin(double lat, double lon, double alt) {
        std::lock_guard<std::mutex> lock(gps_origin_mutex);
        gps_origin.lat = lat;
        gps_origin.lon = lon;
        gps_origin.alt = alt;
        gps_origin_received = true;
        gps_origin_cv.notify_all();
        std::cerr << "[GPS Origin] Set from command line: Lat=" << std::fixed << std::setprecision(7)
                  << lat << " Lon=" << lon << " Alt=" << std::setprecision(2) << alt << "m" << std::endl;
    }

    bool waitForGPSOrigin(int timeout_seconds = 0) {
        std::unique_lock<std::mutex> lock(gps_origin_mutex);

        if (gps_origin_received) {
            return true;
        }

        std::cerr << "\n========================================" << std::endl;
        std::cerr << "Waiting for GPS origin from ArduPilot..." << std::endl;
        std::cerr << "Listening for HOME_POSITION or GPS_GLOBAL_ORIGIN messages" << std::endl;
        std::cerr << "========================================\n" << std::endl;

        if (timeout_seconds > 0) {
            return gps_origin_cv.wait_for(lock, std::chrono::seconds(timeout_seconds),
                                          [this] { return gps_origin_received.load(); });
        } else {
            // Wait indefinitely with periodic status messages
            while (!gps_origin_received) {
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_waiting_msg_time).count();

                if (elapsed >= 30) {
                    std::cerr << "[Status] Still waiting for GPS origin from ArduPilot..." << std::endl;
                    last_waiting_msg_time = now;
                }

                gps_origin_cv.wait_for(lock, std::chrono::seconds(1));
            }
            return true;
        }
    }

    bool hasGPSOrigin() const {
        return gps_origin_received;
    }

    GPSCoord getGPSOrigin() {
        std::lock_guard<std::mutex> lock(gps_origin_mutex);
        return gps_origin;
    }

    void queueGPS(const GPSData& data) {
        gps_queue.push(data);
    }

    size_t getQueueSize() const {
        return const_cast<GPSQueue&>(gps_queue).size();
    }

    bool isConnected() const {
        return connected;
    }

    bool waitForConnection(int timeout_seconds = 10) {
        std::cerr << "[MAVLink] Waiting for connection";

        std::unique_lock<std::mutex> lock(connected_mutex);

        for (int i = 0; i < timeout_seconds; i++) {
            if (connected_cv.wait_for(lock, std::chrono::seconds(1),
                                     [this] { return connected.load(); })) {
                std::cerr << " Connected!" << std::endl;
                return true;
            }
            std::cerr << ".";
        }

        std::cerr << " Timeout!" << std::endl;
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
        requestHomePosition();
        requestLocalPositionNED();

        int heartbeat_counter = 0;

        while (running) {
            if (heartbeat_counter % 100 == 0) {
                sendHeartbeat();
            }

            GPSData gps_data;
            if (gps_queue.pop(gps_data, 10)) {
                sendGPSInput(gps_data);
            }

            receiveMessages();

            usleep(10000);
            heartbeat_counter++;
        }
    }

    void sendHeartbeat() {
        mavlink_message_t msg;
        uint8_t buf[MAVLINK_MAX_PACKET_LEN];

        mavlink_msg_heartbeat_pack(system_id, component_id, &msg,
                                   MAV_TYPE_ONBOARD_CONTROLLER,
                                   MAV_AUTOPILOT_INVALID,
                                   0,
                                   0,
                                   MAV_STATE_ACTIVE);

        uint16_t len = mavlink_msg_to_send_buffer(buf, &msg);
        write(serial_fd, buf, len);
    }

    void requestDataStream() {
        mavlink_message_t msg;
        uint8_t buf[MAVLINK_MAX_PACKET_LEN];

        mavlink_msg_request_data_stream_pack(system_id, component_id, &msg,
                                            1, 1,
                                            MAV_DATA_STREAM_ALL,
                                            1,
                                            1);

        uint16_t len = mavlink_msg_to_send_buffer(buf, &msg);
        write(serial_fd, buf, len);

        std::cerr << "[MAVLink] Requested data stream from ArduPilot" << std::endl;
    }

    void requestHomePosition() {
        mavlink_message_t msg;
        uint8_t buf[MAVLINK_MAX_PACKET_LEN];

        // Request HOME_POSITION message
        mavlink_msg_command_long_pack(
            system_id,
            component_id,
            &msg,
            1, 1,  // target system, target component
            MAV_CMD_GET_HOME_POSITION,
            0,     // confirmation
            0, 0, 0, 0, 0, 0, 0  // params (unused)
        );

        uint16_t len = mavlink_msg_to_send_buffer(buf, &msg);
        write(serial_fd, buf, len);

        std::cerr << "[MAVLink] Requested HOME_POSITION from ArduPilot" << std::endl;
    }

    void requestLocalPositionNED() {
        mavlink_message_t msg;
        uint8_t buf[MAVLINK_MAX_PACKET_LEN];

        // Request LOCAL_POSITION_NED stream at 20Hz for EKF feedback
        mavlink_msg_request_data_stream_pack(system_id, component_id, &msg,
                                            1, 1,
                                            MAV_DATA_STREAM_POSITION,
                                            20,  // 20 Hz
                                            1);

        uint16_t len = mavlink_msg_to_send_buffer(buf, &msg);
        write(serial_fd, buf, len);

        std::cerr << "[MAVLink] Requested LOCAL_POSITION_NED stream at 20Hz" << std::endl;
    }

    void sendGPSInput(const GPSData& data) {
        mavlink_message_t msg;
        uint8_t buf[MAVLINK_MAX_PACKET_LEN];

        mavlink_msg_gps_input_pack(
            system_id,
            component_id,
            &msg,
            data.time_usec,
            0,
            0,
            data.time_usec / 1000,
            0,
            data.fix_type,
            data.lat,
            data.lon,
            data.alt / 1000.0f,
            data.eph / 100.0f,
            data.epv / 100.0f,
            data.vel / 100.0f,
            data.vn,
            data.ve,
            data.vd,
            0.5f,
            0.5f,
            0.5f,
            data.satellites_visible
        );

        uint16_t len = mavlink_msg_to_send_buffer(buf, &msg);
        ssize_t sent = write(serial_fd, buf, len);

        if (sent < 0) {
            std::cerr << "[MAVLink] Failed to send GPS input: " << strerror(errno) << std::endl;
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

                        std::cerr << "[MAVLink] Connected to system " << (int)msg.sysid << std::endl;
                    }
                }
                break;
            }

            // Listen for HOME_POSITION message (ID 242)
            case MAVLINK_MSG_ID_HOME_POSITION: {
                if (!gps_origin_received) {
                    mavlink_home_position_t home_pos;
                    mavlink_msg_home_position_decode(&msg, &home_pos);

                    std::lock_guard<std::mutex> lock(gps_origin_mutex);
                    gps_origin.lat = home_pos.latitude / 1e7;
                    gps_origin.lon = home_pos.longitude / 1e7;
                    gps_origin.alt = home_pos.altitude / 1000.0;  // mm to meters
                    gps_origin_received = true;
                    gps_origin_cv.notify_all();

                    std::cerr << "\n========================================" << std::endl;
                    std::cerr << "[GPS Origin] Received HOME_POSITION from ArduPilot" << std::endl;
                    std::cerr << "Lat: " << std::fixed << std::setprecision(7) << gps_origin.lat
                              << " Lon: " << gps_origin.lon
                              << " Alt: " << std::setprecision(2) << gps_origin.alt << "m" << std::endl;
                    std::cerr << "========================================\n" << std::endl;
                }
                break;
            }

            // Listen for GPS_GLOBAL_ORIGIN message (ID 49)
            case MAVLINK_MSG_ID_GPS_GLOBAL_ORIGIN: {
                if (!gps_origin_received) {
                    mavlink_gps_global_origin_t global_origin;
                    mavlink_msg_gps_global_origin_decode(&msg, &global_origin);

                    std::lock_guard<std::mutex> lock(gps_origin_mutex);
                    gps_origin.lat = global_origin.latitude / 1e7;
                    gps_origin.lon = global_origin.longitude / 1e7;
                    gps_origin.alt = global_origin.altitude / 1000.0;  // mm to meters
                    gps_origin_received = true;
                    gps_origin_cv.notify_all();

                    std::cerr << "\n========================================" << std::endl;
                    std::cerr << "[GPS Origin] Received GPS_GLOBAL_ORIGIN from ArduPilot" << std::endl;
                    std::cerr << "Lat: " << std::fixed << std::setprecision(7) << gps_origin.lat
                              << " Lon: " << gps_origin.lon
                              << " Alt: " << std::setprecision(2) << gps_origin.alt << "m" << std::endl;
                    std::cerr << "========================================\n" << std::endl;
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
                if (++log_counter % 200 == 0) {
                    std::cerr << "[EKF Feedback] Position: (" << std::fixed << std::setprecision(2)
                              << pos.x << ", " << pos.y << ", " << pos.z << ")" << std::endl;
                }
                break;
            }
        }
    }
};

//=============================================================================
// VIO GPS Bridge Class - Coordinate Transform Layer with GPS Output
//=============================================================================

class VIOGPSBridge {
private:
    VIOBridgeState state;
    std::shared_ptr<MAVLinkGPSInterface> mavlink;
    uint64_t start_time_us;

    // IMU data for velocity estimation
    struct IMUData {
        double timestamp;
        Eigen::Vector3f accel;
        Eigen::Vector3f gyro;
        bool valid;

        IMUData() : timestamp(0), accel(0,0,0), gyro(0,0,0), valid(false) {}
    } latest_imu;

public:
    VIOGPSBridge() {
        // Set unbuffered output for real-time logging
        setvbuf(stderr, NULL, _IONBF, 0);

        // Get system start time
        auto now = std::chrono::high_resolution_clock::now();
        start_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
            now.time_since_epoch()).count();

        std::cerr << "==================================================" << std::endl;
        std::cerr << "[VIOGPSBridge v4] Initializing with coordinate bridging" << std::endl;
        std::cerr << "==================================================" << std::endl;

        // Initialize MAVLink GPS interface
        const char* serial_port = "/dev/ttyTHS1";
        int baud_rate = 1500000;

        mavlink = std::make_shared<MAVLinkGPSInterface>(serial_port, baud_rate);
        mavlink->start();
    }

    ~VIOGPSBridge() {
        if (mavlink) {
            mavlink->stop();
        }

        std::cerr << "==================================================" << std::endl;
        std::cerr << "[VIOGPSBridge] Shutdown statistics:" << std::endl;
        std::cerr << "  Messages sent: " << state.messages_sent << std::endl;
        std::cerr << "  Map transitions: " << state.map_transitions << std::endl;
        std::cerr << "  Tracking loss events: " << state.tracking_loss_events << std::endl;
        std::cerr << "==================================================" << std::endl;
    }

    void setGPSOrigin(double lat, double lon, double alt) {
        mavlink->setGPSOrigin(lat, lon, alt);
    }

    bool waitForGPSOrigin() {
        return mavlink->waitForGPSOrigin(0);  // Wait indefinitely
    }

    bool hasGPSOrigin() const {
        return mavlink->hasGPSOrigin();
    }

    bool waitForConnection(int timeout_seconds = 30) {
        if (!mavlink->waitForConnection(timeout_seconds)) {
            std::cerr << "[VIOGPSBridge] Failed to connect to ArduPilot!" << std::endl;
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
    void processORBSLAMPose(double timestamp,
                           const Sophus::SE3f& Tcw,
                           int tracking_state,
                           const Eigen::Vector3f& velocity) {

        if (!hasGPSOrigin()) {
            // Don't process if we don't have GPS origin yet
            return;
        }

        uint32_t current_time_ms = getCurrentTimeMs();

        // Update tracking state history
        state.tracking_state_previous = state.tracking_state_current;
        state.tracking_state_current = tracking_state;

        // Extract VIO pose (inverse of Tcw gives camera position in world frame)
        Sophus::SE3f Twc = Tcw.inverse();
        Eigen::Vector3f position_f = Twc.translation();

        // Convert to double precision for transformations
        Eigen::Vector3d vio_position = position_f.cast<double>();

        // Convert velocity to double
        Eigen::Vector3d vio_velocity = velocity.cast<double>();

        // Handle based on tracking state
        switch (tracking_state) {
            case ORB_SLAM3::Tracking::OK:
            case ORB_SLAM3::Tracking::OK_KLT:
                handleTrackingOK(timestamp, vio_position, vio_velocity, current_time_ms);
                break;

            case ORB_SLAM3::Tracking::RECENTLY_LOST:
                handleRecentlyLost(timestamp, current_time_ms);
                break;

            case ORB_SLAM3::Tracking::LOST:
                // LOST: Send very degraded GPS estimate if we have a previous position
                handleTrackingLostDegraded(timestamp, current_time_ms);
                break;

            case ORB_SLAM3::Tracking::NOT_INITIALIZED:
            case ORB_SLAM3::Tracking::SYSTEM_NOT_READY:
            case ORB_SLAM3::Tracking::NO_IMAGES_YET:
                // No valid position ever received, don't send GPS
                handleTrackingNotInitialized();
                return;
        }
    }

private:
    /**
     * Handle normal tracking (OK or OK_KLT)
     */
    void handleTrackingOK(double timestamp,
                         const Eigen::Vector3d& vio_position,
                         const Eigen::Vector3d& vio_velocity,
                         uint32_t current_time_ms) {

        // Reset tracking loss flag
        static bool was_tracking = true;
        was_tracking = true;

        // Check for map change using heuristics
        bool map_changed = detectMapChange(vio_position);

        if (map_changed) {
            handleMapTransition(vio_position);
        }

        // Apply coordinate transformation
        Eigen::Vector3d ned_position = applyCoordinateTransform(vio_position);
        Eigen::Vector3d ned_velocity = applyVelocityTransform(vio_velocity);

        // Save as last valid pose
        state.last_valid_ned_position = ned_position;
        state.last_valid_vio_position = vio_position;
        state.last_valid_ned_velocity = ned_velocity;

        // Update EKF feedback position
        updateEKFPosition();

        // Send to ArduPilot as GPS with normal quality
        sendGPSData(timestamp, ned_position, ned_velocity, current_time_ms,
                   NORMAL_FIX_TYPE, NORMAL_SATELLITES, NORMAL_EPH_EPV, NORMAL_EPH_EPV);

        state.last_pose_time_ms = current_time_ms;
        state.messages_sent++;
    }

    /**
     * Handle recently lost tracking
     * Continue sending last valid position with high uncertainty (degraded quality)
     */
    void handleRecentlyLost(double timestamp, uint32_t current_time_ms) {
        // Log tracking loss event (once per loss sequence)
        static bool was_tracking = true;
        if (was_tracking) {
            std::cerr << "[VIOGPSBridge] RECENTLY_LOST - sending high-uncertainty GPS estimate" << std::endl;
            state.tracking_loss_events++;
            was_tracking = false;
        }

        // Send last valid position with degraded quality
        sendGPSData(timestamp, state.last_valid_ned_position, Eigen::Vector3d::Zero(),
                   current_time_ms, RECENTLY_LOST_FIX_TYPE, RECENTLY_LOST_SATELLITES,
                   RECENTLY_LOST_EPH_EPV, RECENTLY_LOST_EPH_EPV);

        state.messages_sent++;
    }

    /**
     * Handle complete tracking loss (LOST state)
     * Continue sending GPS with very degraded quality to keep ArduPilot informed
     * Uses last valid position with very high uncertainty
     */
    void handleTrackingLostDegraded(double timestamp, uint32_t current_time_ms) {
        // Log tracking loss event (once per loss sequence)
        static bool was_tracking = true;
        if (was_tracking) {
            std::cerr << "[VIOGPSBridge] LOST - sending very degraded GPS estimate" << std::endl;
            state.tracking_loss_events++;
            was_tracking = false;
        }

        // Only send if we have a previous valid position
        if (state.first_pose_received) {
            // Send last valid position with very degraded quality
            sendGPSData(timestamp, state.last_valid_ned_position, Eigen::Vector3d::Zero(),
                       current_time_ms, LOST_FIX_TYPE, LOST_SATELLITES,
                       LOST_EPH_EPV, LOST_EPH_EPV);

            state.messages_sent++;
        } else {
            // No valid position ever received, can't send GPS
            static bool no_pos_logged = false;
            if (!no_pos_logged) {
                std::cerr << "[VIOGPSBridge] LOST but no valid position available yet" << std::endl;
                no_pos_logged = true;
            }
        }
    }

    /**
     * Handle states where tracking has never been initialized
     * Don't send any GPS messages as we have no valid position
     */
    void handleTrackingNotInitialized() {
        static bool logged = false;
        static int last_state = -1;

        if (!logged || last_state != state.tracking_state_current) {
            std::cerr << "[VIOGPSBridge] NOT INITIALIZED (state=" << state.tracking_state_current
                      << ") - no GPS available yet" << std::endl;
            logged = true;
            last_state = state.tracking_state_current;
        }

        // Don't send GPS messages until we have a valid initial position
    }

    /**
     * Detect map change using heuristic (position discontinuity after tracking recovery)
     */
    bool detectMapChange(const Eigen::Vector3d& new_vio_position) {
        // Case 1: First pose ever - not a map change
        if (!state.first_pose_received) {
            state.first_pose_received = true;
            state.vio_initialized = true;
            std::cerr << "[VIOGPSBridge] VIO initialized with first pose" << std::endl;
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
                std::cerr << "[VIOGPSBridge] Map change detected! Position jump: "
                          << std::fixed << std::setprecision(2) << pos_jump << "m" << std::endl;
                return true;
            } else {
                std::cerr << "[VIOGPSBridge] Tracking recovered in same map (jump: "
                          << std::fixed << std::setprecision(2) << pos_jump << "m)" << std::endl;
            }
        }

        return false;
    }

    /**
     * Handle map transition - calculate new offset to maintain position continuity
     */
    void handleMapTransition(const Eigen::Vector3d& new_vio_position) {

        state.map_transitions++;

        // Get reference position (where we should be)
        Eigen::Vector3d reference_position;

        // Prefer EKF feedback if available
        if (mavlink->hasRecentEKFFeedback()) {
            EKFPositionFeedback ekf = mavlink->getEKFPosition();
            reference_position = ekf.position;

            std::cerr << "[VIOGPSBridge] Using EKF feedback for offset: ("
                      << std::fixed << std::setprecision(2)
                      << reference_position.x() << ", "
                      << reference_position.y() << ", "
                      << reference_position.z() << ")" << std::endl;
        } else {
            // Fall back to last valid NED position
            reference_position = state.last_valid_ned_position;

            std::cerr << "[VIOGPSBridge] No recent EKF feedback, using last NED position: ("
                      << std::fixed << std::setprecision(2)
                      << reference_position.x() << ", "
                      << reference_position.y() << ", "
                      << reference_position.z() << ")" << std::endl;
        }

        // Calculate new offset: reference_position - transformed_vio_position
        // First transform VIO position to NED frame (without current offset)
        Eigen::Vector3d vio_in_ned = transformVIOtoNED(new_vio_position);

        // New offset ensures transformed VIO matches reference position
        state.vio_to_ned_offset = reference_position - vio_in_ned;

        std::cerr << "[VIOGPSBridge] *** MAP TRANSITION #" << state.map_transitions << " ***" << std::endl;
        std::cerr << "  New VIO position (raw): (" << std::fixed << std::setprecision(2)
                  << new_vio_position.x() << ", " << new_vio_position.y() << ", "
                  << new_vio_position.z() << ")" << std::endl;
        std::cerr << "  New offset: (" << state.vio_to_ned_offset.x() << ", "
                  << state.vio_to_ned_offset.y() << ", " << state.vio_to_ned_offset.z() << ")" << std::endl;

        // CRITICAL: Do NOT increment reset_counter for new map creation
        // Only increment for valid relocalization to a merged map
        // Since we're using heuristic detection, we don't know if maps are merged
        // So we never increment reset_counter (conservative approach)

        std::cerr << "  reset_counter unchanged at: " << (int)state.reset_counter << std::endl;
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
            state.ekf_position_valid = true;
        }
    }

    /**
     * Send GPS data via MAVLink GPS_INPUT
     */
    void sendGPSData(double timestamp,
                    const Eigen::Vector3d& ned_position,
                    const Eigen::Vector3d& ned_velocity,
                    uint32_t current_time_ms,
                    uint8_t fix_type,
                    uint8_t satellites,
                    uint16_t eph,
                    uint16_t epv) {

        GPSCoord gps_origin = mavlink->getGPSOrigin();

        // Convert timestamp to microseconds
        uint64_t timestamp_us = start_time_us + static_cast<uint64_t>(timestamp * 1e6);

        // Convert NED to GPS coordinates
        GPSCoord current_gps = ned_to_gps(ned_position.x(), ned_position.y(),
                                          ned_position.z(), gps_origin);

        // Build GPS data packet
        GPSData gps_data;
        gps_data.time_usec = timestamp_us;
        gps_data.lat = static_cast<int32_t>(current_gps.lat * 1e7);
        gps_data.lon = static_cast<int32_t>(current_gps.lon * 1e7);
        gps_data.alt = static_cast<int32_t>(current_gps.alt * 1000);

        gps_data.vn = static_cast<float>(ned_velocity.x());
        gps_data.ve = static_cast<float>(ned_velocity.y());
        gps_data.vd = static_cast<float>(ned_velocity.z());

        float ground_speed = std::sqrt(ned_velocity.x()*ned_velocity.x() +
                                       ned_velocity.y()*ned_velocity.y());
        gps_data.vel = static_cast<uint16_t>(ground_speed * 100);
        gps_data.vz = static_cast<int16_t>(ned_velocity.z() * 100);

        gps_data.cog = static_cast<uint16_t>(std::atan2(ned_velocity.y(), ned_velocity.x()) * 180.0 / M_PI * 100);

        gps_data.fix_type = fix_type;
        gps_data.satellites_visible = satellites;
        gps_data.eph = eph;
        gps_data.epv = epv;
        gps_data.reset_counter = state.reset_counter;

        // Queue GPS data for transmission
        mavlink->queueGPS(gps_data);

        // Log periodically
        static int log_counter = 0;
        if (++log_counter % 30 == 0) {
            std::cerr << "[GPS->MAV] Lat=" << std::fixed << std::setprecision(7)
                      << gps_data.lat/1e7 << " Lon=" << gps_data.lon/1e7
                      << " Alt=" << std::setprecision(2) << gps_data.alt/1000.0 << "m"
                      << " | fix=" << (int)fix_type << " sats=" << (int)satellites
                      << " eph=" << eph << " rst=" << (int)state.reset_counter << std::endl;
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
                std::cerr << "[VIOGPSBridge] WARNING: No pose data for " << pose_age << "ms" << std::endl;
                healthy = false;
            }
        }

        // Check 2: EKF feedback (warning only, not critical)
        if (!mavlink->hasRecentEKFFeedback()) {
            static int warning_count = 0;
            if (++warning_count % 100 == 0) {
                std::cerr << "[VIOGPSBridge] INFO: No recent EKF feedback" << std::endl;
            }
        }

        // Check 3: Position error (if we have EKF feedback)
        if (mavlink->hasRecentEKFFeedback() && state.first_pose_received) {
            EKFPositionFeedback ekf = mavlink->getEKFPosition();
            double position_error = (state.last_valid_ned_position - ekf.position).norm();
            if (position_error > 5.0) {
                std::cerr << "[VIOGPSBridge] WARNING: Large position error: "
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
    // Set unbuffered stderr for real-time output
    setvbuf(stderr, NULL, _IONBF, 0);

    std::cerr << "==================================================" << std::endl;
    std::cerr << "ORB-SLAM3 Stereo-Inertial VIO GPS v4" << std::endl;
    std::cerr << "(Coordinate Bridging for ArduPilot GPS_INPUT)" << std::endl;
    std::cerr << "==================================================" << std::endl;

    // Parse arguments
    // Basic: ./prog vocab settings
    // With GPS origin: ./prog vocab settings lat lon alt
    // With --vis/--novis anywhere in args

    bool enable_visualization = false;  // Default: OFF for headless operation
    std::vector<std::string> args;

    // Extract --vis/--novis flags and collect other arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--vis") {
            enable_visualization = true;
        } else if (arg == "--novis") {
            enable_visualization = false;
        } else {
            args.push_back(arg);
        }
    }

    // Validate remaining arguments
    if (args.size() != 2 && args.size() != 5) {
        std::cerr << "Usage: ./stereo_inertial_realsense_D455_VIO_GPS_v4 path_to_vocabulary path_to_settings [origin_lat origin_lon origin_alt] [--vis|--novis]" << std::endl;
        std::cerr << "\nWith GPS origin from command line:" << std::endl;
        std::cerr << "  ./stereo_inertial_realsense_D455_VIO_GPS_v4 ORBvoc.txt D455.yaml 37.7749 -122.4194 10.0" << std::endl;
        std::cerr << "\nWithout GPS origin (wait for ArduPilot HOME_POSITION):" << std::endl;
        std::cerr << "  ./stereo_inertial_realsense_D455_VIO_GPS_v4 ORBvoc.txt D455.yaml" << std::endl;
        std::cerr << "\nWith visualization:" << std::endl;
        std::cerr << "  ./stereo_inertial_realsense_D455_VIO_GPS_v4 ORBvoc.txt D455.yaml --vis" << std::endl;
        return 1;
    }

    bool gps_from_cmdline = (args.size() == 5);

    std::cerr << "[Main] Visualization = " << (enable_visualization ? "ON" : "OFF") << std::endl;

    // Create VIO GPS Bridge with coordinate frame bridging
    VIOGPSBridge vio_bridge;

    if (gps_from_cmdline) {
        double origin_lat = std::atof(args[2].c_str());
        double origin_lon = std::atof(args[3].c_str());
        double origin_alt = std::atof(args[4].c_str());
        vio_bridge.setGPSOrigin(origin_lat, origin_lon, origin_alt);
    }

    // Configure RealSense
    std::cerr << "[Main] Configuring RealSense D455..." << std::endl;
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

    std::cerr << "[Main] Starting RealSense pipeline..." << std::endl;
    rs2::pipeline_profile profile = pipe.start(cfg, imu_callback);

    std::vector<ORB_SLAM3::IMU::Point> vImuMeas;
    int frame_count = 0;

    std::cerr << "\n========================================" << std::endl;
    std::cerr << "Connecting to ArduPilot..." << std::endl;
    std::cerr << "========================================\n" << std::endl;

    if(!vio_bridge.waitForConnection(30)) {
        std::cerr << "[Main] MAVLink connection failed." << std::endl;
        return 1;
    }
    std::cerr << "[Main] MAVLink connected!" << std::endl;

    // Wait for GPS origin if not provided from command line
    if (!gps_from_cmdline) {
        if (!vio_bridge.waitForGPSOrigin()) {
            std::cerr << "[Main] Failed to get GPS origin from ArduPilot!" << std::endl;
            return 1;
        }
    }

    std::cerr << "\n========================================" << std::endl;
    std::cerr << "GPS Origin Ready - Starting ORB-SLAM3" << std::endl;
    std::cerr << "========================================\n" << std::endl;

    // Create SLAM system AFTER GPS origin is available
    std::cerr << "[Main] Loading ORB-SLAM3..." << std::endl;
    ORB_SLAM3::System SLAM(args[0].c_str(), args[1].c_str(), ORB_SLAM3::System::IMU_STEREO, enable_visualization);

    std::cerr << "[Main] Starting VIO GPS tracking with coordinate bridging..." << std::endl;
    std::cerr << "[Main] Move the camera to initialize the system." << std::endl;

    // Clear IMU vectors
    v_gyro_data.clear();
    v_gyro_timestamp.clear();
    v_accel_data_sync.clear();
    v_accel_timestamp_sync.clear();

    // Health check timer
    uint32_t last_health_check = getCurrentTimeMs();
    const uint32_t HEALTH_CHECK_INTERVAL_MS = 5000;

    std::cerr << "[Main] Entering main loop..." << std::endl;

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
                std::cerr << "[Main] " << count_im_buffer - 1 << " dropped frames" << std::endl;
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
            vio_bridge.updateIMUData(latest_imu_timestamp, latest_accel, latest_gyro);
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
                    std::cerr << "[Tracking] System not ready" << std::endl;
                    break;
                case ORB_SLAM3::Tracking::NO_IMAGES_YET:
                    std::cerr << "[Tracking] No images yet" << std::endl;
                    break;
                case ORB_SLAM3::Tracking::NOT_INITIALIZED:
                    std::cerr << "[Tracking] Not initialized - move camera with rotation!" << std::endl;
                    break;
                case ORB_SLAM3::Tracking::OK:
                    std::cerr << "[Tracking] OK" << std::endl;
                    break;
                case ORB_SLAM3::Tracking::RECENTLY_LOST:
                    std::cerr << "[Tracking] RECENTLY_LOST" << std::endl;
                    break;
                case ORB_SLAM3::Tracking::LOST:
                    std::cerr << "[Tracking] LOST" << std::endl;
                    break;
                case ORB_SLAM3::Tracking::OK_KLT:
                    std::cerr << "[Tracking] OK (KLT)" << std::endl;
                    break;
            }
        }

        // Process pose through VIO GPS Bridge (coordinate bridging happens here)
        vio_bridge.processORBSLAMPose(timestamp, Tcw, tracking_state, velocity);

        // Periodic health check
        uint32_t current_time = getCurrentTimeMs();
        if (current_time - last_health_check > HEALTH_CHECK_INTERVAL_MS) {
            vio_bridge.performHealthCheck();
            last_health_check = current_time;
        }

        // Clear IMU measurements for next frame
        vImuMeas.clear();
    }

    std::cerr << "[Main] Shutting down..." << std::endl;
    SLAM.Shutdown();
    return 0;
}
