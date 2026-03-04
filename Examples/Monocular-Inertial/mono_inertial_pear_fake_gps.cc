/**
 * @file mono_inertial_pear_fake_gps.cc
 * @brief Monocular-Inertial VIO with PearAPI + MAVLink GPS_INPUT for ArduPilot
 *
 * Combines:
 * - Camera/IMU/trigger logic from mono_inertial_pear.cc (PearAPI)
 * - GPS MAVLink infrastructure from stereo_inertial_realsense_D455_VIO_GPS_v4.cc
 *
 * Instead of sending VISION_POSITION_ESTIMATE messages, this version converts
 * VIO NED positions to fake GPS coordinates and sends GPS_INPUT messages to
 * ArduPilot. This allows ArduPilot to use the VIO as a GPS source.
 *
 * Hardware:
 * - OV9281 global shutter camera (via libcamera / PearAPI)
 * - BMI160 IMU via Raspberry Pi Pico (hardware-triggered camera)
 * - UART connection to ArduPilot flight controller
 *
 * Usage: ./mono_inertial_pear_fake_gps path_to_vocabulary path_to_settings
 *            [imu_serial_port] [mavlink_serial_port] [mavlink_baud]
 *            [origin_lat origin_lon origin_alt] [--vis|--novis] [--autogain]
 *
 *   Defaults: /dev/ttyACM0, /dev/ttyAMA0, 1500000, GPS origin from ArduPilot, no visualization
 *
 * Key architectural change from mono_inertial_pear.cc:
 *   NED_position = transformVIOtoNED(VIO_position) + vio_to_ned_offset
 *   GPS_position = ned_to_gps(NED_position, GPS_origin)
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
#include <vector>
#include <string>

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

// GPS quality values for NOT_INITIALIZED (no valid position, maintaining GPS link)
constexpr float NOT_INITIALIZED_EPH_EPV = 9999;   // Maximum uncertainty
constexpr uint8_t NOT_INITIALIZED_SATELLITES = 0;
constexpr uint8_t NOT_INITIALIZED_FIX_TYPE = 0;   // No fix

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

        std::cout << "[MAVLinkGPSInterface] Initializing GPS interface" << std::endl;

        serial_fd = open(serial_port, O_RDWR | O_NOCTTY);
        if (serial_fd < 0) {
            std::cout << "Failed to open serial port " << serial_port
                     << ": " << strerror(errno) << std::endl;
            exit(1);
        }

        struct termios tty;
        memset(&tty, 0, sizeof(tty));

        if (tcgetattr(serial_fd, &tty) != 0) {
            std::cout << "Error getting serial port attributes: "
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
                std::cout << "Unsupported baud rate: " << baud_rate
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
            std::cout << "Error setting serial port attributes: "
                     << strerror(errno) << std::endl;
            close(serial_fd);
            exit(1);
        }

        tcflush(serial_fd, TCIOFLUSH);

        last_waiting_msg_time = std::chrono::steady_clock::now();

        std::cout << "[MAVLink] Interface initialized on " << serial_port
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

    void setGPSOrigin(double lat, double lon, double alt) {
        std::lock_guard<std::mutex> lock(gps_origin_mutex);
        gps_origin.lat = lat;
        gps_origin.lon = lon;
        gps_origin.alt = alt;
        gps_origin_received = true;
        gps_origin_cv.notify_all();
        std::cout << "[GPS Origin] Set from command line: Lat=" << std::fixed << std::setprecision(7)
                  << lat << " Lon=" << lon << " Alt=" << std::setprecision(2) << alt << "m" << std::endl;
    }

    bool waitForGPSOrigin(int timeout_seconds = 0) {
        std::unique_lock<std::mutex> lock(gps_origin_mutex);

        if (gps_origin_received) {
            return true;
        }

        std::cout << "\n========================================" << std::endl;
        std::cout << "Waiting for GPS origin from ArduPilot..." << std::endl;
        std::cout << "Listening for HOME_POSITION or GPS_GLOBAL_ORIGIN messages" << std::endl;
        std::cout << "========================================\n" << std::endl;

        if (timeout_seconds > 0) {
            return gps_origin_cv.wait_for(lock, std::chrono::seconds(timeout_seconds),
                                          [this] { return gps_origin_received.load(); });
        } else {
            // Wait indefinitely with periodic status messages
            while (!gps_origin_received) {
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_waiting_msg_time).count();

                if (elapsed >= 30) {
                    std::cout << "[Status] Still waiting for GPS origin from ArduPilot..." << std::endl;
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
        std::cout << "[MAVLink] Waiting for connection";

        std::unique_lock<std::mutex> lock(connected_mutex);

        for (int i = 0; i < timeout_seconds; i++) {
            if (connected_cv.wait_for(lock, std::chrono::seconds(1),
                                     [this] { return connected.load(); })) {
                std::cout << " Connected!" << std::endl;
                return true;
            }
            std::cout << ".";
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

        std::cout << "[MAVLink] Requested data stream from ArduPilot" << std::endl;
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

        std::cout << "[MAVLink] Requested HOME_POSITION from ArduPilot" << std::endl;
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

        std::cout << "[MAVLink] Requested LOCAL_POSITION_NED stream at 20Hz" << std::endl;
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
            std::cout << "[MAVLink] Failed to send GPS input: " << strerror(errno) << std::endl;
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

                    std::cout << "\n========================================" << std::endl;
                    std::cout << "[GPS Origin] Received HOME_POSITION from ArduPilot" << std::endl;
                    std::cout << "Lat: " << std::fixed << std::setprecision(7) << gps_origin.lat
                              << " Lon: " << gps_origin.lon
                              << " Alt: " << std::setprecision(2) << gps_origin.alt << "m" << std::endl;
                    std::cout << "========================================\n" << std::endl;
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

                    std::cout << "\n========================================" << std::endl;
                    std::cout << "[GPS Origin] Received GPS_GLOBAL_ORIGIN from ArduPilot" << std::endl;
                    std::cout << "Lat: " << std::fixed << std::setprecision(7) << gps_origin.lat
                              << " Lon: " << gps_origin.lon
                              << " Alt: " << std::setprecision(2) << gps_origin.alt << "m" << std::endl;
                    std::cout << "========================================\n" << std::endl;
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
                if (++log_counter % 200 == 0) {
                    std::cout << "[EKF Feedback] Position: (" << std::fixed << std::setprecision(2)
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
    VIOGPSBridge(const std::string& serial_port, int baud_rate) {
        // Set unbuffered stdout for real-time logging
        setvbuf(stdout, NULL, _IONBF, 0);

        // Get system start time
        auto now = std::chrono::high_resolution_clock::now();
        start_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
            now.time_since_epoch()).count();

        std::cout << "==================================================" << std::endl;
        std::cout << "[VIOGPSBridge] Initializing with coordinate bridging" << std::endl;
        std::cout << "  (PearAPI mono-inertial + GPS_INPUT output)" << std::endl;
        std::cout << "==================================================" << std::endl;

        // Initialize MAVLink GPS interface with parameterized serial port and baud rate
        mavlink = std::make_shared<MAVLinkGPSInterface>(serial_port.c_str(), baud_rate);
        mavlink->start();
    }

    ~VIOGPSBridge() {
        if (mavlink) {
            mavlink->stop();
        }

        std::cout << "==================================================" << std::endl;
        std::cout << "[VIOGPSBridge] Shutdown statistics:" << std::endl;
        std::cout << "  Messages sent: " << state.messages_sent << std::endl;
        std::cout << "  Map transitions: " << state.map_transitions << std::endl;
        std::cout << "  Tracking loss events: " << state.tracking_loss_events << std::endl;
        std::cout << "==================================================" << std::endl;
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
            std::cout << "[VIOGPSBridge] Failed to connect to ArduPilot!" << std::endl;
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
                // Send GPS with no fix to maintain ArduPilot GPS link
                handleTrackingNotInitialized();
                break;
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
            std::cout << "[VIOGPSBridge] RECENTLY_LOST - sending high-uncertainty GPS estimate" << std::endl;
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
            std::cout << "[VIOGPSBridge] LOST - sending very degraded GPS estimate" << std::endl;
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
            // No valid position ever received - send origin position with no fix
            static bool no_pos_logged = false;
            if (!no_pos_logged) {
                std::cout << "[VIOGPSBridge] LOST but no valid position - sending origin with no-fix" << std::endl;
                no_pos_logged = true;
            }
            sendDegradedGPSAtOrigin();
        }
    }

    /**
     * Send degraded GPS at origin when no valid position is available
     * Used when LOST state occurs before first valid pose
     */
    void sendDegradedGPSAtOrigin() {
        GPSCoord gps_origin = mavlink->getGPSOrigin();

        auto now = std::chrono::high_resolution_clock::now();
        uint64_t timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
            now.time_since_epoch()).count();

        GPSData gps_data;
        gps_data.time_usec = timestamp_us;
        gps_data.lat = static_cast<int32_t>(gps_origin.lat * 1e7);
        gps_data.lon = static_cast<int32_t>(gps_origin.lon * 1e7);
        gps_data.alt = static_cast<int32_t>(gps_origin.alt * 1000);
        gps_data.vn = 0.0f;
        gps_data.ve = 0.0f;
        gps_data.vd = 0.0f;
        gps_data.vel = 0;
        gps_data.vz = 0;
        gps_data.cog = 0;
        gps_data.fix_type = NOT_INITIALIZED_FIX_TYPE;
        gps_data.satellites_visible = NOT_INITIALIZED_SATELLITES;
        gps_data.eph = NOT_INITIALIZED_EPH_EPV;
        gps_data.epv = NOT_INITIALIZED_EPH_EPV;
        gps_data.reset_counter = state.reset_counter;

        mavlink->queueGPS(gps_data);
        state.messages_sent++;
    }

    /**
     * Handle states where tracking has never been initialized
     * Send GPS with no fix to maintain ArduPilot GPS link
     */
    void handleTrackingNotInitialized() {
        static bool logged = false;
        static int last_state = -1;

        if (!logged || last_state != state.tracking_state_current) {
            std::cout << "[VIOGPSBridge] NOT INITIALIZED (state=" << state.tracking_state_current
                      << ") - sending no-fix GPS to maintain link" << std::endl;
            logged = true;
            last_state = state.tracking_state_current;
        }

        // Send GPS with no fix to maintain ArduPilot GPS link
        GPSCoord gps_origin = mavlink->getGPSOrigin();

        auto now = std::chrono::high_resolution_clock::now();
        uint64_t timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
            now.time_since_epoch()).count();

        GPSData gps_data;
        gps_data.time_usec = timestamp_us;
        gps_data.lat = static_cast<int32_t>(gps_origin.lat * 1e7);
        gps_data.lon = static_cast<int32_t>(gps_origin.lon * 1e7);
        gps_data.alt = static_cast<int32_t>(gps_origin.alt * 1000);
        gps_data.vn = 0.0f;
        gps_data.ve = 0.0f;
        gps_data.vd = 0.0f;
        gps_data.vel = 0;
        gps_data.vz = 0;
        gps_data.cog = 0;
        gps_data.fix_type = NOT_INITIALIZED_FIX_TYPE;
        gps_data.satellites_visible = NOT_INITIALIZED_SATELLITES;
        gps_data.eph = NOT_INITIALIZED_EPH_EPV;
        gps_data.epv = NOT_INITIALIZED_EPH_EPV;
        gps_data.reset_counter = state.reset_counter;

        mavlink->queueGPS(gps_data);
        state.messages_sent++;
    }

    /**
     * Detect map change using heuristic (position discontinuity after tracking recovery)
     */
    bool detectMapChange(const Eigen::Vector3d& new_vio_position) {
        // Case 1: First pose ever - not a map change
        if (!state.first_pose_received) {
            state.first_pose_received = true;
            state.vio_initialized = true;
            std::cout << "[VIOGPSBridge] VIO initialized with first pose" << std::endl;
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
                std::cout << "[VIOGPSBridge] Map change detected! Position jump: "
                          << std::fixed << std::setprecision(2) << pos_jump << "m" << std::endl;
                return true;
            } else {
                std::cout << "[VIOGPSBridge] Tracking recovered in same map (jump: "
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

            std::cout << "[VIOGPSBridge] Using EKF feedback for offset: ("
                      << std::fixed << std::setprecision(2)
                      << reference_position.x() << ", "
                      << reference_position.y() << ", "
                      << reference_position.z() << ")" << std::endl;
        } else {
            // Fall back to last valid NED position
            reference_position = state.last_valid_ned_position;

            std::cout << "[VIOGPSBridge] No recent EKF feedback, using last NED position: ("
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

        std::cout << "[VIOGPSBridge] *** MAP TRANSITION #" << state.map_transitions << " ***" << std::endl;
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
     * Transform VIO position to NED frame
     * ORB-SLAM3 mono-inertial world frame (after gravity alignment):
     *   X: right, Y: forward, Z: up (gravity = -Z)
     * NED: X-north/forward, Y-east/right, Z-down
     */
    Eigen::Vector3d transformVIOtoNED(const Eigen::Vector3d& vio_position) {
        return Eigen::Vector3d(
            vio_position.y(),   //  Y -> X (forward -> north)
            vio_position.x(),   //  X -> Y (right -> east)
            -vio_position.z()   // -Z -> Z (negate up -> down)
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
     * Transform velocity to NED frame (same frame rotation as position)
     */
    Eigen::Vector3d applyVelocityTransform(const Eigen::Vector3d& vio_velocity) {
        return Eigen::Vector3d(
            vio_velocity.y(),   //  Y -> X (forward -> north)
            vio_velocity.x(),   //  X -> Y (right -> east)
            -vio_velocity.z()   // -Z -> Z (negate up -> down)
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
            std::cout << "[GPS->MAV] Lat=" << std::fixed << std::setprecision(7)
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
                std::cout << "[VIOGPSBridge] WARNING: No pose data for " << pose_age << "ms" << std::endl;
                healthy = false;
            }
        }

        // Check 2: EKF feedback (warning only, not critical)
        if (!mavlink->hasRecentEKFFeedback()) {
            static int warning_count = 0;
            if (++warning_count % 100 == 0) {
                std::cout << "[VIOGPSBridge] INFO: No recent EKF feedback" << std::endl;
            }
        }

        // Check 3: Position error (if we have EKF feedback)
        if (mavlink->hasRecentEKFFeedback() && state.first_pose_received) {
            EKFPositionFeedback ekf = mavlink->getEKFPosition();
            double position_error = (state.last_valid_ned_position - ekf.position).norm();
            if (position_error > 5.0) {
                std::cout << "[VIOGPSBridge] WARNING: Large position error: "
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

int main(int argc, char** argv) {
    // Set unbuffered stdout for real-time output
    setvbuf(stdout, NULL, _IONBF, 0);

    // Extract flags first
    bool enable_visualization = false;
    bool enable_autogain = false;
    std::vector<std::string> args;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--vis") {
            enable_visualization = true;
        } else if (arg == "--novis") {
            enable_visualization = false;
        } else if (arg == "--autogain") {
            enable_autogain = true;
        } else {
            args.push_back(arg);
        }
    }

    // Validate remaining arguments
    // Valid counts: 2, 3, 4, 5, 8
    if (args.size() < 2 || args.size() == 6 || args.size() == 7 || args.size() > 8) {
        cout << endl
             << "Usage: ./mono_inertial_pear_fake_gps path_to_vocabulary path_to_settings"
             << endl
             << "           [imu_serial_port] [mavlink_serial_port] [mavlink_baud]"
             << endl
             << "           [origin_lat origin_lon origin_alt] [--vis|--novis]"
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
             << "    GPS origin:          From ArduPilot HOME_POSITION (if not specified)"
             << endl
             << "    visualization:       OFF (use --vis to enable Pangolin viewer)"
             << endl
             << "    --autogain:          Run auto gain tuning on startup"
             << endl
             << endl
             << "  Examples:"
             << endl
             << "    # All defaults, GPS origin from ArduPilot:"
             << endl
             << "    ./mono_inertial_pear_fake_gps ORBvoc.txt settings.yaml"
             << endl
             << endl
             << "    # Custom ports, GPS origin from ArduPilot:"
             << endl
             << "    ./mono_inertial_pear_fake_gps ORBvoc.txt settings.yaml /dev/ttyACM0 /dev/ttyAMA0 1500000"
             << endl
             << endl
             << "    # All ports + GPS origin from command line:"
             << endl
             << "    ./mono_inertial_pear_fake_gps ORBvoc.txt settings.yaml /dev/ttyACM0 /dev/ttyAMA0 1500000 37.7749 -122.4194 10.0"
             << endl
             << endl
             << "    # With visualization:"
             << endl
             << "    ./mono_inertial_pear_fake_gps ORBvoc.txt settings.yaml --vis"
             << endl;
        return 1;
    }

    string vocabularyPath = args[0];
    string settingsPath = args[1];
    string imuSerialPort = (args.size() >= 3) ? args[2] : "/dev/ttyACM0";
    string mavlinkSerialPort = (args.size() >= 4) ? args[3] : "/dev/ttyAMA0";
    int mavlinkBaud = (args.size() >= 5) ? atoi(args[4].c_str()) : 1500000;

    bool gps_from_cmdline = (args.size() == 8);
    double origin_lat = 0, origin_lon = 0, origin_alt = 0;
    if (gps_from_cmdline) {
        origin_lat = std::atof(args[5].c_str());
        origin_lon = std::atof(args[6].c_str());
        origin_alt = std::atof(args[7].c_str());
    }

    // Setup signal handler
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = exit_loop_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);

    cout << "========================================" << endl;
    cout << "Monocular-Inertial VIO GPS (PearAPI + GPS_INPUT)" << endl;
    cout << "========================================" << endl;
    cout << "Vocabulary:      " << vocabularyPath << endl;
    cout << "Settings:        " << settingsPath << endl;
    cout << "IMU Port:        " << imuSerialPort << endl;
    cout << "MAVLink Port:    " << mavlinkSerialPort << endl;
    cout << "MAVLink Baud:    " << mavlinkBaud << endl;
    cout << "GPS Origin:      " << (gps_from_cmdline ?
        ("CLI: " + to_string(origin_lat) + ", " + to_string(origin_lon) + ", " + to_string(origin_alt)) :
        "From ArduPilot HOME_POSITION") << endl;
    cout << "Visualization:   " << (enable_visualization ? "ON" : "OFF") << endl;
    cout << "Auto Gain:       " << (enable_autogain ? "ON" : "OFF") << endl;
    cout << "========================================" << endl;

    // ---- Initialize IMU reader using PearAPI ----
    pearvio::IMUReader imuReader;
    if (!imuReader.open(imuSerialPort)) {
        cout << "Failed to open IMU serial port: " << imuSerialPort << endl;
        cout << "Check that:" << endl;
        cout << "  - Pico is connected and running" << endl;
        cout << "  - User has dialout group access: sudo usermod -aG dialout $USER" << endl;
        return 1;
    }

    // ---- Initialize camera using PearAPI ----
    auto camera = pearvio::CameraBackend::create();
    if (!camera) {
        cout << "Failed to create camera backend" << endl;
        return 1;
    }

    pearvio::CameraConfig camConfig;
    camConfig.cameraIndex = 0;
    camConfig.fps = 20;  // 20 FPS for triggered mode

    // Load camera settings from PearCameraApp config (resolution, exposure, gain, etc.)
    if (camConfig.loadFromIniFile()) {
        cout << "Loaded camera settings from config file" << endl;
        cout << "  Resolution: " << camConfig.width << "x" << camConfig.height << endl;
    } else {
        cout << "No config file found, using defaults ("
             << camConfig.width << "x" << camConfig.height << ")" << endl;
    }

    // VIO always needs trigger mode for hardware sync
    camConfig.triggerMode = true;

    if (!camera->initialize(camConfig)) {
        cout << "Failed to initialize camera" << endl;
        return 1;
    }

    cout << "Camera initialized: " << camera->frameWidth() << "x" << camera->frameHeight() << endl;

    if (!camera->start()) {
        cout << "Failed to start camera" << endl;
        return 1;
    }

    // Apply camera settings AFTER start() - start() reinitializes the sensor,
    // so V4L2 control writes must come after to avoid being reset.
    camera->setTriggerMode(camConfig.triggerMode);
    camera->setAutoExposure(camConfig.autoExposure);
    if (!camConfig.autoExposure) {
        camera->setGain(camConfig.gain);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        camera->setExposureTime(camConfig.exposureTimeUs);
    }
    cout << "Camera settings applied: trigger=" << camConfig.triggerMode
         << " autoExpo=" << camConfig.autoExposure
         << " exposure=" << camera->exposureTime() << "us"
         << " gain=" << camera->gain() << endl;

    // ---- Auto-tune gain if requested via --autogain flag ----
    if (enable_autogain) {
        cout << "Running auto gain..." << endl;
        pearvio::AutoGainConfig agc;
        auto result = camera->autoGain(agc);
        if (result.success) {
            cout << "Auto gain: " << result.gain << " (brightness=" << result.brightness
                 << ", iterations=" << result.iterations << ")" << endl;
            camConfig.gain = result.gain;
            camConfig.autoExposure = false;
            camConfig.saveToIniFile();
        } else {
            cout << "Auto gain failed, using current gain: " << camera->gain() << endl;
        }
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

    // ---- Create VIO GPS Bridge for MAVLink/ArduPilot integration ----
    VIOGPSBridge vio_bridge(mavlinkSerialPort, mavlinkBaud);

    // Set GPS origin from command line if provided
    if (gps_from_cmdline) {
        vio_bridge.setGPSOrigin(origin_lat, origin_lon, origin_alt);
    }

    // Wait for MAVLink connection (fatal if fails)
    cout << "\n========================================" << endl;
    cout << "Connecting to ArduPilot..." << endl;
    cout << "========================================\n" << endl;

    if (!vio_bridge.waitForConnection(30)) {
        cout << "[Main] MAVLink connection failed." << endl;
        camera->setTriggerMode(false);
        camera->stop();
        imuReader.close();
        return 1;
    }
    cout << "[Main] MAVLink connected!" << endl;

    // Wait for GPS origin if not provided from command line
    if (!gps_from_cmdline) {
        if (!vio_bridge.waitForGPSOrigin()) {
            cout << "[Main] Failed to get GPS origin from ArduPilot!" << endl;
            camera->setTriggerMode(false);
            camera->stop();
            imuReader.close();
            return 1;
        }
    }

    cout << "\n========================================" << endl;
    cout << "GPS Origin Ready - Starting ORB-SLAM3" << endl;
    cout << "========================================\n" << endl;

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

    cout << "[Main] Starting VIO GPS tracking with coordinate bridging..." << endl;
    cout << "[Main] Move the camera to initialize the system." << endl;
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

        // Latest IMU sample for VIOGPSBridge angular velocity (in SI units)
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

            // Track latest for VIOGPSBridge
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

        // Update VIOGPSBridge with latest IMU data (SI units: rad/s, m/s^2)
        if (!imuData.empty()) {
            vio_bridge.updateIMUData(lastImuTimestamp, latestAccelSI, latestGyroSI);
        }

        // Process pose through VIO GPS Bridge (coordinate bridging + GPS_INPUT happens here)
        vio_bridge.processORBSLAMPose(frameTime, Tcw, tracking_state, velocity);

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

        // Display tracking status every 30 frames
        if (frameCount % 30 == 0) {
            switch(tracking_state) {
                case ORB_SLAM3::Tracking::SYSTEM_NOT_READY:
                    cout << "[Tracking] System not ready" << endl;
                    break;
                case ORB_SLAM3::Tracking::NO_IMAGES_YET:
                    cout << "[Tracking] No images yet" << endl;
                    break;
                case ORB_SLAM3::Tracking::NOT_INITIALIZED:
                    cout << "[Tracking] Not initialized - move camera with rotation!" << endl;
                    break;
                case ORB_SLAM3::Tracking::OK:
                    cout << "[Tracking] OK" << endl;
                    break;
                case ORB_SLAM3::Tracking::RECENTLY_LOST:
                    cout << "[Tracking] RECENTLY_LOST" << endl;
                    break;
                case ORB_SLAM3::Tracking::LOST:
                    cout << "[Tracking] LOST" << endl;
                    break;
                case ORB_SLAM3::Tracking::OK_KLT:
                    cout << "[Tracking] OK (KLT)" << endl;
                    break;
            }
        }

        // Print statistics every 100 frames
        if (frameCount % 100 == 0) {
            auto now = chrono::steady_clock::now();
            double elapsed = chrono::duration<double>(now - startTime).count();
            double fps = frameCount / elapsed;

            cout << "Frames: " << frameCount
                 << " | IMU: " << imuCount
                 << " | FPS: " << fixed << setprecision(1) << fps
                 << " | GPS sent: " << vio_bridge.getMessagesSent()
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
    cout << "GPS messages sent: " << vio_bridge.getMessagesSent() << endl;
    cout << "Map transitions:   " << vio_bridge.getMapTransitions() << endl;
    cout << "Tracking losses:   " << vio_bridge.getTrackingLossEvents() << endl;
    cout << "========================================" << endl;

    // Save trajectory
    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
    cout << "Trajectories saved to CameraTrajectory.txt and KeyFrameTrajectory.txt" << endl;

    return 0;
}
