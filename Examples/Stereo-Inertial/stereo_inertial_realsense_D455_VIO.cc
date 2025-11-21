#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include <iomanip>
#include <Eigen/Dense>
#include "System.h"

// Interpolate IMU measurements to match gyro timestamps
rs2_vector interpolateMeasure(const double target_time,
                              const rs2_vector current_data, const double current_time,
                              const rs2_vector prev_data, const double prev_time)
{
    // If there are not previous information, the current data is propagated
    if(prev_time == 0)
    {
        return current_data;
    }

    rs2_vector increment;
    rs2_vector value_interp;

    if(target_time > current_time) {
        value_interp = current_data;
    }
    else if(target_time > prev_time)
    {
        increment.x = current_data.x - prev_data.x;
        increment.y = current_data.y - prev_data.y;
        increment.z = current_data.z - prev_data.z;

        double factor = (target_time - prev_time) / (current_time - prev_time);

        value_interp.x = prev_data.x + increment.x * factor;
        value_interp.y = prev_data.y + increment.y * factor;
        value_interp.z = prev_data.z + increment.z * factor;

        // Use current data (zero-order hold)
        value_interp = current_data;
    }
    else
    {
        value_interp = prev_data;
    }

    return value_interp;
}

//==========================================================
#include <cstring>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <errno.h>
#include <cmath>
#include <ctime>
#include <random>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <memory>
// Include MAVLink headers
#include <common/mavlink.h>

// MAVLink transmission mode
enum class MAVLinkMode {
    ODOMETRY,                    // Send ODOMETRY messages (default)
    VISION_POSITION_ESTIMATE,    // Send VISION_POSITION_ESTIMATE messages
    VISION_POSITION_AND_SPEED    // Send both VISION_POSITION_ESTIMATE and VISION_SPEED_ESTIMATE
};

// Structure to hold odometry data
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

// Thread-safe queue for odometry data
class OdometryQueue {
private:
    std::queue<OdometryData> queue;
    std::mutex mutex;
    std::condition_variable cv;
    const size_t max_size = 100;  // Prevent queue from growing too large
    
public:
    void push(const OdometryData& data) {
        std::unique_lock<std::mutex> lock(mutex);
        
        // If queue is full, remove oldest element
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

class MAVLinkInterface {
private:
    int serial_fd;
    uint8_t system_id;      // Read from ArduPilot heartbeat (starts as 255, then adopts ArduPilot's ID)
    uint8_t component_id;   // Set to 191 (MAV_COMP_ID_VISUAL_INERTIAL_ODOMETRY) upon connection
    MAVLinkMode tx_mode;    // Transmission mode (ODOMETRY or VISION_POSITION_ESTIMATE)

    std::atomic<bool> running;
    std::thread mavlink_thread;
    OdometryQueue odometry_queue;
    std::atomic<bool> connected;
    std::mutex connected_mutex;
    std::condition_variable connected_cv;

public:
    MAVLinkInterface(const char* serial_port, int baud_rate = 57600, MAVLinkMode mode = MAVLinkMode::ODOMETRY)
        : system_id(255), component_id(197), running(false), connected(false), tx_mode(mode) {

        std::cout << "[DEBUG MAVLinkInterface] Constructor called with mode = " << static_cast<int>(mode)
                  << " (0=ODOMETRY, 1=VISION_POS, 2=VISION_POS+SPEED)" << std::endl;
        std::cout << "[DEBUG MAVLinkInterface] tx_mode set to = " << static_cast<int>(tx_mode) << std::endl;

        // Open serial port
        serial_fd = open(serial_port, O_RDWR | O_NOCTTY);
        if (serial_fd < 0) {
            std::cerr << "Failed to open serial port " << serial_port 
                     << ": " << strerror(errno) << std::endl;
            exit(1);
        }
        
        // Configure serial port
        struct termios tty;
        memset(&tty, 0, sizeof(tty));
        
        if (tcgetattr(serial_fd, &tty) != 0) {
            std::cerr << "Error getting serial port attributes: " 
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
                std::cerr << "Unsupported baud rate: " << baud_rate 
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
            std::cerr << "Error setting serial port attributes: " 
                     << strerror(errno) << std::endl;
            close(serial_fd);
            exit(1);
        }
        
        tcflush(serial_fd, TCIOFLUSH);
        
        std::cout << "MAVLink interface initialized" << std::endl;
        std::cout << "Serial port: " << serial_port << std::endl;
        std::cout << "Baud rate: " << baud_rate << std::endl;
    }
    
    ~MAVLinkInterface() {
        stop();
        close(serial_fd);
    }
    
    // Start the MAVLink thread
    void start() {
        if (!running) {
            running = true;
            mavlink_thread = std::thread(&MAVLinkInterface::run, this);
            std::cout << "MAVLink thread started" << std::endl;
        }
    }
    
    // Stop the MAVLink thread
    void stop() {
        if (running) {
            running = false;
            if (mavlink_thread.joinable()) {
                mavlink_thread.join();
            }
            std::cout << "MAVLink thread stopped" << std::endl;
        }
    }
    
    // Queue odometry data to be sent (called from main thread)
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
    
    // Get queue size (for monitoring)
    size_t getQueueSize() const {
        return const_cast<OdometryQueue&>(odometry_queue).size();
    }
    
    // Check if connected to ArduPilot
    bool isConnected() const {
        return connected;
    }
    
    // Wait for connection to be established
    bool waitForConnection(int timeout_seconds = 10) {
        std::cout << "Waiting for MAVLink connection";
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
    
private:
    // MAVLink thread main loop
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
            if (odometry_queue.pop(odom_data, 10)) {  // 10ms timeout
                sendData(odom_data);  // Use mode-based sending
            }

            // Receive and process messages
            receiveMessages();

            usleep(50);  // 10ms
            heartbeat_counter++;
        }
    }
    
    // Send heartbeat message
    void sendHeartbeat() {
        mavlink_message_t msg;
        uint8_t buf[MAVLINK_MAX_PACKET_LEN];
        
        mavlink_msg_heartbeat_pack(system_id, component_id, &msg,
                                   MAV_TYPE_ONBOARD_CONTROLLER,  // Type: Onboard controller
                                   MAV_AUTOPILOT_INVALID,        // Autopilot type
                                   0,                             // Base mode
                                   0,                             // Custom mode
                                   MAV_STATE_ACTIVE);             // System status
        
        uint16_t len = mavlink_msg_to_send_buffer(buf, &msg);
        
        ssize_t sent = write(serial_fd, buf, len);
        
        if (sent < 0) {
            std::cerr << "Failed to send heartbeat: " << strerror(errno) << std::endl;
        }
    }
    
    // Request data stream
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
        
        std::cout << "Requested data stream" << std::endl;
    }
    
    // Send odometry data from queued struct
    void sendOdometry(const OdometryData& data) {
        mavlink_message_t msg;
        uint8_t buf[MAVLINK_MAX_PACKET_LEN];

        mavlink_msg_odometry_pack(
            system_id,
            component_id,
            &msg,
            data.time_usec,
            MAV_FRAME_LOCAL_NED,     // Frame of reference
            MAV_FRAME_BODY_FRD,     // Child frame
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
            std::cerr << "Failed to send odometry: " << strerror(errno) << std::endl;
        } else {
            std::cout << "Sent odometry | Pos: (" << data.x << ", " << data.y << ", " << data.z
                     << ") | Vel: (" << data.vx << ", " << data.vy << ", " << data.vz
                     << ") | Q-size: " << getQueueSize() << std::endl;
        }
    }

    // Send vision position estimate from queued odometry data
    void sendVisionPositionEstimate(const OdometryData& data) {
        std::cout << "[DEBUG] Entering sendVisionPositionEstimate" << std::endl;

        // Initialize structures to zero for safety
        mavlink_message_t msg;
        memset(&msg, 0, sizeof(msg));
        uint8_t buf[MAVLINK_MAX_PACKET_LEN];
        memset(buf, 0, sizeof(buf));

        std::cout << "[DEBUG] Converting quaternion to yaw" << std::endl;
        // Convert quaternion to Euler angles for yaw
        float qw = data.q[0];
        float qx = data.q[1];
        float qy = data.q[2];
        float qz = data.q[3];

        // Validate quaternion values
        if (!std::isfinite(qw) || !std::isfinite(qx) || !std::isfinite(qy) || !std::isfinite(qz)) {
            std::cerr << "[ERROR] Invalid quaternion values: qw=" << qw << " qx=" << qx
                     << " qy=" << qy << " qz=" << qz << std::endl;
            return;
        }

        // Calculate yaw from quaternion
        float yaw = atan2(2.0f * (qw * qz + qx * qy), 1.0f - 2.0f * (qy * qy + qz * qz));

        // Validate position and yaw
        if (!std::isfinite(data.x) || !std::isfinite(data.y) || !std::isfinite(data.z) || !std::isfinite(yaw)) {
            std::cerr << "[ERROR] Invalid position or yaw: x=" << data.x << " y=" << data.y
                     << " z=" << data.z << " yaw=" << yaw << std::endl;
            return;
        }

        std::cout << "[DEBUG] About to pack VISION_POSITION_ESTIMATE message" << std::endl;
        std::cout << "[DEBUG] system_id=" << (int)system_id << " component_id=" << (int)component_id << std::endl;
        std::cout << "[DEBUG] pos=(" << data.x << "," << data.y << "," << data.z << ") yaw=" << yaw << std::endl;

        // Pack VISION_POSITION_ESTIMATE message
        // Note: Using MAVLink v2 common message set function signature
        std::cout << "[DEBUG] Calling mavlink_msg_vision_position_estimate_pack..." << std::endl;
        uint16_t msg_len = mavlink_msg_vision_position_estimate_pack(
            system_id,
            component_id,
            &msg,
            data.time_usec,           // Timestamp (microseconds)
            data.x,                   // X position (m)
            data.y,                   // Y position (m)
            data.z,                   // Z position (m)
            0.0f,                     // Roll angle (rad)
            0.0f,                     // Pitch angle (rad)
            yaw,                       // Yaw angle (rad)
            data.pose_covariance,
            data.reset_counter
        );
        std::cout << "[DEBUG] Message packed successfully, return value=" << msg_len << std::endl;

        std::cout << "[DEBUG] Converting to buffer..." << std::endl;
        uint16_t len = mavlink_msg_to_send_buffer(buf, &msg);
        std::cout << "[DEBUG] Buffer created, len=" << len << std::endl;

        std::cout << "[DEBUG] Writing to serial, fd=" << serial_fd << std::endl;
        ssize_t sent = write(serial_fd, buf, len);

        if (sent < 0) {
            std::cerr << "Failed to send vision position estimate: " << strerror(errno) << std::endl;
        } else {
            std::cout << "Sent vision position estimate | Pos: (" << data.x << ", " << data.y << ", " << data.z
                     << ") | Yaw: " << yaw
                     << " | Q-size: " << getQueueSize() << std::endl;
        }
        std::cout << "[DEBUG] Exiting sendVisionPositionEstimate" << std::endl;
    }

    // Send vision speed estimate from queued odometry data
    void sendVisionSpeedEstimate(const OdometryData& data) {
        std::cout << "[DEBUG] Entering sendVisionSpeedEstimate" << std::endl;

        // Initialize structures to zero for safety
        mavlink_message_t msg;
        memset(&msg, 0, sizeof(msg));
        uint8_t buf[MAVLINK_MAX_PACKET_LEN];
        memset(buf, 0, sizeof(buf));

        // Validate velocity values
        if (!std::isfinite(data.vx) || !std::isfinite(data.vy) || !std::isfinite(data.vz)) {
            std::cerr << "[ERROR] Invalid velocity values: vx=" << data.vx << " vy=" << data.vy
                     << " vz=" << data.vz << std::endl;
            return;
        }

        std::cout << "[DEBUG] About to pack VISION_SPEED_ESTIMATE message" << std::endl;

        // Pack VISION_SPEED_ESTIMATE message
        // Note: Using MAVLink v2 common message set function signature
        std::cout << "[DEBUG] Calling mavlink_msg_vision_speed_estimate_pack..." << std::endl;
        uint16_t msg_len = mavlink_msg_vision_speed_estimate_pack(
            system_id,
            component_id,
            &msg,
            data.time_usec,           // Timestamp (microseconds)
            data.vx,                  // X velocity (m/s)
            data.vy,                  // Y velocity (m/s)
            data.vz,                  // Z velocity (m/s)
            nullptr,                  // Covariance (not used)
            data.reset_counter        // Reset counter
        );
        std::cout << "[DEBUG] Message packed successfully, return value=" << msg_len << std::endl;

        std::cout << "[DEBUG] Converting to buffer..." << std::endl;
        uint16_t len = mavlink_msg_to_send_buffer(buf, &msg);
        std::cout << "[DEBUG] Buffer created, len=" << len << std::endl;

        std::cout << "[DEBUG] Writing to serial, fd=" << serial_fd << std::endl;
        ssize_t sent = write(serial_fd, buf, len);

        if (sent < 0) {
            std::cerr << "Failed to send vision speed estimate: " << strerror(errno) << std::endl;
        } else {
            std::cout << "Sent vision speed estimate | Vel: (" << data.vx << ", " << data.vy << ", " << data.vz
                     << ") | Q-size: " << getQueueSize() << std::endl;
        }
        std::cout << "[DEBUG] Exiting sendVisionSpeedEstimate" << std::endl;
    }

    // Send data based on configured mode
    void sendData(const OdometryData& data) {
        std::cout << "[DEBUG sendData] tx_mode = " << static_cast<int>(tx_mode) << " (0=ODOMETRY, 1=VISION_POS, 2=VISION_POS+SPEED)" << std::endl;
        switch (tx_mode) {
            case MAVLinkMode::ODOMETRY:
                std::cout << "[DEBUG sendData] Calling sendOdometry()" << std::endl;
                sendOdometry(data);
                break;
            case MAVLinkMode::VISION_POSITION_ESTIMATE:
                std::cout << "[DEBUG sendData] Calling sendVisionPositionEstimate()" << std::endl;
                sendVisionPositionEstimate(data);
                break;
            case MAVLinkMode::VISION_POSITION_AND_SPEED:
                std::cout << "[DEBUG sendData] Calling sendVisionPositionEstimate() and sendVisionSpeedEstimate()" << std::endl;
                sendVisionPositionEstimate(data);
                sendVisionSpeedEstimate(data);
                break;
        }
    }

    // Receive and process messages
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
				std::cout << "msg.sysid: " << (int)msg.sysid<<"; heartbeat.autopilot: "<< (int)heartbeat.autopilot << std::endl;
                // Check if this is from ArduPilot (not from ourselves)
                if (msg.sysid != system_id && heartbeat.autopilot != MAV_AUTOPILOT_INVALID) {
                    std::cout << "1111111 Maybe Connect\n";
                    if (!connected) {
                        std::cout << "2222 Not connected yet but trying\n";
                        // Adopt ArduPilot's system ID
                        system_id = msg.sysid;
                        component_id = MAV_COMP_ID_VISUAL_INERTIAL_ODOMETRY;//191;  // MAV_COMP_ID_VISUAL_INERTIAL_ODOMETRY
                        
                        std::cout << "33333 \n";
                        std::lock_guard<std::mutex> lock(connected_mutex);
                        std::cout << "44444 \n";
                        connected = true;
                        std::cout << "555555 \n";
                        connected_cv.notify_all();
                        std::cout << "666666 \n";
                        std::cout << "MAVLink connection established with system " 
                                 << (int)msg.sysid << std::endl;
                        std::cout << "Adopted system_id=" << (int)system_id 
                                 << ", component_id=" << (int)component_id << std::endl;
                    }
                }
                
                std::cout << "Heartbeat received from system " << (int)msg.sysid 
                         << " | Type: " << (int)heartbeat.type 
                         << " | Autopilot: " << (int)heartbeat.autopilot 
                         << " | Mode: " << (int)heartbeat.base_mode 
                         << " | State: " << (int)heartbeat.system_status << std::endl;
                break;
            }
            
            case MAVLINK_MSG_ID_ATTITUDE: {
                mavlink_attitude_t attitude;
                mavlink_msg_attitude_decode(&msg, &attitude);
                std::cout << "Attitude | Roll: " << attitude.roll 
                         << " Pitch: " << attitude.pitch 
                         << " Yaw: " << attitude.yaw << std::endl;
                break;
            }
            
            case MAVLINK_MSG_ID_GLOBAL_POSITION_INT: {
                mavlink_global_position_int_t pos;
                mavlink_msg_global_position_int_decode(&msg, &pos);
                std::cout << "Position | Lat: " << pos.lat / 1e7 
                         << " Lon: " << pos.lon / 1e7 
                         << " Alt: " << pos.alt / 1000.0 << "m" << std::endl;
                break;
            }
            
            case MAVLINK_MSG_ID_SYS_STATUS: {
                mavlink_sys_status_t sys_status;
                mavlink_msg_sys_status_decode(&msg, &sys_status);
                std::cout << "System Status | Voltage: " << sys_status.voltage_battery / 1000.0 
                         << "V | Current: " << sys_status.current_battery / 100.0 
                         << "A | Battery: " << (int)sys_status.battery_remaining << "%" << std::endl;
                break;
            }
            
            default:
                break;
        }
    }
};
//==========================================================

class VIOLogger {
private:
    uint64_t start_time_us;
    uint8_t  reset_counter;
	
    bool isTracking;
	bool prevIsTracking;

    bool hasLastGoodCoords;
    float lastGoodPosX;
    float lastGoodPosY;
    float lastGoodPosZ;
    float lastGoodQ[4];

	std::shared_ptr<MAVLinkInterface> mavlink;
	
	    // Store latest IMU data for logging
    struct IMUData {
        double timestamp;
        Eigen::Vector3f accel;
        Eigen::Vector3f gyro;
        bool valid;
        
        IMUData() : timestamp(0), accel(0,0,0), gyro(0,0,0), valid(false){}
    } latest_imu;

public:
    VIOLogger(const std::string& filename, MAVLinkMode mode = MAVLinkMode::ODOMETRY)
        : reset_counter(0)
        , isTracking(false)
        , prevIsTracking(false)
        , hasLastGoodCoords(false)
		, lastGoodPosX(0.0f)
		, lastGoodPosY(0.0f)
		, lastGoodPosZ(0.0f)
		, lastGoodQ{ 1.0f, 0.0f, 0.0f, 0.0f }
    {

        // Get system start time
        auto now = std::chrono::high_resolution_clock::now();
        start_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
            now.time_since_epoch()).count();

        std::cout << "VIO logging started. Output file: " << filename << std::endl;
        std::cout << "MAVLink mode: ";
        if (mode == MAVLinkMode::ODOMETRY) {
            std::cout << "ODOMETRY" << std::endl;
        } else if (mode == MAVLinkMode::VISION_POSITION_ESTIMATE) {
            std::cout << "VISION_POSITION_ESTIMATE" << std::endl;
        } else {
            std::cout << "VISION_POSITION_ESTIMATE + VISION_SPEED_ESTIMATE" << std::endl;
        }

		//=========================================================================
		// MavLink
		const char* serial_port = "/dev/ttyTHS0";
		int baud_rate = 1500000;

		std::cout << "Starting MAVLink UART communication with threaded odometry" << std::endl;
		mavlink = std::make_shared<MAVLinkInterface>(serial_port, baud_rate, mode);
		mavlink->start();
    }

    ~VIOLogger() {
		if(mavlink)
			mavlink->stop();
    }

    void SetTrackingState(bool tracking) 
    {
        if (tracking == isTracking)
            return;

		prevIsTracking = isTracking;
        isTracking = tracking;
	}
	
    int IncrementResetCounter() {
        std::cout << "@@@ Increment reset counter\n";
        prevIsTracking = isTracking;
        return ++reset_counter;
	}

	bool WaitMavlink(int timeout_seconds = 10)
	{
		if (!mavlink->waitForConnection(timeout_seconds)) {  // 30 second timeout
			std::cerr << "Failed to connect to ArduPilot!" << std::endl;
			//std::cerr << "Check:" << std::endl;
			//std::cerr << "  - Serial port is correct (" << serial_port << ")" << std::endl;
			//std::cerr << "  - Baud rate matches (" << baud_rate << ")" << std::endl;
			//std::cerr << "  - ArduPilot is powered and configured" << std::endl;
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

    void logPose(double timestamp, const Sophus::SE3f& Tcw, const Eigen::Vector3f& velocity, bool tracking_good) {

        // Convert timestamp to microseconds since start
        uint64_t timestamp_us = start_time_us + static_cast<uint64_t>(timestamp * 1e6);

        // Get camera pose (inverse of Tcw gives camera position in world frame)
        Sophus::SE3f Twc = Tcw.inverse();
        Eigen::Vector3f position = Twc.translation();
        Eigen::Matrix3f rotation = Twc.rotationMatrix();

        // Convert from ORB-SLAM3 camera frame to NED frame
        // ORB-SLAM3: X-right, Y-down, Z-forward
        // NED: X-north, Y-east, Z-down
        float ned_x = position.z();   // Forward -> North
        float ned_y = -position.x();  // Right -> West, so negate for East
        float ned_z = position.y();   // Down -> Down

        // Transform velocity to NED frame (same transformation as position)
        float ned_vx = velocity.z();   // Forward -> North
        float ned_vy = -velocity.x();  // Right -> West, so negate for East
        float ned_vz = velocity.y();   // Down -> Down

        // Convert rotation matrix to Euler angles (roll, pitch, yaw)
        Eigen::Vector3f euler = rotation.eulerAngles(2, 1, 0); // ZYX order
        float roll = euler[2];
        float pitch = euler[1];
        float yaw = euler[0];

        // Transform Euler angles to NED frame
        float ned_roll = roll;
        float ned_pitch = -pitch;
        float ned_yaw = yaw + M_PI/2;

        // Normalize yaw to [-pi, pi]
        while (ned_yaw > M_PI) ned_yaw -= 2*M_PI;
        while (ned_yaw < -M_PI) ned_yaw += 2*M_PI;
				
		// Extract orientation as quaternion
		Eigen::Quaternionf quaternion = Twc.unit_quaternion();

		// Access quaternion components
		float qw = quaternion.w();  // scalar part
		float qx = quaternion.x();  // vector part x
		float qy = quaternion.y();  // vector part y
		float qz = quaternion.z();  // vector part z
		float q[4] = {qw, qx, qy, qz};  // Quaternion

        // Simple diagonal covariance
        float pos_var = 0.01f;      // 10cm std deviation
        float ang_var = 0.001f;     // ~1.8 degree std deviation

        //std::vector<float> covariance(21, 0.0f);
        //covariance[0] = pos_var;    // x variance
        //covariance[6] = pos_var;    // y variance
        //covariance[11] = pos_var;   // z variance
        //covariance[15] = ang_var;   // roll variance
        //covariance[18] = ang_var;   // pitch variance
        //covariance[20] = ang_var;   // yaw variance

        float pose_covariance[21] = { 0 };
        pose_covariance[0] = 0.01f;
        pose_covariance[6] = 0.01f;
        pose_covariance[11] = 0.01f;

        float velocity_covariance[21] = { 0 };
        velocity_covariance[0] = 0.01f;
        velocity_covariance[6] = 0.01f;
        velocity_covariance[11] = 0.01f;
        int quality = 100 ? tracking_good : 0;

        if(std::abs(ned_x) < 0.0000001 &&
            std::abs(ned_y) < 0.0000001 &&
            std::abs(ned_z) < 0.0000001)
			tracking_good = false;

        // Console output

        if (!tracking_good)
        {
            pose_covariance[0] = 10.01f;
            pose_covariance[6] = 10.01f;
            pose_covariance[11] = 10.01f;

            velocity_covariance[0] = 10.01f;
            velocity_covariance[6] = 10.01f;
            velocity_covariance[11] = 10.01f;
			updateIMUData(timestamp, Eigen::Vector3f(0.0f, 0.0f, 0.0f), Eigen::Vector3f(0.0f, 0.0f, 0.0f));
            //IncrementResetCounter();
            if (hasLastGoodCoords)
            {
                ned_x = lastGoodPosX;
				ned_y = lastGoodPosY;
                ned_z = lastGoodPosZ;
				std::copy(lastGoodQ, lastGoodQ + 4, q);
            }
            std::cout << "!!! VIO: Tracking lost!" << std::endl;
        }
        else if(isTracking)
        {
            if(!prevIsTracking)
				IncrementResetCounter();
            lastGoodPosX = ned_x;
            lastGoodPosY = ned_y;
			lastGoodPosZ = ned_z;
            std::copy(q, q + 4, lastGoodQ);
			hasLastGoodCoords = true;
        }
        std::cout << "!!! VIO: pos=(" << std::setprecision(3) << ned_x << "," << ned_y << "," << ned_z << ")"
                  << " rpy=(" << ned_roll*180/M_PI << "," << ned_pitch*180/M_PI << "," << ned_yaw*180/M_PI << ")"
				  << " timestamp:"<<timestamp_us
				  << " velocity=("<<ned_vx<<","<<ned_vy<<","<<ned_vz<<")"
				  << " angularVelocity=("<<latest_imu.gyro.x()<<","<<latest_imu.gyro.y()<<","<<latest_imu.gyro.z()<<")"
				  << " Q=("<<qw<<","<<qx<<","<<qy<<","<<qz<<")"
                  << std::endl;


		// Queue odometry data (this is thread-safe)
        mavlink->queueOdometry(
            timestamp_us,
            ned_x, ned_y, ned_z,
            q,
            ned_vx, ned_vy, ned_vz,  // Use actual velocity from ORB-SLAM3
            latest_imu.gyro.x(), latest_imu.gyro.y(), latest_imu.gyro.z(),
            pose_covariance,
            velocity_covariance,
            reset_counter,          // reset_counter
            MAV_ESTIMATOR_TYPE_VIO, // estimator_type
            quality                 // quality
        );
    }
};

int main(int argc, char **argv) {
    if(argc < 4 || argc > 5) {
        std::cerr << "Usage: ./stereo_inertial_realsense path_to_vocabulary path_to_settings output_file [mode]" << std::endl;
        std::cerr << "  mode: 0 = ODOMETRY (default)" << std::endl;
        std::cerr << "        1 = VISION_POSITION_ESTIMATE" << std::endl;
        std::cerr << "        2 = VISION_POSITION_ESTIMATE + VISION_SPEED_ESTIMATE" << std::endl;
        return 1;
    }

    // Parse MAVLink mode
    MAVLinkMode mavlink_mode = MAVLinkMode::ODOMETRY;
    std::cout << "[DEBUG main] argc = " << argc << std::endl;
    if (argc == 5) {
        int mode_val = std::atoi(argv[4]);
        std::cout << "[DEBUG main] Mode argument (argv[4]) = " << argv[4] << ", parsed as " << mode_val << std::endl;
        if (mode_val == 1) {
            mavlink_mode = MAVLinkMode::VISION_POSITION_ESTIMATE;
            std::cout << "[DEBUG main] Set mavlink_mode to VISION_POSITION_ESTIMATE (1)" << std::endl;
        } else if (mode_val == 2) {
            mavlink_mode = MAVLinkMode::VISION_POSITION_AND_SPEED;
            std::cout << "[DEBUG main] Set mavlink_mode to VISION_POSITION_AND_SPEED (2)" << std::endl;
        } else if (mode_val != 0) {
            std::cerr << "Invalid mode: " << mode_val << ". Using ODOMETRY mode." << std::endl;
        } else {
            std::cout << "[DEBUG main] Using default ODOMETRY mode (0)" << std::endl;
        }
    } else {
        std::cout << "[DEBUG main] No mode argument provided, using default ODOMETRY mode (0)" << std::endl;
    }
    std::cout << "[DEBUG main] Final mavlink_mode = " << static_cast<int>(mavlink_mode) << std::endl;

    // Create SLAM system
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::IMU_STEREO, false);

    // Create VIO logger with specified mode
    VIOLogger vio_logger(argv[3], mavlink_mode);

    // Configure RealSense
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

    double offset = 0; // Timestamp offset in ms

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

    rs2::pipeline_profile profile = pipe.start(cfg, imu_callback);

    std::vector<ORB_SLAM3::IMU::Point> vImuMeas;
    int frame_count = 0;

    std::cout << "Starting VIO tracking and logging..." << std::endl;
    std::cout << "Move the camera to initialize the system." << std::endl;

	if(!vio_logger.WaitMavlink(30))
	{
		std::cout << "MavLink Connection failed." << std::endl;
		return 1;
	}
	else
	{
		std::cout << "MavLink Connected!" << std::endl;
	}

    // Clear IMU vectors
    v_gyro_data.clear();
    v_gyro_timestamp.clear();
    v_accel_data_sync.clear();
    v_accel_timestamp_sync.clear();

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
                std::cout << count_im_buffer - 1 << " dropped frs\n";
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

            // Copy the IMU data
            vGyro = v_gyro_data;
            vGyro_times = v_gyro_timestamp;
            vAccel = v_accel_data_sync;
            timestamp = timestamp_image;
            left = imCV.clone();
            right = imRightCV.clone();

            // Clear IMU vectors
            v_gyro_data.clear();
            v_gyro_timestamp.clear();
            v_accel_data_sync.clear();
            v_accel_timestamp_sync.clear();

            image_ready = false;
        }

        frame_count++;

        // Build IMU measurements from synchronized data
        for(size_t i = 0; i < vGyro.size(); ++i)
        {
            ORB_SLAM3::IMU::Point imu_point(vAccel[i].x, vAccel[i].y, vAccel[i].z,
                                          vGyro[i].x, vGyro[i].y, vGyro[i].z,
                                          vGyro_times[i]);
            vImuMeas.push_back(imu_point);
        }

        // Store latest IMU data for logging
        if(!vGyro.empty()) {
            latest_gyro = Eigen::Vector3f(vGyro.back().x, vGyro.back().y, vGyro.back().z);
            latest_accel = Eigen::Vector3f(vAccel.back().x, vAccel.back().y, vAccel.back().z);
            latest_imu_timestamp = vGyro_times.back();
            vio_logger.updateIMUData(latest_imu_timestamp, latest_accel, latest_gyro);
        }

        // Track with stereo-inertial
        Sophus::SE3f Tcw = SLAM.TrackStereo(left, right, timestamp, vImuMeas);

        // Get velocity from ORB-SLAM3
        Eigen::Vector3f velocity = SLAM.GetVelocity();

        // Check tracking status
        auto tracking_state = SLAM.GetTrackingState();
        bool tracking_good = (tracking_state == ORB_SLAM3::Tracking::OK);

        // Display tracking status
        switch(tracking_state) {
            case ORB_SLAM3::Tracking::SYSTEM_NOT_READY:
                std::cout << "+++ System not ready" << std::endl;
                vio_logger.SetTrackingState(false);
                break;
            case ORB_SLAM3::Tracking::NO_IMAGES_YET:
                std::cout << "+++ No images yet" << std::endl;
                vio_logger.SetTrackingState(false);
                break;
            case ORB_SLAM3::Tracking::NOT_INITIALIZED:
                std::cout << "+++ Not initialized - move camera with rotation!" << std::endl;
                vio_logger.SetTrackingState(false);
                break;
            case ORB_SLAM3::Tracking::OK:
                std::cout << "+++ Tracking OK - logging VIO data" << std::endl;
                vio_logger.SetTrackingState(true);
                break;
            case ORB_SLAM3::Tracking::RECENTLY_LOST:
                std::cout << "+++ Recently lost tracking" << std::endl;
                vio_logger.SetTrackingState(false);
                break;
            case ORB_SLAM3::Tracking::LOST:
                std::cout << "+++ Lost tracking" << std::endl;
                vio_logger.SetTrackingState(false);
                break;
            case ORB_SLAM3::Tracking::OK_KLT:
                std::cout << "+++ Tracking OK using KLT optical flow" << std::endl;
                vio_logger.SetTrackingState(true);
                break;
        }

        // Log VIO data
        vio_logger.logPose(timestamp, Tcw, velocity, tracking_good);

        // Clear IMU measurements for next frame
        vImuMeas.clear();
    }

    std::cout << "Shutting down..." << std::endl;
    SLAM.Shutdown();
    return 0;
}
