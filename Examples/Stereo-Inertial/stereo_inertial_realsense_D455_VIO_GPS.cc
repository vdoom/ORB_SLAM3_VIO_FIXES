#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include <iomanip>
#include <Eigen/Dense>
#include "System.h"

//==========================================================
#include <cstring>
#include <cmath>
#include <cstdint>
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

// GPS coordinate conversion constants
const double EARTH_RADIUS = 6371000.0;  // Earth radius in meters

// Structure to hold GPS data
struct GPSData {
    uint64_t time_usec;
    int32_t lat;                      // Latitude in 1E7 degrees
    int32_t lon;                      // Longitude in 1E7 degrees
    int32_t alt;                      // Altitude in mm (MSL)
    float vn, ve, vd;                 // Velocity NED (m/s)
    float vx, vy, vz;                 // Velocity body frame (m/s)
    float speed_accuracy;             // Speed accuracy (m/s)
    float horiz_accuracy;             // Horizontal accuracy (m)
    float vert_accuracy;              // Vertical accuracy (m)
    uint8_t fix_type;                 // GPS fix type
    uint8_t satellites_visible;       // Number of satellites
    float yaw;                        // Yaw in radians
};

// Helper functions for GPS coordinate conversion
inline double degToRad(double deg) {
    return deg * M_PI / 180.0;
}

inline double radToDeg(double rad) {
    return rad * 180.0 / M_PI;
}

// Convert NED offset (meters) to GPS coordinates
inline void nedToGPS(double lat_base, double lon_base, double alt_base,
                     double north, double east, double down,
                     int32_t& lat_out, int32_t& lon_out, int32_t& alt_out) {
    // Convert NED offsets to GPS coordinates
    double lat_offset_deg = radToDeg(north / EARTH_RADIUS);
    double lon_offset_deg = radToDeg(east / (EARTH_RADIUS * cos(degToRad(lat_base))));

    double lat_new = lat_base + lat_offset_deg;
    double lon_new = lon_base + lon_offset_deg;
    double alt_new = alt_base - down;  // NED down is opposite of altitude

    // Convert to MAVLink format (1E7 for lat/lon, mm for altitude)
    lat_out = static_cast<int32_t>(lat_new * 1e7);
    lon_out = static_cast<int32_t>(lon_new * 1e7);
    alt_out = static_cast<int32_t>(alt_new * 1000.0);  // meters to mm
}

// Thread-safe queue for GPS data
class GPSQueue {
private:
    std::queue<GPSData> queue;
    std::mutex mutex;
    std::condition_variable cv;
    const size_t max_size = 100;  // Prevent queue from growing too large

public:
    void push(const GPSData& data) {
        std::unique_lock<std::mutex> lock(mutex);

        // If queue is full, remove oldest element
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

class MAVLinkInterface {
private:
    int serial_fd;
    uint8_t system_id;      // Read from ArduPilot heartbeat (starts as 255, then adopts ArduPilot's ID)
    uint8_t component_id;   // Set to MAV_COMP_ID_GPS (220) upon connection

    std::atomic<bool> running;
    std::thread mavlink_thread;
    GPSQueue gps_queue;
    std::atomic<bool> connected;
    std::mutex connected_mutex;
    std::condition_variable connected_cv;
    
public:
    MAVLinkInterface(const char* serial_port, int baud_rate = 57600)
        : system_id(255), component_id(197), running(false), connected(false) {
        
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
    
    // Queue GPS data to be sent (called from main thread)
    void queueGPS(uint64_t time_usec,
                  int32_t lat, int32_t lon, int32_t alt,
                  float vn, float ve, float vd,
                  float vx, float vy, float vz,
                  float horiz_accuracy, float vert_accuracy, float speed_accuracy,
                  uint8_t fix_type, uint8_t satellites_visible,
                  float yaw) {
        GPSData data;
        data.time_usec = time_usec;
        data.lat = lat;
        data.lon = lon;
        data.alt = alt;
        data.vn = vn;
        data.ve = ve;
        data.vd = vd;
        data.vx = vx;
        data.vy = vy;
        data.vz = vz;
        data.horiz_accuracy = horiz_accuracy;
        data.vert_accuracy = vert_accuracy;
        data.speed_accuracy = speed_accuracy;
        data.fix_type = fix_type;
        data.satellites_visible = satellites_visible;
        data.yaw = yaw;

        gps_queue.push(data);
    }

    // Get queue size (for monitoring)
    size_t getQueueSize() const {
        return const_cast<GPSQueue&>(gps_queue).size();
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

            // Check for GPS data in queue and send
            GPSData gps_data;
            if (gps_queue.pop(gps_data, 10)) {  // 10ms timeout
                sendGPS(gps_data);
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
    
    // Send GPS data from queued struct
    void sendGPS(const GPSData& data) {
        mavlink_message_t msg;
        uint8_t buf[MAVLINK_MAX_PACKET_LEN];

        // Use GPS_INPUT message (ID 232) for simulated/external GPS
        mavlink_msg_gps_input_pack(
            system_id,
            component_id,
            &msg,
            data.time_usec,                    // Timestamp (micros since boot or Unix epoch)
            0,                                  // GPS ID (0 for first GPS)
            (uint16_t)0,                       // Ignore flags (0 = use all fields)
            (uint32_t)0,                       // Time week MS
            (uint16_t)0,                       // Time week
            data.fix_type,                     // 0-1: no fix, 2: 2D fix, 3: 3D fix
            data.lat,                          // Latitude (WGS84, degE7)
            data.lon,                          // Longitude (WGS84, degE7)
            static_cast<float>(data.alt) / 1000.0f,  // Altitude (MSL, meters)
            data.horiz_accuracy,               // HDOP (horizontal dilution of precision)
            data.vert_accuracy,                // VDOP (vertical dilution of precision)
            data.vn,                           // Velocity North (m/s)
            data.ve,                           // Velocity East (m/s)
            data.vd,                           // Velocity Down (m/s)
            data.speed_accuracy,               // Speed accuracy (m/s)
            data.horiz_accuracy,               // Horizontal accuracy (meters)
            data.vert_accuracy,                // Vertical accuracy (meters)
            data.satellites_visible,           // Number of satellites visible
            static_cast<uint16_t>(data.yaw * 57.2958f * 100.0f)  // Yaw (centidegrees, 0-36000)
        );

        uint16_t len = mavlink_msg_to_send_buffer(buf, &msg);
        ssize_t sent = write(serial_fd, buf, len);

        if (sent < 0) {
            std::cerr << "Failed to send GPS: " << strerror(errno) << std::endl;
        } else {
            std::cout << "Sent GPS | Lat: " << data.lat / 1e7
                     << " Lon: " << data.lon / 1e7
                     << " Alt: " << data.alt / 1000.0f
                     << " | Vel NED: (" << data.vn << ", " << data.ve << ", " << data.vd
                     << ") | Fix: " << (int)data.fix_type
                     << " | Sats: " << (int)data.satellites_visible
                     << " | Q-size: " << getQueueSize() << std::endl;
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
                        component_id = 220;  // MAV_COMP_ID_GPS
                        
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

    // GPS home coordinates (Tallinn, Estonia)
    const double GPS_HOME_LAT = 59.4370;
    const double GPS_HOME_LON = 24.7536;
    const double GPS_HOME_ALT = 0.0;  // MSL in meters

    // GPS base position (updates on SLAM reset to last known position)
    double gps_base_lat;
    double gps_base_lon;
    double gps_base_alt;

    // NED offset at the base (used when SLAM resets)
    float base_ned_x;
    float base_ned_y;
    float base_ned_z;

    bool hasLastGoodCoords;
    float lastGoodPosX;
    float lastGoodPosY;
    float lastGoodPosZ;
    float lastGoodYaw;

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
    VIOLogger(const std::string& filename)
        : reset_counter(0)
        , isTracking(false)
        , prevIsTracking(false)
        , gps_base_lat(GPS_HOME_LAT)
        , gps_base_lon(GPS_HOME_LON)
        , gps_base_alt(GPS_HOME_ALT)
        , base_ned_x(0.0f)
        , base_ned_y(0.0f)
        , base_ned_z(0.0f)
        , hasLastGoodCoords(false)
        , lastGoodPosX(0.0f)
        , lastGoodPosY(0.0f)
        , lastGoodPosZ(0.0f)
        , lastGoodYaw(0.0f)
    {

        // Get system start time
        auto now = std::chrono::high_resolution_clock::now();
        start_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
            now.time_since_epoch()).count();

        std::cout << "GPS-VIO logging started. Output file: " << filename << std::endl;
        std::cout << "GPS Home Position: Lat=" << GPS_HOME_LAT
                  << " Lon=" << GPS_HOME_LON
                  << " Alt=" << GPS_HOME_ALT << "m" << std::endl;

        //=========================================================================
        // MavLink
        const char* serial_port = "/dev/ttyTHS0";
        int baud_rate = 1500000;

        std::cout << "Starting MAVLink UART communication with GPS data" << std::endl;
        mavlink = std::make_shared<MAVLinkInterface>(serial_port, baud_rate);
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

    void logPose(double timestamp, const Sophus::SE3f& Tcw, bool tracking_good) {

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
                ned_yaw = lastGoodYaw;
            }
            std::cout << "!!! GPS-VIO: Tracking lost!" << std::endl;
        }
        else if(isTracking)
        {
            if(!prevIsTracking) {
                IncrementResetCounter();
                // On reset, update GPS base to last known position
                if (hasLastGoodCoords) {
                    // Calculate GPS position from last known NED
                    int32_t last_lat, last_lon, last_alt;
                    nedToGPS(gps_base_lat, gps_base_lon, gps_base_alt,
                            lastGoodPosX - base_ned_x,
                            lastGoodPosY - base_ned_y,
                            lastGoodPosZ - base_ned_z,
                            last_lat, last_lon, last_alt);

                    // Update base to this position
                    gps_base_lat = last_lat / 1e7;
                    gps_base_lon = last_lon / 1e7;
                    gps_base_alt = last_alt / 1000.0;

                    // Reset NED base to current position
                    base_ned_x = ned_x;
                    base_ned_y = ned_y;
                    base_ned_z = ned_z;

                    std::cout << "!!! GPS-VIO: SLAM reset detected! New GPS base: Lat="
                             << gps_base_lat << " Lon=" << gps_base_lon
                             << " Alt=" << gps_base_alt << "m" << std::endl;
                }
            }
            lastGoodPosX = ned_x;
            lastGoodPosY = ned_y;
            lastGoodPosZ = ned_z;
            lastGoodYaw = ned_yaw;
            hasLastGoodCoords = true;
        }
        // Convert NED position to GPS coordinates
        int32_t gps_lat, gps_lon, gps_alt;
        nedToGPS(gps_base_lat, gps_base_lon, gps_base_alt,
                 ned_x - base_ned_x,
                 ned_y - base_ned_y,
                 ned_z - base_ned_z,
                 gps_lat, gps_lon, gps_alt);

        // GPS fix type based on tracking quality
        uint8_t fix_type = tracking_good ? 3 : 1;  // 3 = 3D fix, 1 = no fix
        uint8_t satellites_visible = tracking_good ? 12 : 0;

        // Accuracy estimates
        float horiz_accuracy = tracking_good ? 0.5f : 10.0f;  // meters
        float vert_accuracy = tracking_good ? 1.0f : 10.0f;   // meters
        float speed_accuracy = 0.1f;  // m/s

        // Velocity in NED frame (using IMU acceleration as approximation)
        float vn = latest_imu.accel.x();
        float ve = latest_imu.accel.y();
        float vd = latest_imu.accel.z();

        std::cout << "!!! GPS-VIO: pos=(" << std::setprecision(3) << ned_x << "," << ned_y << "," << ned_z << ")"
                  << " GPS=(" << gps_lat/1e7 << "," << gps_lon/1e7 << "," << gps_alt/1000.0f << ")"
                  << " rpy=(" << ned_roll*180/M_PI << "," << ned_pitch*180/M_PI << "," << ned_yaw*180/M_PI << ")"
                  << " timestamp:" << timestamp_us
                  << " velocity=(" << vn << "," << ve << "," << vd << ")"
                  << " fix=" << (int)fix_type << " sats=" << (int)satellites_visible
                  << std::endl;

        // Queue GPS data (this is thread-safe)
        mavlink->queueGPS(
            timestamp_us,
            gps_lat, gps_lon, gps_alt,
            vn, ve, vd,                    // Velocity NED
            latest_imu.accel.x(), latest_imu.accel.y(), latest_imu.accel.z(),  // Velocity body frame
            horiz_accuracy, vert_accuracy, speed_accuracy,
            fix_type,
            satellites_visible,
            ned_yaw                        // Yaw in radians
        );
    }
};

int main(int argc, char **argv) {
    if(argc != 4) {
        std::cerr << "Usage: ./stereo_inertial_realsense path_to_vocabulary path_to_settings output_file" << std::endl;
        return 1;
    }

    // Create SLAM system
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::IMU_STEREO, false);

    // Create VIO logger
    VIOLogger vio_logger(argv[3]);

    // Configure RealSense
    rs2::pipeline pipe;
    rs2::config cfg;

    cfg.enable_stream(RS2_STREAM_INFRARED, 1, 640, 480, RS2_FORMAT_Y8, 30);
    cfg.enable_stream(RS2_STREAM_INFRARED, 2, 640, 480, RS2_FORMAT_Y8, 30);
    cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F, 250);
    cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F, 400);

    rs2::pipeline_profile profile = pipe.start(cfg);

    std::vector<ORB_SLAM3::IMU::Point> vImuMeas;
    int frame_count = 0;
	
	// Variables to store latest IMU data for logging
    Eigen::Vector3f latest_accel(0, 0, 0);
    Eigen::Vector3f latest_gyro(0, 0, 0);
    double latest_imu_timestamp = 0;

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

    while(true) {
        rs2::frameset frames = pipe.wait_for_frames();
        frame_count++;

        // Collect IMU data (using your working approach)
        auto motion_frames = frames.first_or_default(RS2_STREAM_ACCEL);
        auto gyro_frames = frames.first_or_default(RS2_STREAM_GYRO);

        if (motion_frames && gyro_frames) {
            auto accel_data = motion_frames.as<rs2::motion_frame>().get_motion_data();
            auto gyro_data = gyro_frames.as<rs2::motion_frame>().get_motion_data();

            double imu_timestamp = motion_frames.get_timestamp() * 1e-3;
			
			// Store latest IMU data for logging
            latest_accel = Eigen::Vector3f(accel_data.x, accel_data.y, accel_data.z);
            latest_gyro = Eigen::Vector3f(gyro_data.x, gyro_data.y, gyro_data.z);
            latest_imu_timestamp = imu_timestamp;
			
			vio_logger.updateIMUData(imu_timestamp, latest_accel, latest_gyro);
			
            // Create IMU measurement
            ORB_SLAM3::IMU::Point imu_point(accel_data.x, accel_data.y, accel_data.z,
                                          gyro_data.x, gyro_data.y, gyro_data.z, imu_timestamp);
            vImuMeas.push_back(imu_point);

            // Debug: print IMU data occasionally
            if (frame_count % 100 == 0) {
                std::cout << "Frame " << frame_count << ": IMU measurements = " << vImuMeas.size() << std::endl;
                std::cout << "  Latest IMU: accel=(" << accel_data.x << "," << accel_data.y << "," << accel_data.z
                          << ") gyro=(" << gyro_data.x << "," << gyro_data.y << "," << gyro_data.z << ")" << std::endl;
            }
        } else {
            if (frame_count % 100 == 0) {
                std::cout << "Frame " << frame_count << ": No IMU data!" << std::endl;
            }
        }

        // Get stereo images
        rs2::video_frame ir_frame_left = frames.get_infrared_frame(1);
        rs2::video_frame ir_frame_right = frames.get_infrared_frame(2);

        if (!ir_frame_left || !ir_frame_right) continue;

        cv::Mat left(cv::Size(640, 480), CV_8UC1, (void*)ir_frame_left.get_data());
        cv::Mat right(cv::Size(640, 480), CV_8UC1, (void*)ir_frame_right.get_data());

        double timestamp = ir_frame_left.get_timestamp() * 1e-3;

        // Track with stereo-inertial (using your working approach)
        Sophus::SE3f Tcw = SLAM.TrackStereo(left, right, timestamp, vImuMeas);

        // Check tracking status
        auto tracking_state = SLAM.GetTrackingState();
        bool tracking_good = (tracking_state == ORB_SLAM3::Tracking::OK);

        // Display tracking status
        //if (frame_count % 30 == 0) {  // Every 30 frames
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
        //}

                // Log VIO data for Pixhawk
        //if (tracking_good) {
            vio_logger.logPose(timestamp, Tcw, tracking_good);
        //}
        vImuMeas.clear(); // Clear after use (like your working code)
    }

    std::cout << "Shutting down..." << std::endl;
    SLAM.Shutdown();
    return 0;
}