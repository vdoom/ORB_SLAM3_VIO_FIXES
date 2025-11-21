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
#include <ctime>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <atomic>
#include <memory>
// Include MAVLink headers
#include <common/mavlink.h>

// Earth radius in meters (WGS84)
const double EARTH_RADIUS = 6378137.0;

// GPS coordinate conversion functions
struct GPSCoord {
    double lat;  // degrees
    double lon;  // degrees
    double alt;  // meters (MSL)
};

// Convert NED offset (meters) to GPS coordinates relative to origin
GPSCoord ned_to_gps(double north, double east, double down, const GPSCoord& origin) {
    GPSCoord result;

    // Convert north/east offsets to lat/lon
    // dLat = north / R_earth (in radians)
    // dLon = east / (R_earth * cos(lat))

    double lat_rad = origin.lat * M_PI / 180.0;

    double dLat = north / EARTH_RADIUS;
    double dLon = east / (EARTH_RADIUS * cos(lat_rad));

    result.lat = origin.lat + (dLat * 180.0 / M_PI);
    result.lon = origin.lon + (dLon * 180.0 / M_PI);
    result.alt = origin.alt - down;  // NED down is negative of altitude

    return result;
}

// Structure to hold GPS data
struct GPSData {
    uint64_t time_usec;
    int32_t lat;           // Latitude * 1e7 (degE7)
    int32_t lon;           // Longitude * 1e7 (degE7)
    int32_t alt;           // Altitude in mm (MSL)
    float vn, ve, vd;      // Velocity NED (m/s)
    uint16_t cog;          // Course over ground (cdeg)
    uint8_t fix_type;      // GPS fix type
    uint8_t satellites_visible;
    uint16_t eph;          // GPS HDOP * 100
    uint16_t epv;          // GPS VDOP * 100
    uint16_t vel;          // Ground speed (cm/s)
    int16_t vz;            // Vertical velocity (cm/s)
    uint8_t reset_counter;
};

// Thread-safe queue for GPS data
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

public:
    MAVLinkGPSInterface(const char* serial_port, int baud_rate = 57600)
        : system_id(255), component_id(197), running(false), connected(false) {

        std::cout << "[MAVLinkGPSInterface] Initializing GPS interface" << std::endl;

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

        std::cout << "MAVLink GPS interface initialized" << std::endl;
        std::cout << "Serial port: " << serial_port << std::endl;
        std::cout << "Baud rate: " << baud_rate << std::endl;
    }

    ~MAVLinkGPSInterface() {
        stop();
        close(serial_fd);
    }

    void start() {
        if (!running) {
            running = true;
            mavlink_thread = std::thread(&MAVLinkGPSInterface::run, this);
            std::cout << "MAVLink GPS thread started" << std::endl;
        }
    }

    void stop() {
        if (running) {
            running = false;
            if (mavlink_thread.joinable()) {
                mavlink_thread.join();
            }
            std::cout << "MAVLink GPS thread stopped" << std::endl;
        }
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
    void run() {
        requestDataStream();

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

            usleep(10000);  // 10ms
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

        std::cout << "Requested data stream" << std::endl;
    }

    void sendGPSInput(const GPSData& data) {
        mavlink_message_t msg;
        uint8_t buf[MAVLINK_MAX_PACKET_LEN];

        mavlink_msg_gps_input_pack(
            system_id,
            component_id,
            &msg,
            data.time_usec,
            0,  // GPS ID
            0,  // ignore_flags (use all fields)
            data.time_usec / 1000,  // time_week_ms
            0,  // time_week
            data.fix_type,
            data.lat,
            data.lon,
            data.alt / 1000.0f,  // altitude in meters
            data.eph / 100.0f,   // HDOP
            data.epv / 100.0f,   // VDOP
            data.vel / 100.0f,   // ground speed (m/s)
            data.vn,
            data.ve,
            data.vd,
            0.5f,  // speed accuracy
            0.5f,  // horiz accuracy
            0.5f,  // vert accuracy
            data.satellites_visible
        );

        uint16_t len = mavlink_msg_to_send_buffer(buf, &msg);
        ssize_t sent = write(serial_fd, buf, len);

        if (sent < 0) {
            std::cerr << "Failed to send GPS input: " << strerror(errno) << std::endl;
        } else {
            std::cout << "Sent GPS | Lat: " << data.lat/1e7 << " Lon: " << data.lon/1e7
                     << " Alt: " << data.alt/1000.0 << "m | Vel: "
                     << data.vn << "," << data.ve << "," << data.vd
                     << " | Sats: " << (int)data.satellites_visible
                     << " | Q-size: " << getQueueSize() << std::endl;
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

                        std::cout << "MAVLink connection established with system "
                                 << (int)msg.sysid << std::endl;
                    }
                }
                break;
            }
        }
    }
};

class VIOGPSLogger {
private:
    uint64_t start_time_us;
    uint8_t  reset_counter;

    bool isTracking;
    bool prevIsTracking;

    // GPS origin (set this to your actual location)
    GPSCoord gps_origin;

    // Last known good GPS position
    bool hasLastGoodGPS;
    GPSCoord lastGoodGPS;

    // Position offset tracking for resets
    float position_offset_x;
    float position_offset_y;
    float position_offset_z;

    std::shared_ptr<MAVLinkGPSInterface> mavlink;

    struct IMUData {
        double timestamp;
        Eigen::Vector3f accel;
        Eigen::Vector3f gyro;
        bool valid;

        IMUData() : timestamp(0), accel(0,0,0), gyro(0,0,0), valid(false){}
    } latest_imu;

public:
    VIOGPSLogger(double origin_lat, double origin_lon, double origin_alt)
        : reset_counter(0)
        , isTracking(false)
        , prevIsTracking(false)
        , hasLastGoodGPS(false)
        , position_offset_x(0.0f)
        , position_offset_y(0.0f)
        , position_offset_z(0.0f)
    {
        // Set GPS origin
        gps_origin.lat = origin_lat;
        gps_origin.lon = origin_lon;
        gps_origin.alt = origin_alt;

        lastGoodGPS = gps_origin;  // Initialize to origin

        // Get system start time
        auto now = std::chrono::high_resolution_clock::now();
        start_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
            now.time_since_epoch()).count();

        std::cout << "VIO GPS logging started." << std::endl;
        std::cout << "GPS Origin: Lat=" << std::fixed << std::setprecision(7)
                  << origin_lat << " Lon=" << origin_lon << " Alt=" << origin_alt << "m" << std::endl;

        const char* serial_port = "/dev/ttyTHS1";
        int baud_rate = 1500000;

        std::cout << "Starting MAVLink UART communication for GPS" << std::endl;
        mavlink = std::make_shared<MAVLinkGPSInterface>(serial_port, baud_rate);
        mavlink->start();
    }

    ~VIOGPSLogger() {
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
        std::cout << "@@@ Increment reset counter - position reset detected" << std::endl;
        prevIsTracking = isTracking;
        return ++reset_counter;
    }

    bool WaitMavlink(int timeout_seconds = 10)
    {
        if (!mavlink->waitForConnection(timeout_seconds)) {
            std::cerr << "Failed to connect to ArduPilot!" << std::endl;
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

        uint64_t timestamp_us = start_time_us + static_cast<uint64_t>(timestamp * 1e6);

        // Get camera pose
        Sophus::SE3f Twc = Tcw.inverse();
        Eigen::Vector3f position = Twc.translation();

        // Convert to NED frame
        float ned_x = position.z();
        float ned_y = -position.x();
        float ned_z = position.y();

        // Apply position offset (for handling resets)
        ned_x += position_offset_x;
        ned_y += position_offset_y;
        ned_z += position_offset_z;

        // Transform velocity to NED
        float ned_vx = velocity.z();
        float ned_vy = -velocity.x();
        float ned_vz = velocity.y();

        if(std::abs(ned_x) < 0.0000001 &&
            std::abs(ned_y) < 0.0000001 &&
            std::abs(ned_z) < 0.0000001)
            tracking_good = false;

        GPSData gps_data;
        gps_data.time_usec = timestamp_us;
        gps_data.reset_counter = reset_counter;

        if (!tracking_good || !isTracking)
        {
            // Use last known good GPS position
            if (hasLastGoodGPS) {
                gps_data.lat = static_cast<int32_t>(lastGoodGPS.lat * 1e7);
                gps_data.lon = static_cast<int32_t>(lastGoodGPS.lon * 1e7);
                gps_data.alt = static_cast<int32_t>(lastGoodGPS.alt * 1000);
            } else {
                // Use origin if no good position yet
                gps_data.lat = static_cast<int32_t>(gps_origin.lat * 1e7);
                gps_data.lon = static_cast<int32_t>(gps_origin.lon * 1e7);
                gps_data.alt = static_cast<int32_t>(gps_origin.alt * 1000);
            }

            // Zero velocity when not tracking
            gps_data.vn = 0.0f;
            gps_data.ve = 0.0f;
            gps_data.vd = 0.0f;
            gps_data.vel = 0;
            gps_data.vz = 0;

            gps_data.fix_type = 0;  // No fix when tracking lost
            gps_data.satellites_visible = 0;
            gps_data.eph = 9999;  // High error
            gps_data.epv = 9999;

            std::cout << "!!! GPS: Tracking lost - using last known position" << std::endl;
        }
        else
        {
            // Convert NED position to GPS coordinates
            GPSCoord current_gps = ned_to_gps(ned_x, ned_y, ned_z, gps_origin);

            // Check for position reset (large jump in position)
            if(hasLastGoodGPS) {
                // If there's a large discontinuity, ORB-SLAM3 likely reset
                // We should continue from last known GPS position
                double lat_diff = std::abs(current_gps.lat - lastGoodGPS.lat);
                double lon_diff = std::abs(current_gps.lon - lastGoodGPS.lon);

                // If jump is larger than 10 meters, consider it a reset
                if(lat_diff > 0.0001 || lon_diff > 0.0001) {
                    std::cout << "@@@ Large position jump detected - adjusting offset" << std::endl;

                    // Calculate the offset needed to make current position continuous
                    GPSCoord offset_gps = ned_to_gps(ned_x - position_offset_x,
                                                      ned_y - position_offset_y,
                                                      ned_z - position_offset_z,
                                                      gps_origin);

                    // Add offset to maintain continuity
                    double delta_lat = lastGoodGPS.lat - offset_gps.lat;
                    double delta_lon = lastGoodGPS.lon - offset_gps.lon;

                    // Convert back to NED offset
                    double lat_rad = gps_origin.lat * M_PI / 180.0;
                    position_offset_x += delta_lat * EARTH_RADIUS * M_PI / 180.0;
                    position_offset_y += delta_lon * EARTH_RADIUS * cos(lat_rad) * M_PI / 180.0;

                    // Recompute GPS with new offset
                    ned_x += position_offset_x;
                    ned_y += position_offset_y;
                    current_gps = ned_to_gps(ned_x, ned_y, ned_z, gps_origin);

                    IncrementResetCounter();
                }
            }

            gps_data.lat = static_cast<int32_t>(current_gps.lat * 1e7);
            gps_data.lon = static_cast<int32_t>(current_gps.lon * 1e7);
            gps_data.alt = static_cast<int32_t>(current_gps.alt * 1000);  // mm

            gps_data.vn = ned_vx;
            gps_data.ve = ned_vy;
            gps_data.vd = ned_vz;

            // Calculate ground speed and vertical speed
            float ground_speed = std::sqrt(ned_vx*ned_vx + ned_vy*ned_vy);
            gps_data.vel = static_cast<uint16_t>(ground_speed * 100);  // cm/s
            gps_data.vz = static_cast<int16_t>(ned_vz * 100);  // cm/s

            // Calculate course over ground
            gps_data.cog = static_cast<uint16_t>(std::atan2(ned_vy, ned_vx) * 180.0 / M_PI * 100);

            gps_data.fix_type = 3;  // 3D fix
            gps_data.satellites_visible = 12;  // Simulated good satellite count
            gps_data.eph = 50;   // 0.5m horizontal accuracy
            gps_data.epv = 50;   // 0.5m vertical accuracy

            // Save as last known good position
            lastGoodGPS = current_gps;
            hasLastGoodGPS = true;

            if(!prevIsTracking)
                IncrementResetCounter();
        }

        std::cout << "!!! GPS: Lat=" << std::fixed << std::setprecision(7)
                  << gps_data.lat/1e7 << " Lon=" << gps_data.lon/1e7
                  << " Alt=" << std::setprecision(2) << gps_data.alt/1000.0 << "m"
                  << " | Vel: N=" << ned_vx << " E=" << ned_vy << " D=" << ned_vz
                  << " | Fix: " << (int)gps_data.fix_type
                  << " | Sats: " << (int)gps_data.satellites_visible
                  << std::endl;

        mavlink->queueGPS(gps_data);
    }
};

int main(int argc, char **argv) {
    if(argc < 6 || argc > 6) {
        std::cerr << "Usage: ./stereo_inertial_realsense_gps path_to_vocabulary path_to_settings origin_lat origin_lon origin_alt" << std::endl;
        std::cerr << "  origin_lat: GPS latitude of origin point (degrees)" << std::endl;
        std::cerr << "  origin_lon: GPS longitude of origin point (degrees)" << std::endl;
        std::cerr << "  origin_alt: GPS altitude of origin point (meters MSL)" << std::endl;
        std::cerr << "Example: ./stereo_inertial_realsense_gps ORBvoc.txt settings.yaml 37.7749 -122.4194 10.0" << std::endl;
        return 1;
    }

    // Parse GPS origin
    double origin_lat = std::atof(argv[3]);
    double origin_lon = std::atof(argv[4]);
    double origin_alt = std::atof(argv[5]);

    std::cout << "GPS Origin: Lat=" << origin_lat << " Lon=" << origin_lon << " Alt=" << origin_alt << std::endl;

    // Create SLAM system
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::IMU_STEREO, false);

    // Create VIO GPS logger
    VIOGPSLogger vio_logger(origin_lat, origin_lon, origin_alt);

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

    double offset = 0;

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

    std::cout << "Starting VIO GPS tracking..." << std::endl;
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

        if(!vGyro.empty()) {
            latest_gyro = Eigen::Vector3f(vGyro.back().x, vGyro.back().y, vGyro.back().z);
            latest_accel = Eigen::Vector3f(vAccel.back().x, vAccel.back().y, vAccel.back().z);
            latest_imu_timestamp = vGyro_times.back();
            vio_logger.updateIMUData(latest_imu_timestamp, latest_accel, latest_gyro);
        }

        // Track
        Sophus::SE3f Tcw = SLAM.TrackStereo(left, right, timestamp, vImuMeas);
        Eigen::Vector3f velocity = SLAM.GetVelocity();

        auto tracking_state = SLAM.GetTrackingState();
        bool tracking_good = (tracking_state == ORB_SLAM3::Tracking::OK ||
                             tracking_state == ORB_SLAM3::Tracking::OK_KLT);

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
                std::cout << "+++ Tracking OK" << std::endl;
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
                std::cout << "+++ Tracking OK (KLT)" << std::endl;
                vio_logger.SetTrackingState(true);
                break;
        }

        vio_logger.logPose(timestamp, Tcw, velocity, tracking_good);
        vImuMeas.clear();
    }

    std::cout << "Shutting down..." << std::endl;
    SLAM.Shutdown();
    return 0;
}
