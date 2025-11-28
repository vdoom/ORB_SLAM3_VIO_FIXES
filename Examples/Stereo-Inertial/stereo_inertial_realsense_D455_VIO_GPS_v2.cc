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
#include <common/mavlink.h>

const double EARTH_RADIUS = 6378137.0;

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

    // GPS origin tracking
    std::atomic<bool> gps_origin_received;
    GPSCoord gps_origin;
    std::mutex gps_origin_mutex;
    std::condition_variable gps_origin_cv;

    std::chrono::steady_clock::time_point last_waiting_msg_time;

public:
    MAVLinkGPSInterface(const char* serial_port, int baud_rate = 57600)
        : system_id(255), component_id(197), running(false), connected(false), gps_origin_received(false) {

        std::cout << "[MAVLinkGPSInterface] Initializing GPS interface" << std::endl;

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
        requestHomePosition();

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

        std::cout << "Requested data stream from ArduPilot" << std::endl;
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

        std::cout << "Requested HOME_POSITION from ArduPilot" << std::endl;
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
        }
    }
};

class VIOGPSLogger {
private:
    uint64_t start_time_us;
    uint8_t  reset_counter;

    bool isTracking;
    bool prevIsTracking;

    bool hasLastGoodGPS;
    GPSCoord lastGoodGPS;

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
    VIOGPSLogger()
        : reset_counter(0)
        , isTracking(false)
        , prevIsTracking(false)
        , hasLastGoodGPS(false)
        , position_offset_x(0.0f)
        , position_offset_y(0.0f)
        , position_offset_z(0.0f)
    {
        auto now = std::chrono::high_resolution_clock::now();
        start_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
            now.time_since_epoch()).count();

        std::cout << "VIO GPS logging started." << std::endl;

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

    void setGPSOrigin(double lat, double lon, double alt) {
        mavlink->setGPSOrigin(lat, lon, alt);

        // Initialize last good GPS to origin
        lastGoodGPS.lat = lat;
        lastGoodGPS.lon = lon;
        lastGoodGPS.alt = alt;
    }

    bool waitForGPSOrigin() {
        return mavlink->waitForGPSOrigin(0);  // Wait indefinitely
    }

    bool hasGPSOrigin() const {
        return mavlink->hasGPSOrigin();
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

        if (!hasGPSOrigin()) {
            // Don't log if we don't have GPS origin yet
            return;
        }

        GPSCoord gps_origin = mavlink->getGPSOrigin();

        uint64_t timestamp_us = start_time_us + static_cast<uint64_t>(timestamp * 1e6);

        Sophus::SE3f Twc = Tcw.inverse();
        Eigen::Vector3f position = Twc.translation();

        float ned_x = position.z();
        float ned_y = -position.x();
        float ned_z = position.y();

        ned_x += position_offset_x;
        ned_y += position_offset_y;
        ned_z += position_offset_z;

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
            if (hasLastGoodGPS) {
                gps_data.lat = static_cast<int32_t>(lastGoodGPS.lat * 1e7);
                gps_data.lon = static_cast<int32_t>(lastGoodGPS.lon * 1e7);
                gps_data.alt = static_cast<int32_t>(lastGoodGPS.alt * 1000);
            } else {
                gps_data.lat = static_cast<int32_t>(gps_origin.lat * 1e7);
                gps_data.lon = static_cast<int32_t>(gps_origin.lon * 1e7);
                gps_data.alt = static_cast<int32_t>(gps_origin.alt * 1000);
            }

            gps_data.vn = 0.0f;
            gps_data.ve = 0.0f;
            gps_data.vd = 0.0f;
            gps_data.vel = 0;
            gps_data.vz = 0;

            gps_data.fix_type = 0;
            gps_data.satellites_visible = 0;
            gps_data.eph = 9999;
            gps_data.epv = 9999;

            std::cout << "!!! GPS: Tracking lost - using last known position" << std::endl;
        }
        else
        {
            GPSCoord current_gps = ned_to_gps(ned_x, ned_y, ned_z, gps_origin);

            if(hasLastGoodGPS) {
                double lat_diff = std::abs(current_gps.lat - lastGoodGPS.lat);
                double lon_diff = std::abs(current_gps.lon - lastGoodGPS.lon);

                if(lat_diff > 0.0001 || lon_diff > 0.0001) {
                    std::cout << "@@@ Large position jump detected - adjusting offset" << std::endl;

                    GPSCoord offset_gps = ned_to_gps(ned_x - position_offset_x,
                                                      ned_y - position_offset_y,
                                                      ned_z - position_offset_z,
                                                      gps_origin);

                    double delta_lat = lastGoodGPS.lat - offset_gps.lat;
                    double delta_lon = lastGoodGPS.lon - offset_gps.lon;

                    double lat_rad = gps_origin.lat * M_PI / 180.0;
                    position_offset_x += delta_lat * EARTH_RADIUS * M_PI / 180.0;
                    position_offset_y += delta_lon * EARTH_RADIUS * cos(lat_rad) * M_PI / 180.0;

                    ned_x += position_offset_x;
                    ned_y += position_offset_y;
                    current_gps = ned_to_gps(ned_x, ned_y, ned_z, gps_origin);

                    IncrementResetCounter();
                }
            }

            gps_data.lat = static_cast<int32_t>(current_gps.lat * 1e7);
            gps_data.lon = static_cast<int32_t>(current_gps.lon * 1e7);
            gps_data.alt = static_cast<int32_t>(current_gps.alt * 1000);

            gps_data.vn = ned_vx;
            gps_data.ve = ned_vy;
            gps_data.vd = ned_vz;

            float ground_speed = std::sqrt(ned_vx*ned_vx + ned_vy*ned_vy);
            gps_data.vel = static_cast<uint16_t>(ground_speed * 100);
            gps_data.vz = static_cast<int16_t>(ned_vz * 100);

            gps_data.cog = static_cast<uint16_t>(std::atan2(ned_vy, ned_vx) * 180.0 / M_PI * 100);

            gps_data.fix_type = 3;
            gps_data.satellites_visible = 12;
            gps_data.eph = 50;
            gps_data.epv = 50;

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
    // Accept either 3 arguments (vocab, settings) or 6 arguments (vocab, settings, lat, lon, alt)
    if(argc != 3 && argc != 6) {
        std::cerr << "Usage: ./stereo_inertial_realsense_gps path_to_vocabulary path_to_settings [origin_lat origin_lon origin_alt]" << std::endl;
        std::cerr << "\nWith GPS origin from command line:" << std::endl;
        std::cerr << "  ./stereo_inertial_realsense_gps ORBvoc.txt settings.yaml 37.7749 -122.4194 10.0" << std::endl;
        std::cerr << "\nWithout GPS origin (wait for ArduPilot HOME_POSITION):" << std::endl;
        std::cerr << "  ./stereo_inertial_realsense_gps ORBvoc.txt settings.yaml" << std::endl;
        return 1;
    }

    bool gps_from_cmdline = (argc == 6);

    VIOGPSLogger vio_logger;

    if (gps_from_cmdline) {
        double origin_lat = std::atof(argv[3]);
        double origin_lon = std::atof(argv[4]);
        double origin_alt = std::atof(argv[5]);
        vio_logger.setGPSOrigin(origin_lat, origin_lon, origin_alt);
    }

    // Configure RealSense
    rs2::pipeline pipe;
    rs2::config cfg;

    cfg.enable_stream(RS2_STREAM_INFRARED, 1, 640, 480, RS2_FORMAT_Y8, 30);
    cfg.enable_stream(RS2_STREAM_INFRARED, 2, 640, 480, RS2_FORMAT_Y8, 30);
    cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
    cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F);

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

    std::cout << "\n========================================" << std::endl;
    std::cout << "Connecting to ArduPilot..." << std::endl;
    std::cout << "========================================\n" << std::endl;

    if(!vio_logger.WaitMavlink(30))
    {
        std::cout << "MavLink Connection failed." << std::endl;
        return 1;
    }
    else
    {
        std::cout << "MavLink Connected!" << std::endl;
    }

    // Wait for GPS origin if not provided from command line
    if (!gps_from_cmdline) {
        if (!vio_logger.waitForGPSOrigin()) {
            std::cout << "Failed to get GPS origin from ArduPilot!" << std::endl;
            return 1;
        }
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "GPS Origin Ready - Starting ORB-SLAM3" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Create SLAM system AFTER GPS origin is available
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::IMU_STEREO, false);

    std::cout << "Starting VIO GPS tracking..." << std::endl;
    std::cout << "Move the camera to initialize the system." << std::endl;

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
