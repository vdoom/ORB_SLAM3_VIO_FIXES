/**
 * Pear VIO Camera API - Serial IMU Reader Implementation
 */

#include "pear/imu.h"

#include <iostream>
#include <regex>
#include <chrono>
#include <cstring>

// Serial port includes
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <errno.h>

using namespace std;

namespace pear {

SerialIMUReader::SerialIMUReader() = default;

SerialIMUReader::~SerialIMUReader() {
    close();
}

bool SerialIMUReader::open(const IMUConfig& config) {
    config_ = config;
    return open(config.serialPort, config.baudRate);
}

bool SerialIMUReader::open(const string& portName, int baudRate) {
    config_.serialPort = portName;
    config_.baudRate = baudRate;

    fd_ = ::open(portName.c_str(), O_RDWR | O_NOCTTY);
    if (fd_ < 0) {
        cerr << "Failed to open " << portName << ": " << strerror(errno) << endl;
        return false;
    }

    struct termios tty;
    if (tcgetattr(fd_, &tty) != 0) {
        cerr << "Failed to get terminal attributes: " << strerror(errno) << endl;
        ::close(fd_);
        fd_ = -1;
        return false;
    }

    // Set baud rate
    speed_t speed = B115200;
    switch (baudRate) {
        case 9600:   speed = B9600; break;
        case 19200:  speed = B19200; break;
        case 38400:  speed = B38400; break;
        case 57600:  speed = B57600; break;
        case 115200: speed = B115200; break;
        case 230400: speed = B230400; break;
        case 460800: speed = B460800; break;
        case 921600: speed = B921600; break;
        default:
            cerr << "Unsupported baud rate: " << baudRate << ", using 115200" << endl;
            speed = B115200;
    }
    cfsetospeed(&tty, speed);
    cfsetispeed(&tty, speed);

    // 8N1 mode
    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;
    tty.c_cflag &= ~CRTSCTS;
    tty.c_cflag |= CREAD | CLOCAL;

    // Non-canonical mode
    tty.c_lflag &= ~(ICANON | ECHO | ECHOE | ECHONL | ISIG);
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL);
    tty.c_oflag &= ~(OPOST | ONLCR);

    // Read timeout
    tty.c_cc[VTIME] = 1;  // 100ms timeout
    tty.c_cc[VMIN] = 0;

    if (tcsetattr(fd_, TCSANOW, &tty) != 0) {
        cerr << "Failed to set terminal attributes: " << strerror(errno) << endl;
        ::close(fd_);
        fd_ = -1;
        return false;
    }

    tcflush(fd_, TCIOFLUSH);
    isOpen_ = true;
    shouldStop_ = false;

    // Start read thread
    readThread_ = thread(&SerialIMUReader::readLoop, this);

    cout << "Serial port opened: " << portName << " at " << baudRate << " baud" << endl;
    return true;
}

void SerialIMUReader::close() {
    shouldStop_ = true;
    if (readThread_.joinable()) {
        readThread_.join();
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    isOpen_ = false;
}

vector<IMUData> SerialIMUReader::getIMUData() {
    return imuQueue_.popAll();
}

uint64_t SerialIMUReader::getCameraTriggerTimestamp() {
    lock_guard<mutex> lock(triggerMtx_);
    if (triggerQueue_.empty()) {
        return 0;
    }
    uint64_t ts = triggerQueue_.front();
    triggerQueue_.pop();
    return ts;
}

bool SerialIMUReader::hasTriggerTimestamp() {
    lock_guard<mutex> lock(triggerMtx_);
    return !triggerQueue_.empty();
}

void SerialIMUReader::clearTriggerQueue() {
    lock_guard<mutex> lock(triggerMtx_);
    while (!triggerQueue_.empty()) {
        triggerQueue_.pop();
    }
}

void SerialIMUReader::clearIMUQueue() {
    imuQueue_.clear();
}

void SerialIMUReader::readLoop() {
    string lineBuffer;
    char buf[256];

    // Regex patterns for parsing Pico output
    regex imuRegex(R"(timestamp:\s*(\d+);\s*\[INT\s*#(\d+)\]\s*A:\s*([-\d.]+),([-\d.]+),([-\d.]+)\s*g\s*\|\s*G:\s*([-\d.]+),([-\d.]+),([-\d.]+))");
    regex triggerRegex(R"(Camera triggered timestamp:\s*(\d+))");

    while (!shouldStop_ && fd_ >= 0) {
        int n = ::read(fd_, buf, sizeof(buf) - 1);
        if (n > 0) {
            buf[n] = '\0';
            lineBuffer += buf;

            size_t pos;
            while ((pos = lineBuffer.find('\n')) != string::npos) {
                string line = lineBuffer.substr(0, pos);
                lineBuffer = lineBuffer.substr(pos + 1);

                // Trim whitespace
                size_t start = line.find_first_not_of(" \t\r\n");
                size_t end = line.find_last_not_of(" \t\r\n");
                if (start != string::npos && end != string::npos) {
                    line = line.substr(start, end - start + 1);
                } else {
                    continue;
                }

                if (line.empty()) continue;

                // Try to parse as camera trigger
                smatch triggerMatch;
                if (regex_search(line, triggerMatch, triggerRegex)) {
                    uint64_t ts = stoull(triggerMatch[1].str());
                    lock_guard<mutex> lock(triggerMtx_);
                    triggerQueue_.push(ts);
                    continue;
                }

                // Try to parse as IMU data
                smatch imuMatch;
                if (regex_search(line, imuMatch, imuRegex)) {
                    IMUData data;
                    data.timestamp_ms = stoull(imuMatch[1].str());
                    data.interrupt_count = stoul(imuMatch[2].str());
                    data.accel_x = stof(imuMatch[3].str());
                    data.accel_y = stof(imuMatch[4].str());
                    data.accel_z = stof(imuMatch[5].str());
                    data.gyro_x = stof(imuMatch[6].str());
                    data.gyro_y = stof(imuMatch[7].str());
                    data.gyro_z = stof(imuMatch[8].str());

                    imuQueue_.push(data);
                }
            }

            // Prevent buffer overflow
            if (lineBuffer.size() > 1024) {
                lineBuffer.clear();
            }
        }
        this_thread::sleep_for(chrono::microseconds(100));
    }
}

}  // namespace pear
