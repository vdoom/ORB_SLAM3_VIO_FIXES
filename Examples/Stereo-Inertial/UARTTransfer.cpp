#include "UARTTransfer.h"
#include <iostream>
#include <iomanip>    // For setprecision
#include <fstream>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <errno.h>
#include <thread>
#include <chrono>

UARTTransfer::UARTTransfer(const std::string& port, int baudrate)
    : port_(port), baudrate_(baudrate), fd_(-1), connected_(false) {
}

UARTTransfer::~UARTTransfer() {
    disconnect();
}

bool UARTTransfer::connect() {
    if (connected_) {
        std::cerr << "Already connected" << std::endl;
        return true;
    }
    
    // Open UART device
    fd_ = open(port_.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
    if (fd_ < 0) {
        std::cerr << "✗ Failed to open " << port_ << ": " << strerror(errno) << std::endl;
        return false;
    }
    
    // Configure port
    if (!configurePort()) {
        close(fd_);
        fd_ = -1;
        return false;
    }
    
    connected_ = true;
    std::cout << "✓ Connected to " << port_ << " at " << baudrate_ << " baud" << std::endl;
    return true;
}

void UARTTransfer::disconnect() {
    if (connected_ && fd_ >= 0) {
        close(fd_);
        fd_ = -1;
        connected_ = false;
        std::cout << "✓ Connection closed" << std::endl;
    }
}

bool UARTTransfer::isConnected() const {
    return connected_;
}

bool UARTTransfer::configurePort() {
    struct termios tty;
    
    // Get current settings
    if (tcgetattr(fd_, &tty) != 0) {
        std::cerr << "✗ Error getting termios attributes: " << strerror(errno) << std::endl;
        return false;
    }
    
    // Set baud rate
    speed_t speed = getBaudrateConstant(baudrate_);
    cfsetospeed(&tty, speed);
    cfsetispeed(&tty, speed);
    
    // Configure: 8N1 (8 data bits, no parity, 1 stop bit)
    tty.c_cflag &= ~PARENB;        // No parity
    tty.c_cflag &= ~CSTOPB;        // 1 stop bit
    tty.c_cflag &= ~CSIZE;         // Clear size bits
    tty.c_cflag |= CS8;            // 8 data bits
    tty.c_cflag &= ~CRTSCTS;       // No hardware flow control
    tty.c_cflag |= CREAD | CLOCAL; // Enable receiver, ignore modem control lines
    
    // Disable canonical mode and echo
    tty.c_lflag &= ~ICANON;
    tty.c_lflag &= ~ECHO;
    tty.c_lflag &= ~ECHOE;
    tty.c_lflag &= ~ECHONL;
    tty.c_lflag &= ~ISIG;
    
    // Disable software flow control
    tty.c_iflag &= ~(IXON | IXOFF | IXANY);
    tty.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL);
    
    // Raw output mode
    tty.c_oflag &= ~OPOST;
    tty.c_oflag &= ~ONLCR;
    
    // Set timeouts
    tty.c_cc[VTIME] = 10;  // Wait for up to 1s (10 deciseconds)
    tty.c_cc[VMIN] = 0;    // Return as soon as any data is received
    
    // Apply settings
    if (tcsetattr(fd_, TCSANOW, &tty) != 0) {
        std::cerr << "✗ Error setting termios attributes: " << strerror(errno) << std::endl;
        return false;
    }
    
    return true;
}

speed_t UARTTransfer::getBaudrateConstant(int baudrate) const {
    switch (baudrate) {
        case 9600:    return B9600;
        case 19200:   return B19200;
        case 38400:   return B38400;
        case 57600:   return B57600;
        case 115200:  return B115200;
        case 230400:  return B230400;
        case 460800:  return B460800;
        case 921600:  return B921600;
        default:
            std::cerr << "Warning: Unsupported baudrate " << baudrate 
                      << ", using 115200" << std::endl;
            return B115200;
    }
}

int UARTTransfer::sendData(const std::string& data) {
    if (!connected_ || fd_ < 0) {
        std::cerr << "✗ Error: Connection not established" << std::endl;
        return -1;
    }
    
    int bytes_written = write(fd_, data.c_str(), data.length());
    if (bytes_written < 0) {
        std::cerr << "✗ Write error: " << strerror(errno) << std::endl;
        return -1;
    }
    
    tcdrain(fd_);  // Wait for data to be transmitted
    std::cout << "✓ Sent " << bytes_written << " bytes" << std::endl;
    return bytes_written;
}

int UARTTransfer::sendData(const std::vector<uint8_t>& data) {
    if (!connected_ || fd_ < 0) {
        std::cerr << "✗ Error: Connection not established" << std::endl;
        return -1;
    }
    
    int bytes_written = write(fd_, data.data(), data.size());
    if (bytes_written < 0) {
        std::cerr << "✗ Write error: " << strerror(errno) << std::endl;
        return -1;
    }
    
    tcdrain(fd_);
    std::cout << "✓ Sent " << bytes_written << " bytes" << std::endl;
    return bytes_written;
}

int UARTTransfer::receiveData(std::vector<uint8_t>& buffer, size_t maxSize, int timeoutMs) {
    if (!connected_ || fd_ < 0) {
        std::cerr << "✗ Error: Connection not established" << std::endl;
        return -1;
    }
    
    buffer.resize(maxSize);
    
    // Set timeout
    struct timeval timeout;
    timeout.tv_sec = timeoutMs / 1000;
    timeout.tv_usec = (timeoutMs % 1000) * 1000;
    
    fd_set readfds;
    FD_ZERO(&readfds);
    FD_SET(fd_, &readfds);
    
    int ret = select(fd_ + 1, &readfds, nullptr, nullptr, &timeout);
    if (ret < 0) {
        std::cerr << "✗ Select error: " << strerror(errno) << std::endl;
        return -1;
    } else if (ret == 0) {
        // Timeout
        buffer.clear();
        return 0;
    }
    
    int bytes_read = read(fd_, buffer.data(), maxSize);
    if (bytes_read < 0) {
        std::cerr << "✗ Read error: " << strerror(errno) << std::endl;
        return -1;
    }
    
    buffer.resize(bytes_read);
    if (bytes_read > 0) {
        std::cout << "✓ Received " << bytes_read << " bytes" << std::endl;
    }
    return bytes_read;
}

int UARTTransfer::receiveData(std::string& data, size_t maxSize, int timeoutMs) {
    std::vector<uint8_t> buffer;
    int bytes_read = receiveData(buffer, maxSize, timeoutMs);
    
    if (bytes_read > 0) {
        data.assign(buffer.begin(), buffer.end());
    }
    
    return bytes_read;
}

bool UARTTransfer::setBaudrate(int baudrate) {
    baudrate_ = baudrate;
    if (connected_) {
        return configurePort();
    }
    return true;
}

int UARTTransfer::getBaudrate() const {
    return baudrate_;
}

void UARTTransfer::flush() {
    if (connected_ && fd_ >= 0) {
        tcflush(fd_, TCIOFLUSH);
    }
}

int UARTTransfer::available() const {
    if (!connected_ || fd_ < 0) {
        return -1;
    }
    
    int bytes;
    if (ioctl(fd_, FIONREAD, &bytes) < 0) {
        return -1;
    }
    return bytes;
}

std::string UARTTransfer::getFilenameFromPath(const std::string& filepath) const {
    size_t pos = filepath.find_last_of("/\\");
    if (pos != std::string::npos) {
        return filepath.substr(pos + 1);
    }
    return filepath;
}

size_t UARTTransfer::getFileSize(const std::string& filepath) const {
    struct stat st;
    if (stat(filepath.c_str(), &st) == 0) {
        return st.st_size;
    }
    return 0;
}

bool UARTTransfer::sendFile(const std::string& filepath, 
                            void (*callback)(size_t current, size_t total)) {
    // Check if file exists
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        std::cerr << "✗ File not found: " << filepath << std::endl;
        return false;
    }
    
    // Get file size and name
    size_t filesize = getFileSize(filepath);
    std::string filename = getFilenameFromPath(filepath);
    
    std::cout << "Sending file: " << filename << " (" << filesize << " bytes)" << std::endl;
    
    // Send header: FILE:filename:size\n
    std::string header = "FILE:" + filename + ":" + std::to_string(filesize) + "\n";
    if (sendData(header) < 0) {
        std::cerr << "✗ Failed to send file header" << std::endl;
        return false;
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Send file data in chunks
    const size_t chunkSize = 1024;
    std::vector<uint8_t> buffer(chunkSize);
    size_t totalSent = 0;
    
    while (file) {
        file.read(reinterpret_cast<char*>(buffer.data()), chunkSize);
        std::streamsize bytesRead = file.gcount();
        
        if (bytesRead > 0) {
            buffer.resize(bytesRead);
            int bytesSent = write(fd_, buffer.data(), bytesRead);
            if (bytesSent < 0) {
                std::cerr << "✗ Error sending file data" << std::endl;
                return false;
            }
            
            totalSent += bytesSent;
            
            if (callback) {
                callback(totalSent, filesize);
            } else {
                float progress = (float)totalSent / filesize * 100.0f;
                std::cout << "\rProgress: " << std::fixed << std::setprecision(1) 
                          << progress << "% (" << totalSent << "/" << filesize << " bytes)" 
                          << std::flush;
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            buffer.resize(chunkSize);
        }
    }
    
    std::cout << std::endl << "✓ File sent successfully: " << filename << std::endl;
    return true;
}

bool UARTTransfer::receiveFile(const std::string& saveDir, 
                               void (*callback)(size_t current, size_t total)) {
    std::cout << "Waiting for file header..." << std::endl;
    
    // Receive header
    std::string header;
    std::string headerBuffer;
    auto startTime = std::chrono::steady_clock::now();
    const int headerTimeoutSec = 30;
    
    while (headerBuffer.find('\n') == std::string::npos) {
        auto currentTime = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(currentTime - startTime).count();
        
        if (elapsed > headerTimeoutSec) {
            std::cerr << "✗ Timeout waiting for file header" << std::endl;
            return false;
        }
        
        std::string chunk;
        int bytes = receiveData(chunk, 1, 100);
        if (bytes > 0) {
            headerBuffer += chunk;
        }
    }
    
    header = headerBuffer.substr(0, headerBuffer.find('\n'));
    
    // Parse header: FILE:filename:size
    if (header.substr(0, 5) != "FILE:") {
        std::cerr << "✗ Invalid file header" << std::endl;
        return false;
    }
    
    size_t firstColon = header.find(':', 5);
    size_t secondColon = header.find(':', firstColon + 1);
    
    if (firstColon == std::string::npos || secondColon == std::string::npos) {
        std::cerr << "✗ Malformed file header" << std::endl;
        return false;
    }
    
    std::string filename = header.substr(5, firstColon - 5);
    size_t filesize = std::stoull(header.substr(firstColon + 1));
    
    std::cout << "Receiving file: " << filename << " (" << filesize << " bytes)" << std::endl;
    
    // Create output file
    std::string filepath = saveDir + "/" + filename;
    std::ofstream outFile(filepath, std::ios::binary);
    if (!outFile) {
        std::cerr << "✗ Failed to create output file: " << filepath << std::endl;
        return false;
    }
    
    // Receive file data
    size_t totalReceived = 0;
    std::vector<uint8_t> buffer;
    
    while (totalReceived < filesize) {
        size_t remaining = filesize - totalReceived;
        size_t toRead = std::min(remaining, size_t(1024));
        
        int bytes = receiveData(buffer, toRead, 5000);
        if (bytes > 0) {
            outFile.write(reinterpret_cast<char*>(buffer.data()), bytes);
            totalReceived += bytes;
            
            if (callback) {
                callback(totalReceived, filesize);
            } else {
                float progress = (float)totalReceived / filesize * 100.0f;
                std::cout << "\rProgress: " << std::fixed << std::setprecision(1) 
                          << progress << "% (" << totalReceived << "/" << filesize << " bytes)" 
                          << std::flush;
            }
        } else if (bytes < 0) {
            std::cerr << std::endl << "✗ Error receiving file data" << std::endl;
            return false;
        }
    }
    
    std::cout << std::endl << "✓ File received successfully: " << filepath << std::endl;
    return true;
}



