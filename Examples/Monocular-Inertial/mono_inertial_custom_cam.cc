 
/**
 * Monocular-Inertial ORB-SLAM3 for Custom Camera (OV9281 + BMI160)
 *
 * Based on mono_inertial_realsense_D435i.cc but adapted for:
 * - libcamera API instead of librealsense
 * - BMI160 IMU via Raspberry Pi Pico over serial
 * - Hardware-synchronized camera trigger (Pico triggers camera every 10 IMU samples)
 * - Both accel and gyro at 200Hz (no interpolation needed)
 *
 * Usage: ./mono_inertial_custom_cam path_to_vocabulary path_to_settings [serial_port]
 *        Default serial port: /dev/ttyACM0
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
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <queue>
#include <regex>
 
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
 
#include <libcamera/libcamera.h>
#include <libcamera/control_ids.h>
 
#include <System.h>
 
// Serial port includes
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <sys/mman.h>
#include <cstring>
#include <errno.h>
 
using namespace std;
using namespace libcamera;
 
// ============================================================================
// Global state
// ============================================================================
 
atomic<bool> b_continue_session{true};
 
void exit_loop_handler(int s) {
    cout << "\nFinishing session..." << endl;
    b_continue_session = false;
}
 
// ============================================================================
// IMU Data structures
// ============================================================================
 
struct IMUData {
    uint64_t timestamp_ms = 0;
    uint32_t interrupt_count = 0;
    float accel_x = 0, accel_y = 0, accel_z = 0;  // in g
    float gyro_x = 0, gyro_y = 0, gyro_z = 0;     // in deg/s
};
 
// Thread-safe queue for IMU data
class IMUQueue {
public:
    void push(const IMUData& data) {
        lock_guard<mutex> lock(mtx_);
        queue_.push(data);
    }
 
    bool pop(IMUData& data) {
        lock_guard<mutex> lock(mtx_);
        if (queue_.empty()) return false;
        data = queue_.front();
        queue_.pop();
        return true;
    }
 
    vector<IMUData> popAll() {
        lock_guard<mutex> lock(mtx_);
        vector<IMUData> result;
        while (!queue_.empty()) {
            result.push_back(queue_.front());
            queue_.pop();
        }
        return result;
    }
 
    size_t size() const {
        lock_guard<mutex> lock(mtx_);
        return queue_.size();
    }
 
private:
    mutable mutex mtx_;
    queue<IMUData> queue_;
};
 
// ============================================================================
// Camera Settings (loaded from viewer_app INI file)
// ============================================================================
 
struct CameraSettings {
    bool autoExposure = true;
    int exposureTimeUs = 10000;   // microseconds
    float gain = 8.0f;
    bool triggerMode = false;
 
    // Load settings from INI file (same format as viewer_app)
    bool loadFromIniFile(const string& path) {
        ifstream file(path);
        if (!file.is_open()) {
            return false;
        }
 
        string line;
        string currentSection;
 
        while (getline(file, line)) {
            // Trim whitespace
            size_t start = line.find_first_not_of(" \t\r\n");
            if (start == string::npos) continue;
            size_t end = line.find_last_not_of(" \t\r\n");
            line = line.substr(start, end - start + 1);
 
            // Skip empty lines and comments
            if (line.empty() || line[0] == '#' || line[0] == ';') continue;
 
            // Section header
            if (line[0] == '[' && line.back() == ']') {
                currentSection = line.substr(1, line.size() - 2);
                continue;
            }
 
            // Key=value pair
            size_t eqPos = line.find('=');
            if (eqPos == string::npos) continue;
 
            string key = line.substr(0, eqPos);
            string value = line.substr(eqPos + 1);
 
            // Trim key and value
            start = key.find_first_not_of(" \t");
            end = key.find_last_not_of(" \t");
            if (start != string::npos) key = key.substr(start, end - start + 1);
 
            start = value.find_first_not_of(" \t");
            end = value.find_last_not_of(" \t");
            if (start != string::npos) value = value.substr(start, end - start + 1);
 
            // Parse camera section
            if (currentSection == "camera") {
                if (key == "auto_exposure") {
                    autoExposure = (value == "true" || value == "1");
                } else if (key == "manual_exposure_us") {
                    exposureTimeUs = stoi(value);
                } else if (key == "manual_gain") {
                    gain = stof(value);
                } else if (key == "trigger_mode") {
                    triggerMode = (value == "true" || value == "1");
                }
            }
        }
 
        return true;
    }
 
    // Get default config path (~/.config/rpi_camera_viewer.ini)
    static string getDefaultConfigPath() {
        const char* home = getenv("HOME");
        if (home) {
            return string(home) + "/.config/rpi_camera_viewer.ini";
        }
        return "";
    }
};
 
// ============================================================================
// Serial IMU Reader
// ============================================================================
 
class SerialIMUReader {
public:
    SerialIMUReader() = default;
    ~SerialIMUReader() { close(); }
 
    bool open(const string& portName, int baudRate = 115200) {
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
            case 115200: speed = B115200; break;
            case 230400: speed = B230400; break;
            case 460800: speed = B460800; break;
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
        tty.c_cc[VTIME] = 1;
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
 
    void close() {
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
 
    bool isOpen() const { return isOpen_; }
 
    // Get all accumulated IMU data
    vector<IMUData> getIMUData() {
        return imuQueue_.popAll();
    }
 
    // Get next camera trigger timestamp from queue (returns 0 if empty)
    uint64_t getCameraTriggerTimestamp() {
        lock_guard<mutex> lock(triggerMtx_);
        if (triggerQueue_.empty()) {
            return 0;
        }
        uint64_t ts = triggerQueue_.front();
        triggerQueue_.pop();
        return ts;
    }
 
private:
    void readLoop() {
        string lineBuffer;
        char buf[256];
 
        // Regex patterns
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
                    line.erase(0, line.find_first_not_of(" \t\r\n"));
                    line.erase(line.find_last_not_of(" \t\r\n") + 1);
 
                    if (line.empty()) continue;
 
                    // Try to parse as camera trigger
                    smatch triggerMatch;
                    if (regex_search(line, triggerMatch, triggerRegex)) {
                        uint64_t ts = stoull(triggerMatch[1].str());
                        lock_guard<mutex> lock(triggerMtx_);
                        triggerQueue_.push(ts);  // Queue trigger (FIFO order)
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
 
    int fd_ = -1;
    atomic<bool> isOpen_{false};
    atomic<bool> shouldStop_{false};
    thread readThread_;
    IMUQueue imuQueue_;
 
    mutex triggerMtx_;
    queue<uint64_t> triggerQueue_;  // Queue of camera trigger timestamps (FIFO)
};
 
// ============================================================================
// Camera Controller (libcamera)
// ============================================================================
 
class CameraController {
public:
    CameraController() = default;
    ~CameraController() { stop(); }
 
    bool initialize(int cameraIndex = 0) {
        cameraManager_ = make_unique<CameraManager>();
        int ret = cameraManager_->start();
        if (ret < 0) {
            cerr << "Failed to start camera manager: " << strerror(-ret) << endl;
            return false;
        }
 
        auto cameras = cameraManager_->cameras();
        if (cameras.empty()) {
            cerr << "No cameras found" << endl;
            return false;
        }
 
        if (cameraIndex >= static_cast<int>(cameras.size())) {
            cerr << "Camera index " << cameraIndex << " out of range" << endl;
            return false;
        }
 
        camera_ = cameras[cameraIndex];
        ret = camera_->acquire();
        if (ret < 0) {
            cerr << "Failed to acquire camera: " << strerror(-ret) << endl;
            return false;
        }
 
        cout << "Acquired camera: " << camera_->id() << endl;
 
        // Configure for 1280x800 R8 (grayscale)
        config_ = camera_->generateConfiguration({StreamRole::Viewfinder});
        if (!config_) {
            cerr << "Failed to generate camera configuration" << endl;
            return false;
        }
 
        StreamConfiguration& streamConfig = config_->at(0);
        streamConfig.size = Size(640, 400);
        streamConfig.pixelFormat = formats::R8;
 
        auto status = config_->validate();
        if (status == CameraConfiguration::Invalid) {
            cerr << "Invalid camera configuration" << endl;
            return false;
        }
 
        if (status == CameraConfiguration::Adjusted) {
            cout << "Camera configuration adjusted to: "
                 << streamConfig.size.width << "x" << streamConfig.size.height
                 << " " << streamConfig.pixelFormat.toString() << endl;
        }
 
        ret = camera_->configure(config_.get());
        if (ret < 0) {
            cerr << "Failed to configure camera: " << strerror(-ret) << endl;
            return false;
        }
 
        frameWidth_ = streamConfig.size.width;
        frameHeight_ = streamConfig.size.height;
        frameStride_ = streamConfig.stride;
        pixelFormat_ = streamConfig.pixelFormat;
        stream_ = streamConfig.stream();
 
        // Allocate buffers
        allocator_ = make_unique<FrameBufferAllocator>(camera_);
        ret = allocator_->allocate(stream_);
        if (ret < 0) {
            cerr << "Failed to allocate buffers: " << strerror(-ret) << endl;
            return false;
        }
 
        cout << "Allocated " << allocator_->buffers(stream_).size() << " buffers" << endl;
 
        // Map buffers
        for (const auto& buffer : allocator_->buffers(stream_)) {
            for (const auto& plane : buffer->planes()) {
                size_t length = plane.length;
                if (length == 0) {
                    length = lseek(plane.fd.get(), 0, SEEK_END);
                    lseek(plane.fd.get(), 0, SEEK_SET);
                }
                void* mem = mmap(nullptr, length, PROT_READ, MAP_SHARED, plane.fd.get(), 0);
                if (mem == MAP_FAILED) {
                    cerr << "Failed to mmap buffer: " << strerror(errno) << endl;
                    return false;
                }
                mappedBuffers_[buffer.get()] = {static_cast<uint8_t*>(mem), length};
            }
        }
 
        // Create requests
        for (const auto& buffer : allocator_->buffers(stream_)) {
            auto request = camera_->createRequest();
            if (!request) {
                cerr << "Failed to create request" << endl;
                return false;
            }
            request->addBuffer(stream_, buffer.get());
            requests_.push_back(move(request));
        }
 
        // Connect callback
        camera_->requestCompleted.connect(this, &CameraController::requestComplete);
 
        cout << "Camera initialized: " << frameWidth_ << "x" << frameHeight_
             << " stride: " << frameStride_
             << " format: " << pixelFormat_.toString() << endl;

        // Check format support
        if (pixelFormat_ == formats::R8) {
            cout << "  -> Using R8 (8-bit grayscale) - optimal" << endl;
        } else if (pixelFormat_ == formats::R16) {
            cout << "  -> Using R16 (16-bit grayscale) - will convert 10-bit to 8-bit" << endl;
        } else if (pixelFormat_ == formats::XRGB8888 || pixelFormat_ == formats::XBGR8888) {
            cout << "  -> Using XRGB8888 (32-bit) - will extract grayscale channel" << endl;
        } else {
            cout << "  -> Warning: Format " << pixelFormat_.toString()
                 << " may not be fully supported" << endl;
        }

        return true;
    }
 
    // Load camera settings from INI file
    bool loadSettings(const string& configPath = "") {
        string path = configPath.empty() ? CameraSettings::getDefaultConfigPath() : configPath;
 
        if (path.empty()) {
            cout << "No config path available, using default camera settings" << endl;
            return false;
        }
 
        if (settings_.loadFromIniFile(path)) {
            cout << "Loaded camera settings from: " << path << endl;
            cout << "  Auto exposure: " << (settings_.autoExposure ? "enabled" : "disabled") << endl;
            if (!settings_.autoExposure) {
                cout << "  Manual exposure: " << settings_.exposureTimeUs << " us" << endl;
                cout << "  Manual gain: " << settings_.gain << endl;
            }
            return true;
        } else {
            cout << "Config file not found: " << path << endl;
            cout << "Using default camera settings (auto exposure)" << endl;
            return false;
        }
    }
 
    bool start() {
        if (!camera_) return false;
        if (running_) return true;
 
        // Set frame duration for consistent FPS (50ms = 20 FPS for triggered mode)
        ControlList controls;
        int64_t frameDuration = 50000;  // 50ms = 20 FPS
        controls.set(controls::FrameDurationLimits,
                     Span<const int64_t, 2>({frameDuration, frameDuration}));
 
        // Apply exposure settings from loaded config
        if (settings_.autoExposure) {
            controls.set(controls::AeEnable, true);
            cout << "Camera using auto exposure" << endl;
        } else {
            // Manual exposure mode - must set BOTH exposure time AND gain
            controls.set(controls::AeEnable, false);
            controls.set(controls::ExposureTime, settings_.exposureTimeUs);
            controls.set(controls::AnalogueGain, settings_.gain);
            cout << "Camera using manual exposure: " << settings_.exposureTimeUs
                 << " us, gain: " << settings_.gain << endl;
        }
 
        int ret = camera_->start(&controls);
        if (ret < 0) {
            cerr << "Failed to start camera: " << strerror(-ret) << endl;
            return false;
        }
 
        running_ = true;
 
        // Queue all requests
        for (auto& request : requests_) {
            camera_->queueRequest(request.get());
        }
 
        cout << "Camera started" << endl;
        return true;
    }
 
    void stop() {
        if (!running_) return;
        running_ = false;
        if (camera_) {
            camera_->stop();
        }
 
        // Cleanup
        for (auto& [buffer, info] : mappedBuffers_) {
            if (info.data) {
                munmap(info.data, info.size);
            }
        }
        mappedBuffers_.clear();
 
        if (camera_) {
            camera_->release();
            camera_.reset();
        }
        if (cameraManager_) {
            cameraManager_->stop();
            cameraManager_.reset();
        }
 
        cout << "Camera stopped" << endl;
    }
 
    // Wait for next frame (blocking)
    bool getFrame(cv::Mat& frame, double& timestamp) {
        unique_lock<mutex> lock(frameMtx_);
        if (!frameReady_) {
            frameCond_.wait(lock, [this] { return frameReady_ || !running_; });
        }
 
        if (!running_) return false;
 
        frame = currentFrame_.clone();
        timestamp = currentTimestamp_;
        frameReady_ = false;
        return true;
    }
 
    int width() const { return frameWidth_; }
    int height() const { return frameHeight_; }
    bool isRunning() const { return running_; }
 
private:
    void requestComplete(Request* request) {
        if (!running_) return;
        if (request->status() == Request::RequestCancelled) return;
 
        // Get buffer
        FrameBuffer* buffer = request->buffers().begin()->second;
        auto it = mappedBuffers_.find(buffer);
        if (it == mappedBuffers_.end()) return;
 
        const auto& info = it->second;

        // Get timestamp (SensorTimestamp is in nanoseconds)
        auto tsOpt = request->metadata().get(controls::SensorTimestamp);
        double timestamp = tsOpt.value_or(0) / 1e9;  // Convert to seconds

        // Convert to cv::Mat based on pixel format
        cv::Mat gray;

        if (pixelFormat_ == formats::R8) {
            // Direct 8-bit grayscale
            gray = cv::Mat(frameHeight_, frameWidth_, CV_8UC1, info.data, frameStride_);
        } else if (pixelFormat_ == formats::R16) {
            // 16-bit grayscale - OV9281 outputs 10-bit packed into 16-bit
            // Data is in upper bits, shift right by 8 to get 8-bit value
            gray = cv::Mat(frameHeight_, frameWidth_, CV_8UC1);
            const uint16_t* src = reinterpret_cast<const uint16_t*>(info.data);
            int srcStride = frameStride_ / 2;  // Stride is in bytes, convert to uint16 elements

            for (int y = 0; y < frameHeight_; y++) {
                const uint16_t* srcLine = src + y * srcStride;
                uint8_t* dstLine = gray.ptr<uint8_t>(y);
                for (int x = 0; x < frameWidth_; x++) {
                    dstLine[x] = static_cast<uint8_t>(srcLine[x] >> 8);
                }
            }
        } else if (pixelFormat_ == formats::XRGB8888 || pixelFormat_ == formats::XBGR8888) {
            // 32-bit XRGB - for mono camera, all channels are same, just take one
            gray = cv::Mat(frameHeight_, frameWidth_, CV_8UC1);
            for (int y = 0; y < frameHeight_; y++) {
                const uint8_t* srcLine = info.data + y * frameStride_;
                uint8_t* dstLine = gray.ptr<uint8_t>(y);
                for (int x = 0; x < frameWidth_; x++) {
                    // Take blue channel (first byte in BGRX layout)
                    dstLine[x] = srcLine[x * 4];
                }
            }
        } else if (pixelFormat_ == formats::RGB888) {
            cv::Mat rgb(frameHeight_, frameWidth_, CV_8UC3, info.data, frameStride_);
            cv::cvtColor(rgb, gray, cv::COLOR_RGB2GRAY);
        } else if (pixelFormat_ == formats::BGR888) {
            cv::Mat bgr(frameHeight_, frameWidth_, CV_8UC3, info.data, frameStride_);
            cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
        } else {
            // Unsupported format - try to use as grayscale anyway
            cerr << "Warning: Unknown pixel format, attempting raw grayscale interpretation" << endl;
            gray = cv::Mat(frameHeight_, frameWidth_, CV_8UC1, info.data, frameStride_);
        }

        {
            lock_guard<mutex> lock(frameMtx_);
            gray.copyTo(currentFrame_);
            currentTimestamp_ = timestamp;
            frameReady_ = true;
        }
        frameCond_.notify_one();

        // Requeue request
        request->reuse(Request::ReuseBuffers);
        camera_->queueRequest(request);
    }
 
    unique_ptr<CameraManager> cameraManager_;
    shared_ptr<Camera> camera_;
    unique_ptr<CameraConfiguration> config_;
    unique_ptr<FrameBufferAllocator> allocator_;
    vector<unique_ptr<Request>> requests_;
    Stream* stream_ = nullptr;
    PixelFormat pixelFormat_;
 
    struct BufferInfo {
        uint8_t* data = nullptr;
        size_t size = 0;
    };
    map<FrameBuffer*, BufferInfo> mappedBuffers_;
 
    int frameWidth_ = 0;
    int frameHeight_ = 0;
    int frameStride_ = 0;
 
    atomic<bool> running_{false};
 
    mutex frameMtx_;
    condition_variable frameCond_;
    cv::Mat currentFrame_;
    double currentTimestamp_ = 0;
    bool frameReady_ = false;
 
    // Camera settings loaded from INI file
    CameraSettings settings_;
};
 
// ============================================================================
// Main
// ============================================================================
 
int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        cerr << endl
             << "Usage: ./mono_inertial_custom_cam path_to_vocabulary path_to_settings [serial_port]"
             << endl
             << "  Default serial_port: /dev/ttyACM0"
             << endl;
        return 1;
    }
 
    string vocabularyPath = argv[1];
    string settingsPath = argv[2];
    string serialPort = (argc == 4) ? argv[3] : "/dev/ttyACM0";
 
    // Setup signal handler
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = exit_loop_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);
 
    cout << "========================================" << endl;
    cout << "Monocular-Inertial VIO (Custom Camera)" << endl;
    cout << "========================================" << endl;
    cout << "Vocabulary: " << vocabularyPath << endl;
    cout << "Settings:   " << settingsPath << endl;
    cout << "IMU Port:   " << serialPort << endl;
    cout << "========================================" << endl;
 
    // Initialize IMU reader
    SerialIMUReader imuReader;
    if (!imuReader.open(serialPort)) {
        cerr << "Failed to open IMU serial port: " << serialPort << endl;
        cerr << "Check that:" << endl;
        cerr << "  - Pico is connected and running" << endl;
        cerr << "  - User has dialout group access: sudo usermod -aG dialout $USER" << endl;
        return 1;
    }
 
    // Initialize camera
    CameraController camera;
    if (!camera.initialize(0)) {
        cerr << "Failed to initialize camera" << endl;
        return 1;
    }
 
    // Load camera settings from viewer_app config (if available)
    camera.loadSettings();
 
    if (!camera.start()) {
        cerr << "Failed to start camera" << endl;
        return 1;
    }
 
    // Wait for first IMU data to establish time base
    cout << "Waiting for IMU data..." << endl;
    uint64_t firstImuTimestamp = 0;
    while (b_continue_session && firstImuTimestamp == 0) {
        auto imuData = imuReader.getIMUData();
        if (!imuData.empty()) {
            firstImuTimestamp = imuData.front().timestamp_ms;
            cout << "First IMU timestamp: " << firstImuTimestamp << " ms" << endl;

            // Print first IMU reading to verify values (gravity should be ~9.8 m/s² on one axis)
            const auto& first = imuData.front();
            cout << "First IMU reading (raw):" << endl;
            cout << "  Accel: [" << first.accel_x << ", " << first.accel_y << ", " << first.accel_z << "] g" << endl;
            cout << "  Gyro:  [" << first.gyro_x << ", " << first.gyro_y << ", " << first.gyro_z << "] deg/s" << endl;
            cout << "  Accel (SI): [" << first.accel_x * 9.80665f << ", " << first.accel_y * 9.80665f
                 << ", " << first.accel_z * 9.80665f << "] m/s²" << endl;
        }
        this_thread::sleep_for(chrono::milliseconds(10));
    }
 
    if (!b_continue_session) {
        cout << "Interrupted before initialization" << endl;
        return 0;
    }
 
    // Create SLAM system
    cout << "Creating ORB-SLAM3 system..." << endl;
    ORB_SLAM3::System SLAM(vocabularyPath, settingsPath, ORB_SLAM3::System::IMU_MONOCULAR, true);
    float imageScale = SLAM.GetImageScale();

    // Clear IMU buffer accumulated during SLAM initialization (loading vocabulary takes time)
    // and reset time base to avoid huge initial backlog
    cout << "Clearing IMU buffer accumulated during initialization..." << endl;
    imuReader.getIMUData();  // Discard accumulated data
    imuReader.getCameraTriggerTimestamp();  // Discard any pending camera trigger

    // Wait for fresh IMU data to establish new time base
    firstImuTimestamp = 0;
    while (b_continue_session && firstImuTimestamp == 0) {
        auto imuData = imuReader.getIMUData();
        if (!imuData.empty()) {
            firstImuTimestamp = imuData.back().timestamp_ms;
            cout << "Reset time base to: " << firstImuTimestamp << " ms" << endl;
        }
        this_thread::sleep_for(chrono::milliseconds(5));
    }

    // Discard first few camera frames to ensure they have timestamps after our time base
    cout << "Waiting for camera frames with valid timestamps..." << endl;
    for (int i = 0; i < 5 && b_continue_session; i++) {
        cv::Mat frame;
        double ts;
        camera.getFrame(frame, ts);
        imuReader.getCameraTriggerTimestamp();  // Discard trigger
        imuReader.getIMUData();  // Discard IMU
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

    // Main loop
    while (!SLAM.isShutDown() && b_continue_session) {
        // Get camera frame FIRST (this blocks until frame is ready)
        cv::Mat frame;
        double cameraTimestamp;

        if (!camera.getFrame(frame, cameraTimestamp)) {
            continue;
        }

        // Get Pico trigger timestamp
        uint64_t triggerTimestamp = imuReader.getCameraTriggerTimestamp();

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
 
        for (const auto& imu : imuData) {
            // Convert to SI units and create IMU::Point
            double t = (imu.timestamp_ms - firstImuTimestamp) / 1000.0;

            // Check for gaps in IMU stream (should be ~5ms at 200Hz)
            if (lastImuTimestamp > 0 && t > lastImuTimestamp) {
                double gap = t - lastImuTimestamp;
                if (gap > 0.010 && imuGapWarnings < 10) {  // More than 10ms gap
                    cout << "WARNING: IMU gap detected: " << fixed << setprecision(1)
                         << gap * 1000 << "ms at t=" << setprecision(3) << t << endl;
                    imuGapWarnings++;
                }
            }
            lastImuTimestamp = t;
 
            // Convert: gyro from deg/s to rad/s, accel from g to m/s²
            ORB_SLAM3::IMU::Point pt(
                imu.accel_x * G_TO_MS2,
                imu.accel_y * G_TO_MS2,
                imu.accel_z * G_TO_MS2,
                imu.gyro_x * DEG_TO_RAD,
                imu.gyro_y * DEG_TO_RAD,
                imu.gyro_z * DEG_TO_RAD,
                t
            );
            vImuMeas.push_back(pt);
            imuCount++;
        }

        // Determine frame time
        double frameTime;
        if (triggerTimestamp > 0 && triggerTimestamp >= firstImuTimestamp) {
            // Use Pico timestamp (relative to first IMU timestamp)
            frameTime = (triggerTimestamp - firstImuTimestamp) / 1000.0;
        } else {
            // No trigger - fall back to last IMU timestamp
            if (!vImuMeas.empty()) {
                frameTime = vImuMeas.back().t;
                if (frameCount < 20) {
                    cout << "No trigger, using last IMU time: " << fixed << setprecision(3) << frameTime << endl;
                }
            } else {
                // No IMU data yet, skip frame
                if (frameCount < 20) {
                    cout << "Skipping frame: no trigger and no IMU data" << endl;
                }
                continue;
            }
        }

        // Filter IMU measurements: only keep those with timestamp <= frameTime
        // Keep measurements after frameTime for next iteration
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
                // Show gap between last IMU and frame time
                double imuFrameGap = frameTime - vImuForFrame.back().t;
                if (imuFrameGap > 0.005) {  // More than 5ms
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

        // Track with filtered IMU data
        SLAM.TrackMonocular(frame, frameTime, vImuForFrame);
 
        frameCount++;
 
        // Print statistics every 100 frames
        if (frameCount % 100 == 0) {
            auto now = chrono::steady_clock::now();
            double elapsed = chrono::duration<double>(now - startTime).count();
            double fps = frameCount / elapsed;
            cout << "Frames: " << frameCount
                 << " | IMU: " << imuCount
                 << " | FPS: " << fixed << setprecision(1) << fps
                 << " | Time: " << fixed << setprecision(1) << elapsed << "s"
                 << endl;
        }
    }
 
    // Cleanup
    cout << endl << "Shutting down..." << endl;
 
    camera.stop();
    imuReader.close();
 
    SLAM.Shutdown();
 
    // Final statistics
    auto endTime = chrono::steady_clock::now();
    double totalTime = chrono::duration<double>(endTime - startTime).count();
 
    cout << "========================================" << endl;
    cout << "Session complete" << endl;
    cout << "Total frames:  " << frameCount << endl;
    cout << "Total IMU:     " << imuCount << endl;
    cout << "Duration:      " << fixed << setprecision(1) << totalTime << " s" << endl;
    cout << "Average FPS:   " << fixed << setprecision(1) << (frameCount / totalTime) << endl;
    cout << "========================================" << endl;
 
    // Save trajectory
    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
    cout << "Trajectories saved to CameraTrajectory.txt and KeyFrameTrajectory.txt" << endl;
 
    return 0;
}
 
