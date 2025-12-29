/**
 * Pear VIO Camera API - Camera Controller Implementation
 */

#include "pear/camera.h"

#include <iostream>
#include <map>
#include <vector>

#include <libcamera/libcamera.h>
#include <libcamera/control_ids.h>

#include <opencv2/imgproc/imgproc.hpp>

#include <sys/mman.h>
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <cerrno>

using namespace std;
using namespace libcamera;

namespace pear {

// Implementation details (pimpl pattern)
struct CameraController::Impl {
    unique_ptr<CameraManager> cameraManager;
    shared_ptr<Camera> camera;
    unique_ptr<CameraConfiguration> config;
    unique_ptr<FrameBufferAllocator> allocator;
    vector<unique_ptr<Request>> requests;
    Stream* stream = nullptr;
    PixelFormat pixelFormat;

    struct BufferInfo {
        uint8_t* data = nullptr;
        size_t size = 0;
    };
    map<FrameBuffer*, BufferInfo> mappedBuffers;
};

CameraController::CameraController() : impl_(make_unique<Impl>()) {}

CameraController::~CameraController() {
    stop();
}

bool CameraController::initialize(const CameraConfig& config) {
    config_ = config;

    impl_->cameraManager = make_unique<CameraManager>();
    int ret = impl_->cameraManager->start();
    if (ret < 0) {
        cerr << "Failed to start camera manager: " << strerror(-ret) << endl;
        return false;
    }

    auto cameras = impl_->cameraManager->cameras();
    if (cameras.empty()) {
        cerr << "No cameras found" << endl;
        return false;
    }

    if (config_.cameraIndex >= static_cast<int>(cameras.size())) {
        cerr << "Camera index " << config_.cameraIndex << " out of range" << endl;
        return false;
    }

    impl_->camera = cameras[config_.cameraIndex];
    ret = impl_->camera->acquire();
    if (ret < 0) {
        cerr << "Failed to acquire camera: " << strerror(-ret) << endl;
        return false;
    }

    cout << "Acquired camera: " << impl_->camera->id() << endl;

    // Configure for grayscale
    impl_->config = impl_->camera->generateConfiguration({StreamRole::Viewfinder});
    if (!impl_->config) {
        cerr << "Failed to generate camera configuration" << endl;
        return false;
    }

    StreamConfiguration& streamConfig = impl_->config->at(0);
    streamConfig.size = Size(config_.width, config_.height);
    streamConfig.pixelFormat = formats::R8;

    auto status = impl_->config->validate();
    if (status == CameraConfiguration::Invalid) {
        cerr << "Invalid camera configuration" << endl;
        return false;
    }

    if (status == CameraConfiguration::Adjusted) {
        cout << "Camera configuration adjusted to: "
             << streamConfig.size.width << "x" << streamConfig.size.height
             << " " << streamConfig.pixelFormat.toString() << endl;
    }

    ret = impl_->camera->configure(impl_->config.get());
    if (ret < 0) {
        cerr << "Failed to configure camera: " << strerror(-ret) << endl;
        return false;
    }

    frameWidth_ = streamConfig.size.width;
    frameHeight_ = streamConfig.size.height;
    frameStride_ = streamConfig.stride;
    impl_->pixelFormat = streamConfig.pixelFormat;
    impl_->stream = streamConfig.stream();

    // Allocate buffers
    impl_->allocator = make_unique<FrameBufferAllocator>(impl_->camera);
    ret = impl_->allocator->allocate(impl_->stream);
    if (ret < 0) {
        cerr << "Failed to allocate buffers: " << strerror(-ret) << endl;
        return false;
    }

    cout << "Allocated " << impl_->allocator->buffers(impl_->stream).size() << " buffers" << endl;

    // Map buffers
    for (const auto& buffer : impl_->allocator->buffers(impl_->stream)) {
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
            impl_->mappedBuffers[buffer.get()] = {static_cast<uint8_t*>(mem), length};
        }
    }

    // Create requests
    for (const auto& buffer : impl_->allocator->buffers(impl_->stream)) {
        auto request = impl_->camera->createRequest();
        if (!request) {
            cerr << "Failed to create request" << endl;
            return false;
        }
        request->addBuffer(impl_->stream, buffer.get());
        impl_->requests.push_back(move(request));
    }

    // Connect callback
    impl_->camera->requestCompleted.connect(this, &CameraController::requestComplete);

    cout << "Camera initialized: " << frameWidth_ << "x" << frameHeight_
         << " stride: " << frameStride_
         << " format: " << impl_->pixelFormat.toString() << endl;

    // Check format support
    if (impl_->pixelFormat == formats::R8) {
        cout << "  -> Using R8 (8-bit grayscale) - optimal" << endl;
    } else if (impl_->pixelFormat == formats::R16) {
        cout << "  -> Using R16 (16-bit grayscale) - will convert 10-bit to 8-bit" << endl;
    } else if (impl_->pixelFormat == formats::XRGB8888 || impl_->pixelFormat == formats::XBGR8888) {
        cout << "  -> Using XRGB8888 (32-bit) - will extract grayscale channel" << endl;
    } else {
        cout << "  -> Warning: Format " << impl_->pixelFormat.toString()
             << " may not be fully supported" << endl;
    }

    return true;
}

bool CameraController::initialize(int cameraIndex) {
    CameraConfig config;
    config.cameraIndex = cameraIndex;
    return initialize(config);
}

bool CameraController::loadSettings(const string& configPath) {
    string path = configPath.empty() ? CameraConfig::getDefaultConfigPath() : configPath;

    if (path.empty()) {
        cout << "No config path available, using default camera settings" << endl;
        return false;
    }

    if (config_.loadFromIniFile(path)) {
        cout << "Loaded camera settings from: " << path << endl;
        cout << "  Auto exposure: " << (config_.autoExposure ? "enabled" : "disabled") << endl;
        if (!config_.autoExposure) {
            cout << "  Manual exposure: " << config_.exposureTimeUs << " us" << endl;
            cout << "  Manual gain: " << config_.gain << endl;
        }
        return true;
    } else {
        cout << "Config file not found: " << path << endl;
        cout << "Using default camera settings (auto exposure)" << endl;
        return false;
    }
}

bool CameraController::start() {
    if (!impl_->camera) return false;
    if (running_) return true;

    // Set frame duration for consistent FPS
    ControlList controls;
    controls.set(controls::FrameDurationLimits,
                 Span<const int64_t, 2>({config_.frameDurationUs, config_.frameDurationUs}));

    // Apply exposure settings
    if (config_.autoExposure) {
        controls.set(controls::AeEnable, true);
        cout << "Camera using auto exposure" << endl;
    } else {
        controls.set(controls::AeEnable, false);
        controls.set(controls::ExposureTime, config_.exposureTimeUs);
        controls.set(controls::AnalogueGain, config_.gain);
        cout << "Camera using manual exposure: " << config_.exposureTimeUs
             << " us, gain: " << config_.gain << endl;
    }

    int ret = impl_->camera->start(&controls);
    if (ret < 0) {
        cerr << "Failed to start camera: " << strerror(-ret) << endl;
        return false;
    }

    running_ = true;

    // Queue all requests
    for (auto& request : impl_->requests) {
        impl_->camera->queueRequest(request.get());
    }

    cout << "Camera started" << endl;
    return true;
}

void CameraController::stop() {
    if (!running_) return;
    running_ = false;

    if (impl_->camera) {
        impl_->camera->stop();
    }

    // Cleanup mapped buffers
    for (auto& [buffer, info] : impl_->mappedBuffers) {
        if (info.data) {
            munmap(info.data, info.size);
        }
    }
    impl_->mappedBuffers.clear();

    if (impl_->camera) {
        impl_->camera->release();
        impl_->camera.reset();
    }
    if (impl_->cameraManager) {
        impl_->cameraManager->stop();
        impl_->cameraManager.reset();
    }

    // Notify any waiting threads
    frameCond_.notify_all();

    cout << "Camera stopped" << endl;
}

bool CameraController::getFrame(cv::Mat& frame, double& timestamp) {
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

void CameraController::requestComplete(Request* request) {
    if (!running_) return;
    if (request->status() == Request::RequestCancelled) return;

    // Get buffer
    FrameBuffer* buffer = request->buffers().begin()->second;
    auto it = impl_->mappedBuffers.find(buffer);
    if (it == impl_->mappedBuffers.end()) return;

    const auto& info = it->second;

    // Get timestamp (SensorTimestamp is in nanoseconds)
    auto tsOpt = request->metadata().get(controls::SensorTimestamp);
    double timestamp = tsOpt.value_or(0) / 1e9;  // Convert to seconds

    // Convert to cv::Mat based on pixel format
    cv::Mat gray;

    if (impl_->pixelFormat == formats::R8) {
        // Direct 8-bit grayscale
        gray = cv::Mat(frameHeight_, frameWidth_, CV_8UC1, info.data, frameStride_);
    } else if (impl_->pixelFormat == formats::R16) {
        // 16-bit grayscale - OV9281 outputs 10-bit packed into 16-bit
        gray = cv::Mat(frameHeight_, frameWidth_, CV_8UC1);
        const uint16_t* src = reinterpret_cast<const uint16_t*>(info.data);
        int srcStride = frameStride_ / 2;

        for (int y = 0; y < frameHeight_; y++) {
            const uint16_t* srcLine = src + y * srcStride;
            uint8_t* dstLine = gray.ptr<uint8_t>(y);
            for (int x = 0; x < frameWidth_; x++) {
                dstLine[x] = static_cast<uint8_t>(srcLine[x] >> 8);
            }
        }
    } else if (impl_->pixelFormat == formats::XRGB8888 || impl_->pixelFormat == formats::XBGR8888) {
        // 32-bit XRGB - for mono camera, all channels are same
        gray = cv::Mat(frameHeight_, frameWidth_, CV_8UC1);
        for (int y = 0; y < frameHeight_; y++) {
            const uint8_t* srcLine = info.data + y * frameStride_;
            uint8_t* dstLine = gray.ptr<uint8_t>(y);
            for (int x = 0; x < frameWidth_; x++) {
                dstLine[x] = srcLine[x * 4];
            }
        }
    } else if (impl_->pixelFormat == formats::RGB888) {
        cv::Mat rgb(frameHeight_, frameWidth_, CV_8UC3, info.data, frameStride_);
        cv::cvtColor(rgb, gray, cv::COLOR_RGB2GRAY);
    } else if (impl_->pixelFormat == formats::BGR888) {
        cv::Mat bgr(frameHeight_, frameWidth_, CV_8UC3, info.data, frameStride_);
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    } else {
        // Unsupported format - try raw grayscale
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
    impl_->camera->queueRequest(request);
}

// ============================================================================
// OV9281 I2C Trigger Mode Control
// ============================================================================

namespace {

// Write a single register to OV9281 via I2C
int ov9281WriteReg(int fd, uint16_t reg, uint8_t val) {
    uint8_t buf[3] = {
        static_cast<uint8_t>((reg >> 8) & 0xFF),
        static_cast<uint8_t>(reg & 0xFF),
        val
    };
    if (write(fd, buf, 3) != 3) {
        cerr << "I2C write failed for reg 0x" << hex << reg
             << ": " << strerror(errno) << dec << endl;
        return -1;
    }
    return 0;
}

}  // anonymous namespace

bool CameraController::setTriggerMode(bool enable) {
    char devPath[32];
    snprintf(devPath, sizeof(devPath), "/dev/i2c-%d", config_.i2cBus);

    int fd = open(devPath, O_RDWR);
    if (fd < 0) {
        cerr << "Failed to open I2C bus " << devPath << ": " << strerror(errno) << endl;
        return false;
    }

    if (ioctl(fd, I2C_SLAVE_FORCE, config_.i2cAddress) < 0) {
        cerr << "Failed to set I2C slave address 0x" << hex << config_.i2cAddress
             << ": " << strerror(errno) << dec << endl;
        close(fd);
        return false;
    }

    int ret = 0;
    if (enable) {
        // Full initialization sequence for trigger mode
        // These registers configure OV9281 to wait for external trigger
        ret |= ov9281WriteReg(fd, 0x3006, 0x0C);  // Timing control
        ret |= ov9281WriteReg(fd, 0x3027, 0x00);
        ret |= ov9281WriteReg(fd, 0x4F00, 0x01);  // Enable trigger
        ret |= ov9281WriteReg(fd, 0x3030, 0x04);
        ret |= ov9281WriteReg(fd, 0x303F, 0x01);
        ret |= ov9281WriteReg(fd, 0x302C, 0x00);
        ret |= ov9281WriteReg(fd, 0x302F, 0x7F);
        ret |= ov9281WriteReg(fd, 0x3023, 0x00);
        ret |= ov9281WriteReg(fd, 0x0100, 0x00);  // Standby (wait for trigger)

        if (ret == 0) {
            cout << "OV9281 trigger mode ENABLED on I2C bus " << config_.i2cBus << endl;
            triggerModeEnabled_ = true;
        }
    } else {
        // Simplified sequence for free-run mode
        ret |= ov9281WriteReg(fd, 0x3006, 0x04);  // Normal timing
        ret |= ov9281WriteReg(fd, 0x4F00, 0x00);  // Disable trigger
        ret |= ov9281WriteReg(fd, 0x0100, 0x01);  // Start streaming

        if (ret == 0) {
            cout << "OV9281 trigger mode disabled" << endl;
            triggerModeEnabled_ = false;
        }
    }

    close(fd);

    if (ret != 0) {
        cerr << "Warning: Some I2C register writes failed" << endl;
        return false;
    }

    return true;
}

}  // namespace pear
