// Undefine Qt macros BEFORE any includes to avoid conflicts with libcamera
#ifdef signals
#undef signals
#endif
#ifdef slots
#undef slots
#endif
#ifdef emit
#undef emit
#endif

#include <libcamera/libcamera.h>
#include <libcamera/control_ids.h>

#include "CameraController.h"

#include <QDebug>
#include <QMutex>
#include <sys/mman.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <map>
#include <vector>
#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>

using namespace libcamera;

// ============================================================================
// OV9281 I2C Trigger Mode Control
// ============================================================================
namespace {

constexpr uint8_t OV9281_I2C_ADDR = 0x60;

int ov9281_write_reg(int fd, uint16_t reg, uint8_t val)
{
    uint8_t buf[3] = {
        static_cast<uint8_t>((reg >> 8) & 0xFF),
        static_cast<uint8_t>(reg & 0xFF),
        val
    };
    if (write(fd, buf, 3) != 3) {
        qWarning() << "I2C write failed for register" << Qt::hex << reg;
        return -1;
    }
    return 0;
}

int ov9281_set_trigger_mode(int bus, bool enable)
{
    char dev_path[20];
    snprintf(dev_path, sizeof(dev_path), "/dev/i2c-%d", bus);

    int fd = open(dev_path, O_RDWR);
    if (fd < 0) {
        qWarning() << "Failed to open I2C bus:" << dev_path;
        return -1;
    }

    if (ioctl(fd, I2C_SLAVE_FORCE, OV9281_I2C_ADDR) < 0) {
        qWarning() << "Failed to set I2C slave address 0x60";
        close(fd);
        return -1;
    }

    int ret = 0;
    if (enable) {
        // Full initialization sequence for trigger mode
        ret |= ov9281_write_reg(fd, 0x3006, 0x0C);  // Timing control
        ret |= ov9281_write_reg(fd, 0x3027, 0x00);
        ret |= ov9281_write_reg(fd, 0x4F00, 0x01);  // Enable trigger
        ret |= ov9281_write_reg(fd, 0x3030, 0x04);
        ret |= ov9281_write_reg(fd, 0x303F, 0x01);
        ret |= ov9281_write_reg(fd, 0x302C, 0x00);
        ret |= ov9281_write_reg(fd, 0x302F, 0x7F);
        ret |= ov9281_write_reg(fd, 0x3023, 0x00);
        ret |= ov9281_write_reg(fd, 0x0100, 0x00);  // Standby - wait for trigger
    } else {
        // Simplified sequence for free-run mode
        ret |= ov9281_write_reg(fd, 0x3006, 0x04);  // Normal timing
        ret |= ov9281_write_reg(fd, 0x4F00, 0x00);  // Disable trigger
        ret |= ov9281_write_reg(fd, 0x0100, 0x01);  // Start streaming
    }

    close(fd);
    return ret;
}

// Try common I2C buses for RPi camera
int ov9281_set_trigger_mode_auto(bool enable)
{
    // RPi 5 typically uses bus 0 or 10 for CSI cameras
    const int buses[] = {0, 10, 1, 4};

    for (int bus : buses) {
        char dev_path[20];
        snprintf(dev_path, sizeof(dev_path), "/dev/i2c-%d", bus);

        // Check if bus exists
        if (access(dev_path, F_OK) != 0) {
            continue;
        }

        int ret = ov9281_set_trigger_mode(bus, enable);
        if (ret == 0) {
            qDebug() << "OV9281 trigger mode" << (enable ? "enabled" : "disabled")
                     << "via I2C bus" << bus;
            return 0;
        }
    }

    qWarning() << "Failed to set OV9281 trigger mode on any I2C bus";
    return -1;
}

} // anonymous namespace

// ============================================================================
// PIMPL Implementation - contains all libcamera code
// ============================================================================
class CameraController::Impl
{
public:
    explicit Impl(CameraController *parent) : parent_(parent) {}
    ~Impl();
    
    bool initialize(int cameraIndex);
    bool start();
    void stop();
    
    bool isRunning() const { return running_; }
    uint64_t frameCount() const { return frameCount_; }
    double currentFps() const { return currentFps_; }
    QString cameraId() const;
    QSize frameSize() const { return frameSize_; }

    void setTriggerMode(bool enabled);
    bool triggerMode() const { return triggerMode_; }

    void setAutoExposure(bool enabled);
    bool autoExposure() const { return autoExposure_; }
    void setExposureTime(int microseconds);
    int exposureTime() const { return exposureTime_; }

    void setGain(float gain);
    float gain() const { return gain_; }

private:
    void requestComplete(Request *request);
    void processFrame(FrameBuffer *buffer, Request *request);
    QImage convertToQImage(const uint8_t *data, size_t size,
                           const StreamConfiguration &config);
    void applyExposureControls(Request *request);

    CameraController *parent_;

    // libcamera objects
    std::unique_ptr<CameraManager> cameraManager_;
    std::shared_ptr<Camera> camera_;
    std::unique_ptr<CameraConfiguration> config_;
    std::unique_ptr<FrameBufferAllocator> allocator_;
    std::vector<std::unique_ptr<Request>> requests_;
    Stream *stream_ = nullptr;

    // Buffer info struct to replace Span usage
    struct BufferPlaneInfo {
        uint8_t *data;
        size_t size;
    };
    std::map<FrameBuffer *, std::vector<BufferPlaneInfo>> mappedBuffers_;

    // State
    std::atomic<bool> running_{false};
    std::atomic<bool> triggerMode_{false};
    std::atomic<bool> autoExposure_{true};
    std::atomic<int> exposureTime_{10000};  // Default 10ms
    std::atomic<int> currentExposure_{0};
    std::atomic<float> gain_{8.0f};  // Default gain for manual mode
    QSize frameSize_;

    // Statistics
    std::atomic<uint64_t> frameCount_{0};
    std::atomic<double> currentFps_{0.0};
    std::chrono::steady_clock::time_point lastFpsUpdate_;
    uint64_t lastFrameCount_ = 0;

    QMutex mutex_;
};

CameraController::Impl::~Impl()
{
    stop();
    
    // Clean up mapped buffers
    for (auto &[buffer, planes] : mappedBuffers_) {
        for (auto &plane : planes) {
            if (plane.data) {
                munmap(plane.data, plane.size);
            }
        }
    }
    mappedBuffers_.clear();
    
    // Release camera resources
    if (camera_) {
        camera_->release();
        camera_.reset();
    }
    
    if (cameraManager_) {
        cameraManager_->stop();
        cameraManager_.reset();
    }
}

bool CameraController::Impl::initialize(int cameraIndex)
{
    QMutexLocker locker(&mutex_);
    
    // Create and start camera manager (like rpicam-hello)
    cameraManager_ = std::make_unique<CameraManager>();
    int ret = cameraManager_->start();
    if (ret < 0) {
        QMetaObject::invokeMethod(parent_, "errorOccurred", Qt::QueuedConnection,
            Q_ARG(QString, QString("Failed to start camera manager: %1").arg(strerror(-ret))));
        return false;
    }
    
    // Get available cameras
    auto cameras = cameraManager_->cameras();
    if (cameras.empty()) {
        QMetaObject::invokeMethod(parent_, "errorOccurred", Qt::QueuedConnection,
            Q_ARG(QString, QString("No cameras found")));
        return false;
    }
    
    if (cameraIndex >= static_cast<int>(cameras.size())) {
        QMetaObject::invokeMethod(parent_, "errorOccurred", Qt::QueuedConnection,
            Q_ARG(QString, QString("Camera index %1 out of range (found %2 cameras)")
                          .arg(cameraIndex).arg(cameras.size())));
        return false;
    }
    
    // Acquire camera
    camera_ = cameras[cameraIndex];
    ret = camera_->acquire();
    if (ret < 0) {
        QMetaObject::invokeMethod(parent_, "errorOccurred", Qt::QueuedConnection,
            Q_ARG(QString, QString("Failed to acquire camera: %1").arg(strerror(-ret))));
        return false;
    }
    
    qDebug() << "Acquired camera:" << QString::fromStdString(camera_->id());
    
    // Configure camera for video/viewfinder (similar to rpicam-hello approach)
    config_ = camera_->generateConfiguration({StreamRole::Viewfinder});
    if (!config_) {
        QMetaObject::invokeMethod(parent_, "errorOccurred", Qt::QueuedConnection,
            Q_ARG(QString, QString("Failed to generate camera configuration")));
        return false;
    }
    
    // Configure stream for OV9281 (monochrome global shutter camera)
    StreamConfiguration &streamConfig = config_->at(0);
    // Camera resolution (640x400 for lower CPU usage)
    streamConfig.size = Size(640, 400);
    // Try to request R8 (8-bit grayscale) format for mono sensor
    // This avoids expensive XRGB8888 -> Grayscale conversion
    streamConfig.pixelFormat = formats::R8;

    qDebug() << "Requesting format: R8 (8-bit grayscale)";

    // Validate and apply configuration
    CameraConfiguration::Status status = config_->validate();
    if (status == CameraConfiguration::Invalid) {
        QMetaObject::invokeMethod(parent_, "errorOccurred", Qt::QueuedConnection,
            Q_ARG(QString, QString("Invalid camera configuration")));
        return false;
    }
    
    if (status == CameraConfiguration::Adjusted) {
        qDebug() << "Camera configuration adjusted by libcamera";
        qDebug() << "  Actual format:" << QString::fromStdString(streamConfig.pixelFormat.toString());
        qDebug() << "  Actual size:" << streamConfig.size.width << "x" << streamConfig.size.height;
    }
    
    ret = camera_->configure(config_.get());
    if (ret < 0) {
        QMetaObject::invokeMethod(parent_, "errorOccurred", Qt::QueuedConnection,
            Q_ARG(QString, QString("Failed to configure camera: %1").arg(strerror(-ret))));
        return false;
    }
    
    // Store frame size
    frameSize_ = QSize(streamConfig.size.width, streamConfig.size.height);
    stream_ = streamConfig.stream();
    
    qDebug() << "Camera configured:" 
             << frameSize_.width() << "x" << frameSize_.height()
             << QString::fromStdString(streamConfig.pixelFormat.toString());
    
    // Allocate frame buffers (like rpicam-hello)
    allocator_ = std::make_unique<FrameBufferAllocator>(camera_);
    ret = allocator_->allocate(stream_);
    if (ret < 0) {
        QMetaObject::invokeMethod(parent_, "errorOccurred", Qt::QueuedConnection,
            Q_ARG(QString, QString("Failed to allocate buffers: %1").arg(strerror(-ret))));
        return false;
    }
    
    qDebug() << "Allocated" << allocator_->buffers(stream_).size() << "buffers";
    
    // Memory map the buffers for CPU access
    for (const std::unique_ptr<FrameBuffer> &buffer : allocator_->buffers(stream_)) {
        std::vector<BufferPlaneInfo> planeInfos;
        
        for (const FrameBuffer::Plane &plane : buffer->planes()) {
            // Get the file descriptor size
            size_t length = plane.length;
            
            // If length is 0, get actual size from fd
            if (length == 0) {
                off_t fdSize = lseek(plane.fd.get(), 0, SEEK_END);
                if (fdSize < 0) {
                    QMetaObject::invokeMethod(parent_, "errorOccurred", Qt::QueuedConnection,
                        Q_ARG(QString, QString("Failed to get buffer size: %1").arg(strerror(errno))));
                    return false;
                }
                length = static_cast<size_t>(fdSize);
                lseek(plane.fd.get(), 0, SEEK_SET);  // Reset position
            }
            
            if (length == 0) {
                qWarning() << "Buffer plane has zero length, skipping";
                continue;
            }
            
            void *mem = mmap(nullptr, length, PROT_READ, MAP_SHARED,
                            plane.fd.get(), 0);  // Use offset 0, plane.offset is for DMA
            if (mem == MAP_FAILED) {
                QMetaObject::invokeMethod(parent_, "errorOccurred", Qt::QueuedConnection,
                    Q_ARG(QString, QString("Failed to mmap buffer (size=%1): %2")
                        .arg(length).arg(strerror(errno))));
                return false;
            }
            
            qDebug() << "Mapped buffer plane:" << length << "bytes at" << mem;
            planeInfos.push_back({static_cast<uint8_t*>(mem), length});
        }
        
        if (!planeInfos.empty()) {
            mappedBuffers_[buffer.get()] = std::move(planeInfos);
        }
    }
    
    // Create requests (one per buffer)
    for (const std::unique_ptr<FrameBuffer> &buffer : allocator_->buffers(stream_)) {
        auto request = camera_->createRequest();
        if (!request) {
            QMetaObject::invokeMethod(parent_, "errorOccurred", Qt::QueuedConnection,
                Q_ARG(QString, QString("Failed to create request")));
            return false;
        }
        
        ret = request->addBuffer(stream_, buffer.get());
        if (ret < 0) {
            QMetaObject::invokeMethod(parent_, "errorOccurred", Qt::QueuedConnection,
                Q_ARG(QString, QString("Failed to add buffer to request: %1").arg(strerror(-ret))));
            return false;
        }
        
        requests_.push_back(std::move(request));
    }
    
    // Connect request completed signal
    camera_->requestCompleted.connect(this, &Impl::requestComplete);
    
    return true;
}

bool CameraController::Impl::start()
{
    QMutexLocker locker(&mutex_);

    if (!camera_) {
        QMetaObject::invokeMethod(parent_, "errorOccurred", Qt::QueuedConnection,
            Q_ARG(QString, QString("Camera not initialized")));
        return false;
    }

    if (running_) {
        return true;
    }

    // Reset statistics
    frameCount_ = 0;
    lastFrameCount_ = 0;
    currentFps_ = 0.0;
    lastFpsUpdate_ = std::chrono::steady_clock::now();

    // Set camera controls to maintain consistent frame rate
    // FrameDurationLimits caps max exposure time so FPS doesn't drop in low light
    ControlList controls;

    // Set frame duration limits: min=33333us, max=33333us (exactly 30 FPS)
    // This prevents auto-exposure from increasing exposure beyond 1/30s
    // The camera will use gain instead for low-light compensation
    int64_t frameDurationMin = 33333;  // 30 FPS in microseconds
    int64_t frameDurationMax = 33333;
    controls.set(controls::FrameDurationLimits,
                 Span<const int64_t, 2>({frameDurationMin, frameDurationMax}));

    // Set initial exposure controls - this ensures saved settings are applied from first frame
    if (autoExposure_) {
        controls.set(controls::AeEnable, true);
        qDebug() << "Starting with auto exposure enabled";
    } else {
        // Manual exposure mode - must set BOTH exposure time AND gain
        controls.set(controls::AeEnable, false);
        controls.set(controls::ExposureTime, exposureTime_.load());
        controls.set(controls::AnalogueGain, gain_.load());
        qDebug() << "Starting with manual exposure:" << exposureTime_.load() << "us, gain:" << gain_.load();
    }

    qDebug() << "Setting FrameDurationLimits:" << frameDurationMin << "-" << frameDurationMax << "us (30 FPS)";

    // Start the camera with controls
    int ret = camera_->start(&controls);
    if (ret < 0) {
        QMetaObject::invokeMethod(parent_, "errorOccurred", Qt::QueuedConnection,
            Q_ARG(QString, QString("Failed to start camera: %1").arg(strerror(-ret))));
        return false;
    }

    running_ = true;

    // Queue all requests with exposure controls applied
    for (auto &request : requests_) {
        applyExposureControls(request.get());
        ret = camera_->queueRequest(request.get());
        if (ret < 0) {
            QMetaObject::invokeMethod(parent_, "errorOccurred", Qt::QueuedConnection,
                Q_ARG(QString, QString("Failed to queue request: %1").arg(strerror(-ret))));
            stop();
            return false;
        }
    }

    qDebug() << "Camera started";
    return true;
}

void CameraController::Impl::stop()
{
    if (!running_) {
        return;
    }
    
    running_ = false;
    
    if (camera_) {
        camera_->stop();
    }
    
    qDebug() << "Camera stopped";
}

QString CameraController::Impl::cameraId() const
{
    if (camera_) {
        return QString::fromStdString(camera_->id());
    }
    return QString();
}

void CameraController::Impl::setTriggerMode(bool enabled)
{
    triggerMode_ = enabled;

    // Set OV9281 trigger mode via I2C
    int ret = ov9281_set_trigger_mode_auto(enabled);

    if (ret == 0) {
        qDebug() << "Trigger mode" << (enabled ? "enabled" : "disabled") << "successfully";
    } else {
        qWarning() << "Failed to set trigger mode - check I2C permissions (add user to i2c group)";
    }
}

void CameraController::Impl::requestComplete(Request *request)
{
    if (!running_) {
        return;
    }

    if (request->status() == Request::RequestCancelled) {
        return;
    }

    // Get the buffer from completed request
    FrameBuffer *buffer = request->buffers().begin()->second;
    processFrame(buffer, request);

    // Reuse request and apply exposure controls
    request->reuse(Request::ReuseBuffers);
    applyExposureControls(request);
    camera_->queueRequest(request);
}

void CameraController::Impl::applyExposureControls(Request *request)
{
    ControlList &controls = request->controls();

    if (autoExposure_) {
        // Enable auto-exposure (controls both exposure time and gain)
        controls.set(controls::AeEnable, true);
    } else {
        // Manual exposure mode - must set BOTH exposure time AND gain
        // When AeEnable is false, libcamera expects manual control of both
        controls.set(controls::AeEnable, false);
        controls.set(controls::ExposureTime, exposureTime_.load());
        controls.set(controls::AnalogueGain, gain_.load());
    }
}

void CameraController::Impl::setGain(float gain)
{
    // Clamp to reasonable range (1.0 to 16.0)
    gain = std::max(1.0f, std::min(16.0f, gain));
    gain_ = gain;
    qDebug() << "Gain set to" << gain;
}

void CameraController::Impl::setAutoExposure(bool enabled)
{
    autoExposure_ = enabled;
    qDebug() << "Auto exposure" << (enabled ? "enabled" : "disabled");
}

void CameraController::Impl::setExposureTime(int microseconds)
{
    // Clamp to reasonable range (100us to 100ms)
    microseconds = std::max(100, std::min(100000, microseconds));
    exposureTime_ = microseconds;
    qDebug() << "Manual exposure set to" << microseconds << "us";
}

void CameraController::Impl::processFrame(FrameBuffer *buffer, Request *request)
{
    frameCount_++;

    // Read actual exposure time from frame metadata
    const ControlList &metadata = request->metadata();
    if (metadata.contains(controls::ExposureTime.id())) {
        auto expValue = metadata.get(controls::ExposureTime);
        int actualExposure = expValue ? *expValue : 0;
        if (actualExposure != currentExposure_) {
            currentExposure_ = actualExposure;
            QMetaObject::invokeMethod(parent_, "exposureUpdated", Qt::QueuedConnection,
                Q_ARG(int, actualExposure));
        }
    }

    // Update FPS every second
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastFpsUpdate_).count();

    if (elapsed >= 1000) {
        uint64_t frames = frameCount_ - lastFrameCount_;
        currentFps_ = frames * 1000.0 / elapsed;
        lastFrameCount_ = frameCount_;
        lastFpsUpdate_ = now;

        QMetaObject::invokeMethod(parent_, "fpsUpdated", Qt::QueuedConnection,
            Q_ARG(double, currentFps_.load()), Q_ARG(uint64_t, frameCount_.load()));
    }

    // Get mapped memory
    auto it = mappedBuffers_.find(buffer);
    if (it == mappedBuffers_.end()) {
        return;
    }

    const auto &planes = it->second;
    if (planes.empty()) {
        return;
    }
    
    // Convert frame to QImage
    QImage image = convertToQImage(planes[0].data, planes[0].size, config_->at(0));
    
    if (!image.isNull()) {
        QMetaObject::invokeMethod(parent_, "frameReady", Qt::QueuedConnection,
            Q_ARG(QImage, image));
    }
}

QImage CameraController::Impl::convertToQImage(const uint8_t *data, size_t size,
                                                const StreamConfiguration &config)
{
    int width = config.size.width;
    int height = config.size.height;
    int stride = config.stride;
    
    // Handle R8 format (8-bit grayscale)
    if (config.pixelFormat == formats::R8) {
        QImage grayImage(data, width, height, stride, QImage::Format_Grayscale8);
        return grayImage.copy();
    }
    
    // Handle R16 format (16-bit grayscale) - OV9281 outputs 10-bit packed into 16-bit
    if (config.pixelFormat == formats::R16) {
        QImage grayImage(width, height, QImage::Format_Grayscale8);
        const uint16_t *src = reinterpret_cast<const uint16_t*>(data);

        // Stride is in bytes, convert to uint16 elements
        int srcStride = stride / 2;

        for (int y = 0; y < height; y++) {
            const uint16_t *srcLine = src + y * srcStride;
            uint8_t *dstLine = grayImage.scanLine(y);

            // Process 8 pixels at a time for better performance
            int x = 0;
            const int unrollCount = 8;
            const int unrolledWidth = width - (width % unrollCount);

            for (; x < unrolledWidth; x += unrollCount) {
                // OV9281 is 10-bit sensor, data is in upper bits of 16-bit value
                // Shift right by 8 to get 8-bit value (keeps upper 8 bits of 10-bit data)
                dstLine[x + 0] = static_cast<uint8_t>(srcLine[x + 0] >> 8);
                dstLine[x + 1] = static_cast<uint8_t>(srcLine[x + 1] >> 8);
                dstLine[x + 2] = static_cast<uint8_t>(srcLine[x + 2] >> 8);
                dstLine[x + 3] = static_cast<uint8_t>(srcLine[x + 3] >> 8);
                dstLine[x + 4] = static_cast<uint8_t>(srcLine[x + 4] >> 8);
                dstLine[x + 5] = static_cast<uint8_t>(srcLine[x + 5] >> 8);
                dstLine[x + 6] = static_cast<uint8_t>(srcLine[x + 6] >> 8);
                dstLine[x + 7] = static_cast<uint8_t>(srcLine[x + 7] >> 8);
            }

            // Handle remaining pixels
            for (; x < width; x++) {
                dstLine[x] = static_cast<uint8_t>(srcLine[x] >> 8);
            }
        }

        return grayImage;
    }
    
    // Handle Y8 format (another 8-bit grayscale variant)
    // Note: Some libcamera versions may not have all these formats

    // Handle XRGB8888 format (32-bit with X as padding, layout: BGRX in memory)
    // This is the format PiSP outputs for the OV9281 mono camera
    if (config.pixelFormat == formats::XRGB8888) {
        QImage rgbImage(width, height, QImage::Format_RGB32);

        // XRGB8888 memory layout: B, G, R, X (padding) - same as Qt's Format_RGB32
        // Process as 32-bit words and set alpha to 0xff for Qt compatibility
        for (int y = 0; y < height; y++) {
            const uint32_t *srcLine = reinterpret_cast<const uint32_t*>(data + y * stride);
            uint32_t *dstLine = reinterpret_cast<uint32_t*>(rgbImage.scanLine(y));

            // Process 8 pixels at a time for better performance
            int x = 0;
            const int unrollCount = 8;
            const int unrolledWidth = width - (width % unrollCount);

            for (; x < unrolledWidth; x += unrollCount) {
                // Set alpha channel to 0xff (top byte), keep BGR as-is
                dstLine[x + 0] = srcLine[x + 0] | 0xff000000;
                dstLine[x + 1] = srcLine[x + 1] | 0xff000000;
                dstLine[x + 2] = srcLine[x + 2] | 0xff000000;
                dstLine[x + 3] = srcLine[x + 3] | 0xff000000;
                dstLine[x + 4] = srcLine[x + 4] | 0xff000000;
                dstLine[x + 5] = srcLine[x + 5] | 0xff000000;
                dstLine[x + 6] = srcLine[x + 6] | 0xff000000;
                dstLine[x + 7] = srcLine[x + 7] | 0xff000000;
            }

            // Handle remaining pixels
            for (; x < width; x++) {
                dstLine[x] = srcLine[x] | 0xff000000;
            }
        }

        return rgbImage;
    }
    
    // Handle XBGR8888 format
    if (config.pixelFormat == formats::XBGR8888) {
        QImage rgbImage(width, height, QImage::Format_RGB32);
        
        for (int y = 0; y < height; y++) {
            const uint8_t *srcLine = data + y * stride;
            uint32_t *dstLine = reinterpret_cast<uint32_t*>(rgbImage.scanLine(y));
            
            for (int x = 0; x < width; x++) {
                // XBGR8888: memory order is R, G, B, X (little endian)
                uint8_t r = srcLine[x * 4 + 0];
                uint8_t g = srcLine[x * 4 + 1];
                uint8_t b = srcLine[x * 4 + 2];
                dstLine[x] = qRgb(r, g, b);
            }
        }
        
        return rgbImage;
    }
    
    // Handle ARGB8888 format
    if (config.pixelFormat == formats::ARGB8888) {
        // Qt's Format_ARGB32 matches this layout
        QImage argbImage(data, width, height, stride, QImage::Format_ARGB32);
        return argbImage.copy();
    }
    
    // Handle ABGR8888 format
    if (config.pixelFormat == formats::ABGR8888) {
        QImage rgbImage(width, height, QImage::Format_RGB32);
        
        for (int y = 0; y < height; y++) {
            const uint8_t *srcLine = data + y * stride;
            uint32_t *dstLine = reinterpret_cast<uint32_t*>(rgbImage.scanLine(y));
            
            for (int x = 0; x < width; x++) {
                uint8_t r = srcLine[x * 4 + 0];
                uint8_t g = srcLine[x * 4 + 1];
                uint8_t b = srcLine[x * 4 + 2];
                dstLine[x] = qRgb(r, g, b);
            }
        }
        
        return rgbImage;
    }
    
    // Handle YUV420 format (common for cameras)
    if (config.pixelFormat == formats::YUV420) {
        // YUV420 has Y plane followed by U and V planes (each 1/4 size)
        QImage rgbImage(width, height, QImage::Format_RGB888);
        
        const uint8_t *yPlane = data;
        const uint8_t *uPlane = data + width * height;
        const uint8_t *vPlane = uPlane + (width * height) / 4;
        
        for (int y = 0; y < height; y++) {
            uint8_t *rgbLine = rgbImage.scanLine(y);
            
            for (int x = 0; x < width; x++) {
                int yIdx = y * width + x;
                int uvIdx = (y / 2) * (width / 2) + (x / 2);
                
                int Y = yPlane[yIdx];
                int U = uPlane[uvIdx] - 128;
                int V = vPlane[uvIdx] - 128;
                
                // YUV to RGB conversion
                int R = Y + ((359 * V) >> 8);
                int G = Y - ((88 * U + 183 * V) >> 8);
                int B = Y + ((454 * U) >> 8);
                
                // Clamp values
                R = qBound(0, R, 255);
                G = qBound(0, G, 255);
                B = qBound(0, B, 255);
                
                rgbLine[x * 3 + 0] = R;
                rgbLine[x * 3 + 1] = G;
                rgbLine[x * 3 + 2] = B;
            }
        }
        
        return rgbImage;
    }
    
    // Handle RGB/BGR formats
    if (config.pixelFormat == formats::RGB888) {
        return QImage(data, width, height, config.stride, QImage::Format_RGB888).copy();
    }
    
    if (config.pixelFormat == formats::BGR888) {
        QImage img(data, width, height, config.stride, QImage::Format_RGB888);
        return img.rgbSwapped();
    }
    
    // Handle YUYV (packed YUV 4:2:2)
    if (config.pixelFormat == formats::YUYV) {
        QImage rgbImage(width, height, QImage::Format_RGB888);
        
        for (int y = 0; y < height; y++) {
            const uint8_t *srcLine = data + y * config.stride;
            uint8_t *dstLine = rgbImage.scanLine(y);
            
            for (int x = 0; x < width; x += 2) {
                int Y0 = srcLine[x * 2];
                int U = srcLine[x * 2 + 1] - 128;
                int Y1 = srcLine[x * 2 + 2];
                int V = srcLine[x * 2 + 3] - 128;
                
                // First pixel
                int R = Y0 + ((359 * V) >> 8);
                int G = Y0 - ((88 * U + 183 * V) >> 8);
                int B = Y0 + ((454 * U) >> 8);
                
                dstLine[x * 3 + 0] = qBound(0, R, 255);
                dstLine[x * 3 + 1] = qBound(0, G, 255);
                dstLine[x * 3 + 2] = qBound(0, B, 255);
                
                // Second pixel
                R = Y1 + ((359 * V) >> 8);
                G = Y1 - ((88 * U + 183 * V) >> 8);
                B = Y1 + ((454 * U) >> 8);
                
                dstLine[(x + 1) * 3 + 0] = qBound(0, R, 255);
                dstLine[(x + 1) * 3 + 1] = qBound(0, G, 255);
                dstLine[(x + 1) * 3 + 2] = qBound(0, B, 255);
            }
        }
        
        return rgbImage;
    }
    
    qWarning() << "Unsupported pixel format:" 
               << QString::fromStdString(config.pixelFormat.toString());
    return QImage();
}

// ============================================================================
// CameraController - delegates to Impl
// ============================================================================
CameraController::CameraController(QObject *parent)
    : QObject(parent)
    , impl_(std::make_unique<Impl>(this))
{
    // Register QImage for cross-thread signal/slot
    qRegisterMetaType<QImage>("QImage");
}

CameraController::~CameraController() = default;

bool CameraController::initialize(int cameraIndex)
{
    return impl_->initialize(cameraIndex);
}

bool CameraController::start()
{
    return impl_->start();
}

void CameraController::stop()
{
    impl_->stop();
}

bool CameraController::isRunning() const
{
    return impl_->isRunning();
}

uint64_t CameraController::frameCount() const
{
    return impl_->frameCount();
}

double CameraController::currentFps() const
{
    return impl_->currentFps();
}

QString CameraController::cameraId() const
{
    return impl_->cameraId();
}

QSize CameraController::frameSize() const
{
    return impl_->frameSize();
}

void CameraController::setTriggerMode(bool enabled)
{
    impl_->setTriggerMode(enabled);
}

bool CameraController::triggerMode() const
{
    return impl_->triggerMode();
}

void CameraController::setAutoExposure(bool enabled)
{
    impl_->setAutoExposure(enabled);
}

bool CameraController::autoExposure() const
{
    return impl_->autoExposure();
}

void CameraController::setExposureTime(int microseconds)
{
    impl_->setExposureTime(microseconds);
}

int CameraController::exposureTime() const
{
    return impl_->exposureTime();
}

void CameraController::setGain(float gain)
{
    impl_->setGain(gain);
}

float CameraController::gain() const
{
    return impl_->gain();
}
