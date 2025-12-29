/**
 * Pear VIO Camera API - Camera Controller
 *
 * libcamera-based camera controller for OV9281 sensor.
 */

#ifndef PEAR_CAMERA_H
#define PEAR_CAMERA_H

#include <memory>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <map>

#include <opencv2/core/core.hpp>

#include "config.h"

// Forward declarations for libcamera types
namespace libcamera {
    class CameraManager;
    class Camera;
    class CameraConfiguration;
    class FrameBufferAllocator;
    class Request;
    class Stream;
    class FrameBuffer;
    class PixelFormat;
}

namespace pear {

/**
 * Camera controller for OV9281 sensor using libcamera API.
 *
 * Features:
 * - Grayscale image capture (R8 format)
 * - Manual or auto exposure control
 * - Hardware trigger support
 * - Thread-safe frame retrieval
 */
class CameraController {
public:
    CameraController();
    ~CameraController();

    // Non-copyable
    CameraController(const CameraController&) = delete;
    CameraController& operator=(const CameraController&) = delete;

    /**
     * Initialize camera with given configuration.
     * @param config Camera configuration
     * @return true if successful
     */
    bool initialize(const CameraConfig& config);

    /**
     * Initialize camera with default configuration.
     * @param cameraIndex Camera device index (default 0)
     * @return true if successful
     */
    bool initialize(int cameraIndex = 0);

    /**
     * Load additional settings from INI file (exposure, gain, etc.).
     * @param configPath Path to INI file (empty for default path)
     * @return true if loaded successfully
     */
    bool loadSettings(const std::string& configPath = "");

    /**
     * Start camera capture.
     * @return true if successful
     */
    bool start();

    /**
     * Stop camera capture and release resources.
     */
    void stop();

    /**
     * Wait for and retrieve next frame (blocking).
     * @param frame Output grayscale image
     * @param timestamp Output sensor timestamp in seconds
     * @return true if frame retrieved, false if camera stopped
     */
    bool getFrame(cv::Mat& frame, double& timestamp);

    /**
     * Check if camera is running.
     */
    bool isRunning() const { return running_.load(); }

    /**
     * Get frame width.
     */
    int width() const { return frameWidth_; }

    /**
     * Get frame height.
     */
    int height() const { return frameHeight_; }

    /**
     * Get current exposure time in microseconds.
     */
    int exposureTimeUs() const { return config_.exposureTimeUs; }

    /**
     * Get current configuration.
     */
    const CameraConfig& config() const { return config_; }

private:
    // libcamera callback (implemented in cpp)
    void requestComplete(libcamera::Request* request);

    // Configuration
    CameraConfig config_;

    // libcamera objects (pimpl-like, actual types in cpp)
    struct Impl;
    std::unique_ptr<Impl> impl_;

    // Frame dimensions
    int frameWidth_ = 0;
    int frameHeight_ = 0;
    int frameStride_ = 0;

    // State
    std::atomic<bool> running_{false};

    // Frame synchronization
    std::mutex frameMtx_;
    std::condition_variable frameCond_;
    cv::Mat currentFrame_;
    double currentTimestamp_ = 0;
    bool frameReady_ = false;
};

}  // namespace pear

#endif  // PEAR_CAMERA_H
