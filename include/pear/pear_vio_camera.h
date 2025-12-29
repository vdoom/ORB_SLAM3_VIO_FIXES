/**
 * Pear VIO Camera API - High-Level Interface
 *
 * Unified interface combining camera and IMU for Visual-Inertial Odometry.
 */

#ifndef PEAR_VIO_CAMERA_H
#define PEAR_VIO_CAMERA_H

#include <memory>
#include <vector>
#include <string>

#include <opencv2/core/core.hpp>

#include "types.h"
#include "config.h"

namespace pear {

// Forward declarations
class CameraController;
class SerialIMUReader;

/**
 * Frame data with associated IMU measurements.
 */
struct VIOFrame {
    cv::Mat image;                      // Grayscale image
    double timestamp = 0;               // Frame timestamp in seconds (relative to start)
    std::vector<IMUData> imuData;       // IMU measurements since last frame
    bool hasTrigger = false;            // True if frame has hardware trigger timestamp
    uint64_t triggerTimestampMs = 0;    // Hardware trigger timestamp (if available)
};

/**
 * High-level VIO camera interface.
 *
 * Combines CameraController and SerialIMUReader into a single interface
 * for easy integration with VIO/SLAM libraries.
 *
 * Usage:
 *   pear::PearVIOCamera camera;
 *   camera.open("/dev/ttyACM0");
 *   camera.start();
 *
 *   while (running) {
 *       pear::VIOFrame frame;
 *       if (camera.getFrame(frame)) {
 *           // Process frame.image and frame.imuData
 *       }
 *   }
 *
 *   camera.stop();
 */
class PearVIOCamera {
public:
    PearVIOCamera();
    ~PearVIOCamera();

    // Non-copyable
    PearVIOCamera(const PearVIOCamera&) = delete;
    PearVIOCamera& operator=(const PearVIOCamera&) = delete;

    /**
     * Open camera and IMU with configuration.
     * @param config Full configuration
     * @return true if successful
     */
    bool open(const PearConfig& config);

    /**
     * Open camera and IMU with default configuration.
     * @param serialPort Serial port for IMU (default "/dev/ttyACM0")
     * @param cameraIndex Camera device index (default 0)
     * @return true if successful
     */
    bool open(const std::string& serialPort = "/dev/ttyACM0", int cameraIndex = 0);

    /**
     * Load camera settings from INI file.
     * @param configPath Path to INI file (empty for default)
     * @return true if loaded
     */
    bool loadSettings(const std::string& configPath = "");

    /**
     * Start camera and IMU capture.
     * @return true if successful
     */
    bool start();

    /**
     * Stop capture and release resources.
     */
    void stop();

    /**
     * Check if camera is running.
     */
    bool isRunning() const;

    /**
     * Wait for next frame with IMU data (blocking).
     *
     * This method:
     * 1. Waits for next camera frame
     * 2. Collects all IMU data received since last frame
     * 3. Associates hardware trigger timestamp if available
     *
     * @param frame Output frame with image and IMU data
     * @return true if successful, false if camera stopped
     */
    bool getFrame(VIOFrame& frame);

    /**
     * Get frame image only (without IMU data).
     * @param image Output grayscale image
     * @param timestamp Output timestamp in seconds
     * @return true if successful
     */
    bool getFrameOnly(cv::Mat& image, double& timestamp);

    /**
     * Get all accumulated IMU data.
     * @return Vector of IMU measurements
     */
    std::vector<IMUData> getIMUData();

    /**
     * Get next camera trigger timestamp.
     * @return Timestamp in ms, or 0 if none available
     */
    uint64_t getTriggerTimestamp();

    /**
     * Clear IMU and trigger buffers.
     * Useful after initialization to start fresh.
     */
    void clearBuffers();

    /**
     * Wait for first IMU data and establish time base.
     * @param timeoutMs Timeout in milliseconds
     * @return First IMU timestamp in ms, or 0 if timeout
     */
    uint64_t waitForIMU(int timeoutMs = 5000);

    /**
     * Get frame dimensions.
     */
    int width() const;
    int height() const;

    /**
     * Get exposure time in microseconds.
     */
    int exposureTimeUs() const;

    /**
     * Enable or disable hardware trigger mode on OV9281 via I2C.
     *
     * Must be called AFTER start(). When enabled, the camera waits for
     * external trigger pulses from the Pico instead of free-running.
     *
     * @param enable true to enable trigger mode, false for free-run
     * @return true if successful
     */
    bool setTriggerMode(bool enable);

    /**
     * Check if trigger mode is currently enabled.
     */
    bool isTriggerModeEnabled() const;

    /**
     * Get current configuration.
     */
    const PearConfig& config() const { return config_; }

    /**
     * Get first IMU timestamp (time base).
     * Set after waitForIMU() or first successful getFrame().
     */
    uint64_t firstIMUTimestamp() const { return firstImuTimestamp_; }

private:
    PearConfig config_;
    std::unique_ptr<CameraController> camera_;
    std::unique_ptr<SerialIMUReader> imu_;
    uint64_t firstImuTimestamp_ = 0;
    bool started_ = false;
};

}  // namespace pear

#endif  // PEAR_VIO_CAMERA_H
