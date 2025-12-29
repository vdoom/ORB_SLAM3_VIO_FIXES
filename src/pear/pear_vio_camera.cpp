/**
 * Pear VIO Camera API - High-Level Interface Implementation
 */

#include "pear/pear_vio_camera.h"
#include "pear/camera.h"
#include "pear/imu.h"

#include <iostream>
#include <chrono>
#include <thread>

using namespace std;

namespace pear {

PearVIOCamera::PearVIOCamera()
    : camera_(make_unique<CameraController>())
    , imu_(make_unique<SerialIMUReader>()) {
}

PearVIOCamera::~PearVIOCamera() {
    stop();
}

bool PearVIOCamera::open(const PearConfig& config) {
    config_ = config;

    // Initialize IMU reader
    if (!imu_->open(config.imu)) {
        cerr << "Failed to open IMU serial port: " << config.imu.serialPort << endl;
        cerr << "Check that:" << endl;
        cerr << "  - Pico is connected and running" << endl;
        cerr << "  - User has dialout group access: sudo usermod -aG dialout $USER" << endl;
        return false;
    }

    // Initialize camera
    if (!camera_->initialize(config.camera)) {
        cerr << "Failed to initialize camera" << endl;
        imu_->close();
        return false;
    }

    return true;
}

bool PearVIOCamera::open(const string& serialPort, int cameraIndex) {
    PearConfig config;
    config.imu.serialPort = serialPort;
    config.camera.cameraIndex = cameraIndex;
    return open(config);
}

bool PearVIOCamera::loadSettings(const string& configPath) {
    bool result = camera_->loadSettings(configPath);

    // Also try to load full config if path provided
    if (!configPath.empty()) {
        config_.loadFromIniFile(configPath);
    }

    return result;
}

bool PearVIOCamera::start() {
    if (started_) return true;

    if (!camera_->start()) {
        cerr << "Failed to start camera" << endl;
        return false;
    }

    started_ = true;
    return true;
}

void PearVIOCamera::stop() {
    if (!started_) return;

    started_ = false;
    camera_->stop();
    imu_->close();

    cout << "PearVIOCamera stopped" << endl;
}

bool PearVIOCamera::isRunning() const {
    return started_ && camera_->isRunning();
}

bool PearVIOCamera::getFrame(VIOFrame& frame) {
    if (!started_) return false;

    // Get camera frame (blocking)
    cv::Mat image;
    double cameraTimestamp;
    if (!camera_->getFrame(image, cameraTimestamp)) {
        return false;
    }

    // Get trigger timestamp (if available)
    uint64_t triggerTimestamp = imu_->getCameraTriggerTimestamp();

    // Get IMU data
    vector<IMUData> imuData = imu_->getIMUData();

    // Initialize time base if needed
    if (firstImuTimestamp_ == 0 && !imuData.empty()) {
        firstImuTimestamp_ = imuData.front().timestamp_ms;
        cout << "Time base established: " << firstImuTimestamp_ << " ms" << endl;
    }

    // Fill frame structure
    frame.image = image;
    frame.imuData = std::move(imuData);
    frame.hasTrigger = (triggerTimestamp > 0);
    frame.triggerTimestampMs = triggerTimestamp;

    // Calculate frame timestamp
    if (triggerTimestamp > 0 && firstImuTimestamp_ > 0) {
        frame.timestamp = static_cast<double>(triggerTimestamp - firstImuTimestamp_) / 1000.0;
    } else {
        // Fallback to camera timestamp (relative)
        frame.timestamp = cameraTimestamp;
    }

    return true;
}

bool PearVIOCamera::getFrameOnly(cv::Mat& image, double& timestamp) {
    if (!started_) return false;
    return camera_->getFrame(image, timestamp);
}

vector<IMUData> PearVIOCamera::getIMUData() {
    return imu_->getIMUData();
}

uint64_t PearVIOCamera::getTriggerTimestamp() {
    return imu_->getCameraTriggerTimestamp();
}

void PearVIOCamera::clearBuffers() {
    imu_->clearIMUQueue();
    imu_->clearTriggerQueue();
}

uint64_t PearVIOCamera::waitForIMU(int timeoutMs) {
    cout << "Waiting for IMU data..." << endl;

    auto startTime = chrono::steady_clock::now();

    while (true) {
        auto imuData = imu_->getIMUData();
        if (!imuData.empty()) {
            firstImuTimestamp_ = imuData.front().timestamp_ms;
            cout << "First IMU timestamp: " << firstImuTimestamp_ << " ms" << endl;

            // Print first IMU reading to verify values
            const auto& first = imuData.front();
            cout << "First IMU reading (raw):" << endl;
            cout << "  Accel: [" << first.accel_x << ", " << first.accel_y << ", " << first.accel_z << "] g" << endl;
            cout << "  Gyro:  [" << first.gyro_x << ", " << first.gyro_y << ", " << first.gyro_z << "] deg/s" << endl;

            float ax, ay, az;
            first.getAccelSI(ax, ay, az);
            cout << "  Accel (SI): [" << ax << ", " << ay << ", " << az << "] m/s²" << endl;

            return firstImuTimestamp_;
        }

        // Check timeout
        auto elapsed = chrono::duration_cast<chrono::milliseconds>(
            chrono::steady_clock::now() - startTime).count();
        if (elapsed >= timeoutMs) {
            cerr << "Timeout waiting for IMU data" << endl;
            return 0;
        }

        this_thread::sleep_for(chrono::milliseconds(10));
    }
}

int PearVIOCamera::width() const {
    return camera_->width();
}

int PearVIOCamera::height() const {
    return camera_->height();
}

int PearVIOCamera::exposureTimeUs() const {
    return camera_->exposureTimeUs();
}

bool PearVIOCamera::setTriggerMode(bool enable) {
    return camera_->setTriggerMode(enable);
}

bool PearVIOCamera::isTriggerModeEnabled() const {
    return camera_->isTriggerModeEnabled();
}

}  // namespace pear
