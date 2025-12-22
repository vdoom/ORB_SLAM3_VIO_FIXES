#ifndef ROSBAG_RECORDER_H
#define ROSBAG_RECORDER_H

/**
 * RosbagRecorder - Records camera and IMU data to ROS bag files
 *
 * This class provides a Qt-friendly interface for recording synchronized
 * camera and IMU data to ROS bag format for use with Kalibr calibration.
 *
 * Usage:
 *   RosbagRecorder recorder;
 *   recorder.startRecording("/path/to/output.bag");
 *
 *   // Connect to data sources
 *   connect(cameraController, &CameraController::frameReady,
 *           &recorder, &RosbagRecorder::addCameraFrame);
 *   connect(imuReader, &SerialIMUReader::dataReceived,
 *           &recorder, &RosbagRecorder::addImuSample);
 *
 *   // Later...
 *   recorder.stopRecording();
 *
 * Topics written:
 *   /cam0/image_raw - sensor_msgs/Image (mono8)
 *   /imu0 - sensor_msgs/Imu
 */

#include <QObject>
#include <QString>
#include <QImage>
#include <QMutex>
#include <memory>
#include <atomic>

#include "SerialIMUReader.h"

namespace rosbag {
    class RosBagWriter;
}

class RosbagRecorder : public QObject
{
    Q_OBJECT

public:
    explicit RosbagRecorder(QObject *parent = nullptr);
    ~RosbagRecorder() override;

    // Check if currently recording
    bool isRecording() const { return isRecording_; }

    // Get statistics
    uint64_t frameCount() const { return frameCount_; }
    uint64_t imuCount() const { return imuCount_; }
    uint64_t fileSizeBytes() const;
    QString currentFilePath() const { return currentFilePath_; }

    // Topic names (can be customized before recording starts)
    void setCameraTopic(const QString& topic) { cameraTopic_ = topic; }
    void setImuTopic(const QString& topic) { imuTopic_ = topic; }
    QString cameraTopic() const { return cameraTopic_; }
    QString imuTopic() const { return imuTopic_; }

public slots:
    // Start recording to the specified file
    // Returns true if recording started successfully
    bool startRecording(const QString& filePath);

    // Stop recording and close the bag file
    void stopRecording();

    // Add a camera frame to the recording
    // The frame will be converted to grayscale (mono8) if needed
    // Uses pending camera trigger timestamp from Pico
    void addCameraFrame(const QImage& frame);

    // Add an IMU sample to the recording
    // Uses timestamp from Pico (data.timestamp_ms)
    void addImuSample(const IMUData& data);

    // Set pending camera trigger timestamp (called BEFORE addCameraFrame)
    // This timestamp comes from the Pico via serial
    void setCameraTriggerTimestamp(uint64_t timestamp_ms);

signals:
    // Emitted when recording starts
    void recordingStarted(const QString& filePath);

    // Emitted when recording stops
    void recordingStopped(const QString& filePath, uint64_t frameCount, uint64_t imuCount);

    // Emitted on error
    void errorOccurred(const QString& error);

    // Emitted when a camera frame is skipped (no trigger timestamp available)
    void frameSkipped(const QString& reason);

    // Emitted when duplicate trigger detected (exposure too slow)
    void duplicateTriggerWarning(uint64_t previousTimestamp, uint64_t newTimestamp);

    // Emitted periodically with recording status
    void statusUpdated(uint64_t frameCount, uint64_t imuCount, uint64_t fileSizeBytes);

private:
    // Convert Pico timestamp (ms) to ROS time (nsec)
    // Uses the first Pico timestamp as epoch reference (t=0)
    uint64_t picoTimestampToNsec(uint64_t timestamp_ms);

    std::unique_ptr<rosbag::RosBagWriter> writer_;
    QString currentFilePath_;

    // Topic names
    QString cameraTopic_ = "/cam0/image_raw";
    QString imuTopic_ = "/imu0";

    // Recording state
    std::atomic<bool> isRecording_{false};
    std::atomic<uint64_t> frameCount_{0};
    std::atomic<uint64_t> imuCount_{0};
    std::atomic<uint64_t> skippedFrameCount_{0};

    // Pico timestamp handling (common time base for camera and IMU)
    uint64_t firstPicoTimestampMs_ = 0;  // First Pico timestamp as epoch reference
    bool hasFirstPicoTimestamp_ = false;

    // Pending camera trigger timestamp from Pico
    uint64_t pendingCameraTriggerMs_ = 0;
    bool hasPendingCameraTrigger_ = false;

    // Thread safety
    mutable QMutex mutex_;
};

#endif // ROSBAG_RECORDER_H
