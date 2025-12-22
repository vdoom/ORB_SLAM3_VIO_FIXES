#include "RosbagRecorder.h"
#include "RosBagWriter.h"

#include <QDebug>
#include <QFileInfo>

// Unit conversion constants
static constexpr double DEG_TO_RAD = 0.0174532925;  // pi/180
static constexpr double G_TO_MS2 = 9.80665;

RosbagRecorder::RosbagRecorder(QObject *parent)
    : QObject(parent)
{
}

RosbagRecorder::~RosbagRecorder()
{
    if (isRecording_) {
        stopRecording();
    }
}

bool RosbagRecorder::startRecording(const QString& filePath)
{
    QMutexLocker locker(&mutex_);

    if (isRecording_) {
        emit errorOccurred("Already recording");
        return false;
    }

    // Create writer
    writer_ = std::make_unique<rosbag::RosBagWriter>();

    if (!writer_->open(filePath.toStdString())) {
        emit errorOccurred(QString("Failed to open bag file: %1").arg(filePath));
        writer_.reset();
        return false;
    }

    // Add topics
    int camConnId = writer_->addImageTopic(cameraTopic_.toStdString(), "cam0");
    if (camConnId < 0) {
        emit errorOccurred("Failed to add camera topic");
        writer_.reset();
        return false;
    }

    int imuConnId = writer_->addImuTopic(imuTopic_.toStdString(), "imu0");
    if (imuConnId < 0) {
        emit errorOccurred("Failed to add IMU topic");
        writer_.reset();
        return false;
    }

    // Reset state
    currentFilePath_ = filePath;
    frameCount_ = 0;
    imuCount_ = 0;
    skippedFrameCount_ = 0;
    firstPicoTimestampMs_ = 0;
    hasFirstPicoTimestamp_ = false;
    pendingCameraTriggerMs_ = 0;
    hasPendingCameraTrigger_ = false;
    isRecording_ = true;

    qDebug() << "Rosbag recording started:" << filePath;
    qDebug() << "  Camera topic:" << cameraTopic_;
    qDebug() << "  IMU topic:" << imuTopic_;

    emit recordingStarted(filePath);
    return true;
}

void RosbagRecorder::stopRecording()
{
    QMutexLocker locker(&mutex_);

    if (!isRecording_) {
        return;
    }

    isRecording_ = false;

    if (writer_) {
        writer_->close();
        writer_.reset();
    }

    qDebug() << "Rosbag recording stopped:" << currentFilePath_;
    qDebug() << "  Frames:" << frameCount_.load();
    qDebug() << "  IMU samples:" << imuCount_.load();

    emit recordingStopped(currentFilePath_, frameCount_, imuCount_);
}

void RosbagRecorder::addCameraFrame(const QImage& frame)
{
    // Quick check without lock
    if (!isRecording_) {
        return;
    }

    // Convert to grayscale OUTSIDE the mutex (expensive operation)
    QImage grayFrame;
    if (frame.format() == QImage::Format_Grayscale8) {
        grayFrame = frame.copy();  // Make a copy to ensure data ownership
    } else {
        grayFrame = frame.convertToFormat(QImage::Format_Grayscale8);
    }

    // Validate the converted image
    if (grayFrame.isNull() || grayFrame.width() == 0 || grayFrame.height() == 0) {
        qWarning() << "RosbagRecorder: Invalid grayscale frame";
        return;
    }

    // Prepare data for writing
    std::string topic = cameraTopic_.toStdString();
    uint32_t width = static_cast<uint32_t>(grayFrame.width());
    uint32_t height = static_cast<uint32_t>(grayFrame.height());

    // Now lock and check for pending trigger timestamp
    QMutexLocker locker(&mutex_);

    if (!writer_ || !isRecording_) {
        return;
    }

    // Check if we have a pending camera trigger timestamp from Pico
    if (!hasPendingCameraTrigger_) {
        // No trigger timestamp available - skip this frame
        skippedFrameCount_++;
        QString reason = QString("No trigger timestamp available (skipped %1 frames)")
                            .arg(skippedFrameCount_.load());
        locker.unlock();
        emit frameSkipped(reason);
        return;
    }

    // Use the Pico trigger timestamp
    uint64_t timestampNsec = picoTimestampToNsec(pendingCameraTriggerMs_);
    rosbag::Time stamp = rosbag::Time::fromNsec(timestampNsec);

    // Clear the pending trigger (it's been consumed)
    hasPendingCameraTrigger_ = false;

    // Write image
    bool ok = writer_->writeImage(
        topic,
        stamp,
        width,
        height,
        grayFrame.constBits(),
        static_cast<size_t>(width * height),  // Use actual data size for mono8
        "mono8"
    );

    if (ok) {
        frameCount_++;

        // Check if we should emit status update
        bool shouldEmit = (frameCount_ % 30 == 0);
        uint64_t fileSize = 0;
        if (shouldEmit && writer_) {
            fileSize = writer_->fileSize();
        }

        // Release mutex BEFORE emitting signal to avoid deadlock
        locker.unlock();

        if (shouldEmit) {
            emit statusUpdated(frameCount_, imuCount_, fileSize);
        }
    }
}

void RosbagRecorder::addImuSample(const IMUData& data)
{
    if (!isRecording_) {
        return;
    }

    QMutexLocker locker(&mutex_);

    if (!writer_) {
        return;
    }

    // Convert Pico timestamp to ROS time
    uint64_t timestampNsec = picoTimestampToNsec(data.timestamp_ms);
    rosbag::Time stamp = rosbag::Time::fromNsec(timestampNsec);

    // Convert to SI units (m/s² and rad/s)
    double accel[3] = {
        data.accel_x * G_TO_MS2,
        data.accel_y * G_TO_MS2,
        data.accel_z * G_TO_MS2
    };

    double gyro[3] = {
        data.gyro_x * DEG_TO_RAD,
        data.gyro_y * DEG_TO_RAD,
        data.gyro_z * DEG_TO_RAD
    };

    // Write IMU message
    bool ok = writer_->writeImu(imuTopic_.toStdString(), stamp, accel, gyro);

    if (ok) {
        imuCount_++;
    }
}

void RosbagRecorder::setCameraTriggerTimestamp(uint64_t timestamp_ms)
{
    if (!isRecording_) {
        return;
    }

    QMutexLocker locker(&mutex_);

    // Check for duplicate trigger (previous trigger not consumed by a frame)
    if (hasPendingCameraTrigger_) {
        // This is an error condition - exposure took too long
        uint64_t prevTs = pendingCameraTriggerMs_;
        locker.unlock();
        emit duplicateTriggerWarning(prevTs, timestamp_ms);
        locker.relock();
    }

    pendingCameraTriggerMs_ = timestamp_ms;
    hasPendingCameraTrigger_ = true;
}

uint64_t RosbagRecorder::picoTimestampToNsec(uint64_t timestamp_ms)
{
    // Track first Pico timestamp to establish common epoch for camera and IMU
    if (!hasFirstPicoTimestamp_) {
        firstPicoTimestampMs_ = timestamp_ms;
        hasFirstPicoTimestamp_ = true;
    }

    // Convert relative to first timestamp, then to nanoseconds
    uint64_t relativeMs = timestamp_ms - firstPicoTimestampMs_;
    return relativeMs * 1000000ULL;  // ms to ns
}

uint64_t RosbagRecorder::fileSizeBytes() const
{
    QMutexLocker locker(&mutex_);
    if (writer_) {
        return writer_->fileSize();
    }
    return 0;
}
