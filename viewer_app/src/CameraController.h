#ifndef CAMERA_CONTROLLER_H
#define CAMERA_CONTROLLER_H

#include <QObject>
#include <QImage>
#include <QSize>
#include <memory>

/**
 * CameraController - Handles libcamera operations similar to rpicam-hello
 * 
 * Uses libcamera's request-based API to capture frames from the CSI camera.
 * Emits signals when new frames are available for display.
 * 
 * Implementation uses PIMPL pattern to isolate libcamera from Qt headers.
 */
class CameraController : public QObject
{
    Q_OBJECT

public:
    explicit CameraController(QObject *parent = nullptr);
    ~CameraController();

    bool initialize(int cameraIndex = 0);
    bool start();
    void stop();

    bool isRunning() const;

    // Trigger mode control
    void setTriggerMode(bool enabled);
    bool triggerMode() const;

    // Exposure control
    void setAutoExposure(bool enabled);
    bool autoExposure() const;
    void setExposureTime(int microseconds);
    int exposureTime() const;

    // Gain control (only used when auto exposure is off)
    void setGain(float gain);
    float gain() const;

    // Frame statistics
    uint64_t frameCount() const;
    double currentFps() const;

    // Camera info
    QString cameraId() const;
    QSize frameSize() const;

Q_SIGNALS:
    void frameReady(const QImage &frame);
    void errorOccurred(const QString &error);
    void fpsUpdated(double fps, uint64_t frameCount);
    void exposureUpdated(int exposureUs);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

#endif // CAMERA_CONTROLLER_H
