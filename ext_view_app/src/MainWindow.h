#ifndef MAIN_WINDOW_H
#define MAIN_WINDOW_H

#include <QMainWindow>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QToolBar>
#include <QAction>
#include <QSlider>
#include <QLabel>
#include <QTextEdit>
#include <QGroupBox>
#include <QLineEdit>
#include <QSettings>
#include <QFile>
#include <QTextStream>
#include <memory>

#include "SerialIMUReader.h"

class CameraWidget;
class RosbagRecorder;
class CameraController;
class Axes3DWidget;

/**
 * MainWindow - Main application window
 *
 * Layout structure:
 * +--------------------------------------------------+
 * | Toolbar                                          |
 * +----------------------------------+---------------+
 * |                                  | GyroView      |
 * |                                  | (3D axes)     |
 * |       CameraView                 +---------------+
 * |       (main camera feed)         | AccelView     |
 * |                                  | (3D axes)     |
 * |                                  +---------------+
 * |                                  | InfoPanel     |
 * |                                  | (text info)   |
 * +----------------------------------+---------------+
 *
 * Widget naming:
 * - CameraView: Main camera feed display
 * - GyroView: 3D axes for gyroscope visualization
 * - AccelView: 3D axes for accelerometer visualization
 * - InfoPanel: Text box for additional telemetry info
 */
class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

protected:
    void closeEvent(QCloseEvent *event) override;
    void keyPressEvent(QKeyEvent *event) override;

private slots:
    void onFpsUpdated(double fps, uint64_t frameCount);
    void onCameraError(const QString &error);
    void onTriggerModeToggled(bool enabled);
    void onAutoExposureToggled(bool enabled);
    void onExposureSliderChanged(int value);
    void onGainSliderChanged(int value);
    void onExposureUpdated(int exposureUs);

    // IMU slots
    void onIMUDataReceived(const IMUData &data);
    void onIMUError(const QString &error);
    void onIMUConnectionChanged(bool connected);
    void onConnectIMUClicked();
    void onLogIMUToggled(bool enabled);
    void onSIUnitsToggled(bool enabled);

    // Rosbag recording slots
    void onRecordRosbagToggled(bool enabled);
    void onRosbagRecordingStarted(const QString& filePath);
    void onRosbagRecordingStopped(const QString& filePath, uint64_t frameCount, uint64_t imuCount);
    void onRosbagStatusUpdated(uint64_t frameCount, uint64_t imuCount, uint64_t fileSizeBytes);
    void onRosbagError(const QString& error);

private:
    void setupUi();
    void setupToolbar();
    void setupSidePanel();
    void setupCamera();
    void setupIMU();
    void updateWindowTitle();
    void updateExposureSliderRange();
    void updateInfoPanel();
    void loadSettings();
    void saveSettings();

    // Main layout components
    QWidget *centralWidget_ = nullptr;
    QVBoxLayout *mainLayout_ = nullptr;
    QHBoxLayout *contentLayout_ = nullptr;

    // Toolbar
    QToolBar *toolbar_ = nullptr;
    QAction *triggerModeAction_ = nullptr;
    QAction *autoExposureAction_ = nullptr;
    QSlider *exposureSlider_ = nullptr;
    QLabel *exposureLabel_ = nullptr;
    QSlider *gainSlider_ = nullptr;
    QLabel *gainLabel_ = nullptr;

    // Camera components (left side - main view)
    CameraWidget *cameraWidget_ = nullptr;
    std::unique_ptr<CameraController> cameraController_;

    // Side panel (right side)
    QWidget *sidePanel_ = nullptr;
    QVBoxLayout *sidePanelLayout_ = nullptr;

    // GyroView - 3D visualization for gyroscope data
    QGroupBox *gyroGroupBox_ = nullptr;
    Axes3DWidget *gyroView_ = nullptr;

    // AccelView - 3D visualization for accelerometer data
    QGroupBox *accelGroupBox_ = nullptr;
    Axes3DWidget *accelView_ = nullptr;

    // InfoPanel - Text display for telemetry data
    QGroupBox *infoGroupBox_ = nullptr;
    QTextEdit *infoPanel_ = nullptr;

    // Settings
    std::unique_ptr<QSettings> settings_;
    bool triggerModeEnabled_ = false;
    bool autoExposureEnabled_ = true;
    int manualExposureUs_ = 10000;  // Default 10ms
    float manualGain_ = 8.0f;       // Default gain for manual mode

    // Statistics
    double currentFps_ = 0.0;
    uint64_t frameCount_ = 0;
    int currentExposureUs_ = 0;

    // IMU components
    std::unique_ptr<SerialIMUReader> imuReader_;
    QLineEdit *serialPortEdit_ = nullptr;
    QAction *connectIMUAction_ = nullptr;
    QAction *logIMUAction_ = nullptr;
    QAction *siUnitsAction_ = nullptr;
    QString serialPortName_ = "/dev/ttyACM0";
    bool useSIUnits_ = false;  // false = deg/s and g, true = rad/s and m/s²

    // Last IMU data for display
    IMUData lastIMUData_;
    uint64_t lastFrameTimestamp_ = 0;

    // IMU logging
    std::unique_ptr<QFile> logFile_;
    std::unique_ptr<QTextStream> logStream_;
    bool isLogging_ = false;

    // Rosbag recording
    std::unique_ptr<RosbagRecorder> rosbagRecorder_;
    QAction *recordRosbagAction_ = nullptr;

    // UI settings (saved to config)
    int sidePanelWidth_ = 250;
    int infoPanelFontSize_ = 8;
    int windowWidth_ = 1400;
    int windowHeight_ = 800;
};

#endif // MAIN_WINDOW_H
