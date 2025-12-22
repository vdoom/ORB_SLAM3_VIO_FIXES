#include "MainWindow.h"
#include "CameraWidget.h"
#include "CameraController.h"
#include "Axes3DWidget.h"
#include "RosbagRecorder.h"

#include <QCloseEvent>
#include <QKeyEvent>
#include <QMessageBox>
#include <QApplication>
#include <QCoreApplication>
#include <QStandardPaths>
#include <QDateTime>
#include <QDir>
#include <QDebug>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    loadSettings();
    setupUi();
    setupSidePanel();
    setupToolbar();
    setupCamera();
    setupIMU();
}

MainWindow::~MainWindow()
{
    saveSettings();

    // Stop rosbag recording
    if (rosbagRecorder_ && rosbagRecorder_->isRecording()) {
        rosbagRecorder_->stopRecording();
    }

    // Stop IMU logging
    if (isLogging_) {
        onLogIMUToggled(false);
    }

    if (imuReader_) {
        imuReader_->close();
    }
    if (cameraController_) {
        cameraController_->stop();
    }
}

void MainWindow::setupUi()
{
    // Create central widget and main layout
    centralWidget_ = new QWidget(this);
    setCentralWidget(centralWidget_);

    mainLayout_ = new QVBoxLayout(centralWidget_);
    mainLayout_->setContentsMargins(2, 2, 2, 2);
    mainLayout_->setSpacing(2);

    // Create horizontal layout for camera + side panel
    contentLayout_ = new QHBoxLayout();
    contentLayout_->setSpacing(4);

    // Camera view takes most of the space (left side)
    cameraWidget_ = new CameraWidget(centralWidget_);
    contentLayout_->addWidget(cameraWidget_, 4);  // 4/5 of width

    mainLayout_->addLayout(contentLayout_);

    // Set initial window properties
    setWindowTitle("RPi Camera Viewer");
    resize(windowWidth_, windowHeight_);

    // Set minimum size
    setMinimumSize(800, 600);
}

void MainWindow::setupSidePanel()
{
    // Create side panel container (right side)
    sidePanel_ = new QWidget(centralWidget_);
    sidePanel_->setFixedWidth(sidePanelWidth_);

    sidePanelLayout_ = new QVBoxLayout(sidePanel_);
    sidePanelLayout_->setContentsMargins(0, 0, 0, 0);
    sidePanelLayout_->setSpacing(4);

    // GyroView - 3D axes for gyroscope visualization
    gyroGroupBox_ = new QGroupBox("GyroView", sidePanel_);
    QVBoxLayout *gyroLayout = new QVBoxLayout(gyroGroupBox_);
    gyroLayout->setContentsMargins(2, 2, 2, 2);
    gyroView_ = new Axes3DWidget(gyroGroupBox_);
    gyroView_->setTitle("Gyroscope");
    gyroLayout->addWidget(gyroView_);
    sidePanelLayout_->addWidget(gyroGroupBox_, 1);

    // AccelView - 3D axes for accelerometer visualization
    accelGroupBox_ = new QGroupBox("AccelView", sidePanel_);
    QVBoxLayout *accelLayout = new QVBoxLayout(accelGroupBox_);
    accelLayout->setContentsMargins(2, 2, 2, 2);
    accelView_ = new Axes3DWidget(accelGroupBox_);
    accelView_->setTitle("Accelerometer");
    accelLayout->addWidget(accelView_);
    sidePanelLayout_->addWidget(accelGroupBox_, 1);

    // InfoPanel - Text display for telemetry info
    infoGroupBox_ = new QGroupBox("InfoPanel", sidePanel_);
    QVBoxLayout *infoLayout = new QVBoxLayout(infoGroupBox_);
    infoLayout->setContentsMargins(2, 2, 2, 2);
    infoPanel_ = new QTextEdit(infoGroupBox_);
    infoPanel_->setReadOnly(true);
    infoPanel_->setFont(QFont("Sans", infoPanelFontSize_));
    infoPanel_->setPlainText("Hello");
    infoPanel_->setMinimumHeight(100);
    infoLayout->addWidget(infoPanel_);
    sidePanelLayout_->addWidget(infoGroupBox_, 1);

    // Add side panel to content layout
    contentLayout_->addWidget(sidePanel_);
}

void MainWindow::setupCamera()
{
    // Create camera controller
    cameraController_ = std::make_unique<CameraController>(this);
    
    // Connect signals
    connect(cameraController_.get(), &CameraController::frameReady,
            cameraWidget_, &CameraWidget::updateFrame,
            Qt::QueuedConnection);
    
    connect(cameraController_.get(), &CameraController::fpsUpdated,
            this, &MainWindow::onFpsUpdated,
            Qt::QueuedConnection);
    
    connect(cameraController_.get(), &CameraController::errorOccurred,
            this, &MainWindow::onCameraError,
            Qt::QueuedConnection);

    connect(cameraController_.get(), &CameraController::exposureUpdated,
            this, &MainWindow::onExposureUpdated,
            Qt::QueuedConnection);

    // Connect camera frames to rosbag recorder
    if (rosbagRecorder_) {
        connect(cameraController_.get(), &CameraController::frameReady,
                rosbagRecorder_.get(), &RosbagRecorder::addCameraFrame,
                Qt::QueuedConnection);
    }

    // Initialize camera 0 (first CSI camera)
    if (!cameraController_->initialize(0)) {
        QMessageBox::critical(this, "Camera Error",
            "Failed to initialize camera 0.\n\n"
            "Make sure:\n"
            "- A CSI camera is connected\n"
            "- Camera is enabled in raspi-config\n"
            "- You have permissions (try running with sudo or add user to video group)");
        return;
    }

    // Apply exposure settings before starting (these are handled by libcamera)
    cameraController_->setAutoExposure(autoExposureEnabled_);
    if (!autoExposureEnabled_) {
        cameraController_->setExposureTime(manualExposureUs_);
        cameraController_->setGain(manualGain_);
    }

    // Start capturing
    if (!cameraController_->start()) {
        QMessageBox::critical(this, "Camera Error", "Failed to start camera");
        return;
    }

    // Apply trigger mode AFTER start() - libcamera's start() reinitializes the sensor,
    // so I2C register writes for trigger mode must come after.
    // Always do disable->enable sequence to ensure clean state transition,
    // as the sensor may need to go through free-run before entering trigger mode.
    if (triggerModeEnabled_) {
        cameraController_->setTriggerMode(false);  // First ensure free-run state
        QThread::msleep(50);  // Small delay for sensor state to settle
        cameraController_->setTriggerMode(true);   // Then enable trigger mode
    }

    qDebug() << "Camera initialized:" << cameraController_->cameraId()
             << cameraController_->frameSize();
}

void MainWindow::closeEvent(QCloseEvent *event)
{
    saveSettings();
    if (cameraController_) {
        cameraController_->stop();
    }
    event->accept();
}

void MainWindow::keyPressEvent(QKeyEvent *event)
{
    switch (event->key()) {
    case Qt::Key_Escape:
    case Qt::Key_Q:
        close();
        break;
        
    case Qt::Key_F:
    case Qt::Key_F11:
        // Toggle fullscreen
        if (isFullScreen()) {
            showNormal();
        } else {
            showFullScreen();
        }
        break;
        
    case Qt::Key_Space:
        // Toggle ROS bag recording
        if (recordRosbagAction_) {
            recordRosbagAction_->trigger();
        }
        break;
        
    default:
        QMainWindow::keyPressEvent(event);
    }
}

void MainWindow::onFpsUpdated(double fps, uint64_t frameCount)
{
    currentFps_ = fps;
    frameCount_ = frameCount;
    updateWindowTitle();
}

void MainWindow::onCameraError(const QString &error)
{
    qWarning() << "Camera error:" << error;
    QMessageBox::warning(this, "Camera Error", error);
}

void MainWindow::updateWindowTitle()
{
    QString title;

    if (cameraController_ && cameraController_->isRunning()) {
        QString modeStr = triggerModeEnabled_ ? " [TRIGGER]" : "";
        QString expStr;
        if (currentExposureUs_ > 0) {
            if (currentExposureUs_ >= 1000) {
                expStr = QString(" | Exp: %1ms").arg(currentExposureUs_ / 1000.0, 0, 'f', 1);
            } else {
                expStr = QString(" | Exp: %1us").arg(currentExposureUs_);
            }
        }
        title = QString("RPi Camera Viewer - FPS: %1 | Frames: %2%3%4")
                .arg(currentFps_, 0, 'f', 1)
                .arg(frameCount_)
                .arg(expStr)
                .arg(modeStr);
    } else {
        title = "RPi Camera Viewer - Stopped";
    }

    setWindowTitle(title);
}

void MainWindow::setupToolbar()
{
    toolbar_ = new QToolBar("Controls", this);
    toolbar_->setMovable(false);
    addToolBar(Qt::TopToolBarArea, toolbar_);

    // Trigger mode toggle action
    triggerModeAction_ = new QAction("Trigger Mode", this);
    triggerModeAction_->setCheckable(true);
    triggerModeAction_->setChecked(triggerModeEnabled_);
    triggerModeAction_->setToolTip("Enable external trigger mode for camera synchronization");

    connect(triggerModeAction_, &QAction::toggled,
            this, &MainWindow::onTriggerModeToggled);

    toolbar_->addAction(triggerModeAction_);
    toolbar_->addSeparator();

    // Auto exposure toggle action
    autoExposureAction_ = new QAction("Auto Exp", this);
    autoExposureAction_->setCheckable(true);
    autoExposureAction_->setChecked(autoExposureEnabled_);
    autoExposureAction_->setToolTip("Toggle between auto and manual exposure");

    connect(autoExposureAction_, &QAction::toggled,
            this, &MainWindow::onAutoExposureToggled);

    toolbar_->addAction(autoExposureAction_);

    // Exposure label
    exposureLabel_ = new QLabel(" Exp:", this);
    toolbar_->addWidget(exposureLabel_);

    // Exposure slider (100us to 33ms, logarithmic feel via slider range)
    exposureSlider_ = new QSlider(Qt::Horizontal, this);
    exposureSlider_->setMinimum(1);      // 100us (value * 100)
    exposureSlider_->setMaximum(330);    // 33000us = 33ms
    exposureSlider_->setValue(manualExposureUs_ / 100);
    exposureSlider_->setFixedWidth(150);
    exposureSlider_->setToolTip("Manual exposure time (100us - 33ms)");
    exposureSlider_->setEnabled(!autoExposureEnabled_);

    connect(exposureSlider_, &QSlider::valueChanged,
            this, &MainWindow::onExposureSliderChanged);

    toolbar_->addWidget(exposureSlider_);

    // Gain label
    gainLabel_ = new QLabel(" Gain:", this);
    toolbar_->addWidget(gainLabel_);

    // Gain slider (1.0 to 16.0)
    gainSlider_ = new QSlider(Qt::Horizontal, this);
    gainSlider_->setMinimum(10);     // 1.0 (value / 10)
    gainSlider_->setMaximum(160);    // 16.0
    gainSlider_->setValue(static_cast<int>(manualGain_ * 10));
    gainSlider_->setFixedWidth(100);
    gainSlider_->setToolTip("Manual gain (1.0 - 16.0)");
    gainSlider_->setEnabled(!autoExposureEnabled_);

    connect(gainSlider_, &QSlider::valueChanged,
            this, &MainWindow::onGainSliderChanged);

    toolbar_->addWidget(gainSlider_);
    toolbar_->addSeparator();

    // IMU serial port input
    QLabel *portLabel = new QLabel(" IMU Port:", this);
    toolbar_->addWidget(portLabel);

    serialPortEdit_ = new QLineEdit(this);
    serialPortEdit_->setText(serialPortName_);
    serialPortEdit_->setFixedWidth(120);
    serialPortEdit_->setToolTip("Serial port for IMU (e.g., /dev/ttyACM0)");
    toolbar_->addWidget(serialPortEdit_);

    // Connect IMU action
    connectIMUAction_ = new QAction("Connect IMU", this);
    connectIMUAction_->setCheckable(true);
    connectIMUAction_->setToolTip("Connect/disconnect IMU serial port");

    connect(connectIMUAction_, &QAction::triggered,
            this, &MainWindow::onConnectIMUClicked);

    toolbar_->addAction(connectIMUAction_);

    // Log IMU action
    logIMUAction_ = new QAction("Log IMU", this);
    logIMUAction_->setCheckable(true);
    logIMUAction_->setToolTip("Start/stop logging IMU data to file");

    connect(logIMUAction_, &QAction::triggered,
            this, &MainWindow::onLogIMUToggled);

    toolbar_->addAction(logIMUAction_);

    // SI Units toggle action
    siUnitsAction_ = new QAction("SI Units", this);
    siUnitsAction_->setCheckable(true);
    siUnitsAction_->setChecked(useSIUnits_);
    siUnitsAction_->setToolTip("Toggle SI units (rad/s, m/s²) vs default (°/s, g)");

    connect(siUnitsAction_, &QAction::triggered,
            this, &MainWindow::onSIUnitsToggled);

    toolbar_->addAction(siUnitsAction_);

    toolbar_->addSeparator();

    // Record Rosbag action
    recordRosbagAction_ = new QAction("Record Bag", this);
    recordRosbagAction_->setCheckable(true);
    recordRosbagAction_->setToolTip("Record camera and IMU data to ROS bag for Kalibr calibration");

    connect(recordRosbagAction_, &QAction::triggered,
            this, &MainWindow::onRecordRosbagToggled);

    toolbar_->addAction(recordRosbagAction_);

    // Setup rosbag recorder
    rosbagRecorder_ = std::make_unique<RosbagRecorder>(this);

    connect(rosbagRecorder_.get(), &RosbagRecorder::recordingStarted,
            this, &MainWindow::onRosbagRecordingStarted,
            Qt::QueuedConnection);

    connect(rosbagRecorder_.get(), &RosbagRecorder::recordingStopped,
            this, &MainWindow::onRosbagRecordingStopped,
            Qt::QueuedConnection);

    connect(rosbagRecorder_.get(), &RosbagRecorder::statusUpdated,
            this, &MainWindow::onRosbagStatusUpdated,
            Qt::QueuedConnection);

    connect(rosbagRecorder_.get(), &RosbagRecorder::errorOccurred,
            this, &MainWindow::onRosbagError,
            Qt::QueuedConnection);

    // Connect warning signals for frame skipping and duplicate triggers
    connect(rosbagRecorder_.get(), &RosbagRecorder::frameSkipped,
            this, [](const QString& reason) {
                qWarning() << "RosBag: Frame skipped -" << reason;
            }, Qt::QueuedConnection);

    connect(rosbagRecorder_.get(), &RosbagRecorder::duplicateTriggerWarning,
            this, [](uint64_t prevTs, uint64_t newTs) {
                qWarning() << "RosBag: Duplicate camera trigger detected!"
                           << "Previous:" << prevTs << "ms, New:" << newTs << "ms"
                           << "- Exposure may be too slow for frame rate";
            }, Qt::QueuedConnection);
}

void MainWindow::updateExposureSliderRange()
{
    // Update sliders enabled state based on auto exposure
    if (exposureSlider_) {
        exposureSlider_->setEnabled(!autoExposureEnabled_);
    }
    if (gainSlider_) {
        gainSlider_->setEnabled(!autoExposureEnabled_);
    }
}

void MainWindow::loadSettings()
{
    // Use INI format config file in user's config directory
    QString configPath = QStandardPaths::writableLocation(QStandardPaths::ConfigLocation);
    QString configFile = configPath + "/rpi_camera_viewer.ini";

    settings_ = std::make_unique<QSettings>(configFile, QSettings::IniFormat);

    // Load camera settings
    triggerModeEnabled_ = settings_->value("camera/trigger_mode", false).toBool();
    autoExposureEnabled_ = settings_->value("camera/auto_exposure", true).toBool();
    manualExposureUs_ = settings_->value("camera/manual_exposure_us", 10000).toInt();
    manualGain_ = settings_->value("camera/manual_gain", 8.0f).toFloat();

    // Load IMU settings
    serialPortName_ = settings_->value("imu/serial_port", "/dev/ttyACM0").toString();
    useSIUnits_ = settings_->value("imu/si_units", false).toBool();

    // Load UI settings
    windowWidth_ = settings_->value("ui/window_width", 1400).toInt();
    windowHeight_ = settings_->value("ui/window_height", 800).toInt();
    sidePanelWidth_ = settings_->value("ui/side_panel_width", 250).toInt();
    infoPanelFontSize_ = settings_->value("ui/info_panel_font_size", 8).toInt();

    qDebug() << "Loaded settings from:" << configFile;
    qDebug() << "Trigger mode:" << triggerModeEnabled_;
    qDebug() << "Auto exposure:" << autoExposureEnabled_;
    qDebug() << "Manual exposure:" << manualExposureUs_ << "us";
    qDebug() << "Manual gain:" << manualGain_;
    qDebug() << "IMU port:" << serialPortName_;
    qDebug() << "Window size:" << windowWidth_ << "x" << windowHeight_;
    qDebug() << "Side panel width:" << sidePanelWidth_;
    qDebug() << "Info panel font size:" << infoPanelFontSize_;
}

void MainWindow::saveSettings()
{
    if (!settings_) {
        return;
    }

    // Save camera settings
    settings_->setValue("camera/trigger_mode", triggerModeEnabled_);
    settings_->setValue("camera/auto_exposure", autoExposureEnabled_);
    settings_->setValue("camera/manual_exposure_us", manualExposureUs_);
    settings_->setValue("camera/manual_gain", static_cast<double>(manualGain_));

    // Save IMU settings
    if (serialPortEdit_) {
        serialPortName_ = serialPortEdit_->text().trimmed();
    }
    settings_->setValue("imu/serial_port", serialPortName_);
    settings_->setValue("imu/si_units", useSIUnits_);

    // Save UI settings (current window size)
    settings_->setValue("ui/window_width", width());
    settings_->setValue("ui/window_height", height());
    settings_->setValue("ui/side_panel_width", sidePanelWidth_);
    settings_->setValue("ui/info_panel_font_size", infoPanelFontSize_);

    settings_->sync();

    qDebug() << "Settings saved";
}

void MainWindow::onTriggerModeToggled(bool enabled)
{
    triggerModeEnabled_ = enabled;

    if (cameraController_) {
        cameraController_->setTriggerMode(enabled);
    }

    updateWindowTitle();
    qDebug() << "Trigger mode:" << (enabled ? "enabled" : "disabled");
}

void MainWindow::onAutoExposureToggled(bool enabled)
{
    autoExposureEnabled_ = enabled;

    if (cameraController_) {
        cameraController_->setAutoExposure(enabled);
        if (!enabled) {
            // Apply current manual exposure and gain values
            cameraController_->setExposureTime(manualExposureUs_);
            cameraController_->setGain(manualGain_);
        }
    }

    updateExposureSliderRange();
    qDebug() << "Auto exposure:" << (enabled ? "enabled" : "disabled");
}

void MainWindow::onExposureSliderChanged(int value)
{
    // Convert slider value to microseconds (slider value * 100)
    manualExposureUs_ = value * 100;

    if (cameraController_ && !autoExposureEnabled_) {
        cameraController_->setExposureTime(manualExposureUs_);
    }

    // Update tooltip with current value
    if (exposureSlider_) {
        QString tooltip;
        if (manualExposureUs_ >= 1000) {
            tooltip = QString("Exposure: %1ms").arg(manualExposureUs_ / 1000.0, 0, 'f', 1);
        } else {
            tooltip = QString("Exposure: %1us").arg(manualExposureUs_);
        }
        exposureSlider_->setToolTip(tooltip);
    }
}

void MainWindow::onGainSliderChanged(int value)
{
    // Convert slider value to gain (slider value / 10)
    manualGain_ = value / 10.0f;

    if (cameraController_ && !autoExposureEnabled_) {
        cameraController_->setGain(manualGain_);
    }

    // Update tooltip with current value
    if (gainSlider_) {
        QString tooltip = QString("Gain: %1").arg(manualGain_, 0, 'f', 1);
        gainSlider_->setToolTip(tooltip);
    }
}

void MainWindow::onExposureUpdated(int exposureUs)
{
    currentExposureUs_ = exposureUs;
    updateWindowTitle();
}

void MainWindow::setupIMU()
{
    imuReader_ = std::make_unique<SerialIMUReader>(this);

    // Connect IMU signals
    connect(imuReader_.get(), &SerialIMUReader::dataReceived,
            this, &MainWindow::onIMUDataReceived,
            Qt::QueuedConnection);

    connect(imuReader_.get(), &SerialIMUReader::errorOccurred,
            this, &MainWindow::onIMUError,
            Qt::QueuedConnection);

    connect(imuReader_.get(), &SerialIMUReader::connectionChanged,
            this, &MainWindow::onIMUConnectionChanged,
            Qt::QueuedConnection);

    // Connect IMU data to rosbag recorder
    if (rosbagRecorder_) {
        connect(imuReader_.get(), &SerialIMUReader::dataReceived,
                rosbagRecorder_.get(), &RosbagRecorder::addImuSample,
                Qt::QueuedConnection);

        // Connect camera trigger timestamp to rosbag recorder
        // This signal arrives BEFORE the camera frame from libcamera
        connect(imuReader_.get(), &SerialIMUReader::cameraTriggerReceived,
                rosbagRecorder_.get(), &RosbagRecorder::setCameraTriggerTimestamp,
                Qt::QueuedConnection);
    }

    qDebug() << "IMU reader initialized, default port:" << serialPortName_;
}

void MainWindow::onIMUDataReceived(const IMUData &data)
{
    lastIMUData_ = data;

    // Update GyroView with gyroscope vector
    if (gyroView_) {
        gyroView_->setDataVector(data.gyroVector());
    }

    // Update AccelView with accelerometer vector
    if (accelView_) {
        accelView_->setDataVector(data.accelVector());
    }

    // Log IMU data if logging is enabled
    if (isLogging_ && logStream_) {
        // Unit conversion constants
        constexpr float DEG_TO_RAD = 0.0174532925f;  // π/180
        constexpr float G_TO_MS2 = 9.80665f;

        if (useSIUnits_) {
            // Log in SI units: m/s² and rad/s
            *logStream_ << data.timestamp_ms << ","
                        << data.interrupt_count << ","
                        << (data.accel_x * G_TO_MS2) << ","
                        << (data.accel_y * G_TO_MS2) << ","
                        << (data.accel_z * G_TO_MS2) << ","
                        << (data.gyro_x * DEG_TO_RAD) << ","
                        << (data.gyro_y * DEG_TO_RAD) << ","
                        << (data.gyro_z * DEG_TO_RAD) << "\n";
        } else {
            // Log in default units: g and °/s
            *logStream_ << data.timestamp_ms << ","
                        << data.interrupt_count << ","
                        << data.accel_x << ","
                        << data.accel_y << ","
                        << data.accel_z << ","
                        << data.gyro_x << ","
                        << data.gyro_y << ","
                        << data.gyro_z << "\n";
        }
        logStream_->flush();
    }

    // Update InfoPanel
    updateInfoPanel();
}

void MainWindow::onIMUError(const QString &error)
{
    qWarning() << "IMU error:" << error;

    // Update info panel with error
    if (infoPanel_) {
        infoPanel_->append(QString("<font color='red'>IMU Error: %1</font>").arg(error));
    }
}

void MainWindow::onIMUConnectionChanged(bool connected)
{
    if (connectIMUAction_) {
        connectIMUAction_->setText(connected ? "Disconnect IMU" : "Connect IMU");
        connectIMUAction_->setChecked(connected);
    }

    if (serialPortEdit_) {
        serialPortEdit_->setEnabled(!connected);
    }

    qDebug() << "IMU connection:" << (connected ? "connected" : "disconnected");
    updateInfoPanel();
}

void MainWindow::onConnectIMUClicked()
{
    if (!imuReader_) {
        return;
    }

    if (imuReader_->isOpen()) {
        imuReader_->close();
    } else {
        if (serialPortEdit_) {
            serialPortName_ = serialPortEdit_->text().trimmed();
        }

        if (serialPortName_.isEmpty()) {
            onIMUError("Serial port name is empty");
            return;
        }

        if (!imuReader_->open(serialPortName_)) {
            // Error already emitted via signal
            if (connectIMUAction_) {
                connectIMUAction_->setChecked(false);
            }
        }
    }
}

void MainWindow::onLogIMUToggled(bool enabled)
{
    if (enabled) {
        // Create cam_logs directory next to executable
        QString appDir = QCoreApplication::applicationDirPath();
        QString logDir = appDir + "/cam_logs";

        QDir dir(logDir);
        if (!dir.exists()) {
            if (!dir.mkpath(".")) {
                onIMUError("Failed to create log directory: " + logDir);
                logIMUAction_->setChecked(false);
                return;
            }
        }

        // Create log file with timestamp
        QString timestamp = QDateTime::currentDateTime().toString("yyyy-MM-dd_HH-mm-ss");
        QString logPath = logDir + "/imu_log_" + timestamp + ".csv";

        logFile_ = std::make_unique<QFile>(logPath);
        if (!logFile_->open(QIODevice::WriteOnly | QIODevice::Text)) {
            onIMUError("Failed to create log file: " + logPath);
            logFile_.reset();
            logIMUAction_->setChecked(false);
            return;
        }

        logStream_ = std::make_unique<QTextStream>(logFile_.get());

        // Write CSV header with units
        if (useSIUnits_) {
            *logStream_ << "timestamp_ms,interrupt_count,accel_x_ms2,accel_y_ms2,accel_z_ms2,gyro_x_rads,gyro_y_rads,gyro_z_rads\n";
        } else {
            *logStream_ << "timestamp_ms,interrupt_count,accel_x_g,accel_y_g,accel_z_g,gyro_x_degs,gyro_y_degs,gyro_z_degs\n";
        }
        logStream_->flush();

        isLogging_ = true;
        logIMUAction_->setText("Stop Log");

        qDebug() << "IMU logging started:" << logPath;
    } else {
        // Stop logging
        isLogging_ = false;

        if (logStream_) {
            logStream_->flush();
            logStream_.reset();
        }

        if (logFile_) {
            logFile_->close();
            logFile_.reset();
        }

        logIMUAction_->setText("Log IMU");

        qDebug() << "IMU logging stopped";
    }

    updateInfoPanel();
}

void MainWindow::onSIUnitsToggled(bool enabled)
{
    useSIUnits_ = enabled;
    updateInfoPanel();
    qDebug() << "SI Units:" << (enabled ? "enabled (rad/s, m/s²)" : "disabled (°/s, g)");
}

void MainWindow::updateInfoPanel()
{
    if (!infoPanel_) {
        return;
    }

    QString info;

    // Frame timestamp
    info += QString("<b>Frame:</b> #%1<br>").arg(frameCount_);
    info += QString("<b>FPS:</b> %1<br>").arg(currentFps_, 0, 'f', 1);

    if (currentExposureUs_ > 0) {
        if (currentExposureUs_ >= 1000) {
            info += QString("<b>Exposure:</b> %1 ms<br>").arg(currentExposureUs_ / 1000.0, 0, 'f', 1);
        } else {
            info += QString("<b>Exposure:</b> %1 us<br>").arg(currentExposureUs_);
        }
    }

    info += "<hr>";

    // IMU data
    if (imuReader_ && imuReader_->isOpen()) {
        info += QString("<b>IMU Time:</b> %1 ms<br>").arg(lastIMUData_.timestamp_ms);
        info += QString("<b>INT Count:</b> %1<br>").arg(lastIMUData_.interrupt_count);

        if (isLogging_) {
            info += "<b><font color='#00ff00'>LOGGING CSV</font></b><br>";
        }
        if (rosbagRecorder_ && rosbagRecorder_->isRecording()) {
            info += QString("<b><font color='#ff8800'>RECORDING BAG</font></b><br>");
            info += QString("Frames: %1 | IMU: %2<br>")
                    .arg(rosbagRecorder_->frameCount())
                    .arg(rosbagRecorder_->imuCount());
        }
        info += "<br>";

        // Unit conversion constants
        constexpr float DEG_TO_RAD = 0.0174532925f;  // π/180
        constexpr float G_TO_MS2 = 9.80665f;

        if (useSIUnits_) {
            // SI units: m/s² and rad/s
            float ax = lastIMUData_.accel_x * G_TO_MS2;
            float ay = lastIMUData_.accel_y * G_TO_MS2;
            float az = lastIMUData_.accel_z * G_TO_MS2;

            float gx = lastIMUData_.gyro_x * DEG_TO_RAD;
            float gy = lastIMUData_.gyro_y * DEG_TO_RAD;
            float gz = lastIMUData_.gyro_z * DEG_TO_RAD;

            info += QString("<b>Accel (m/s²):</b><br>");
            info += QString("  X: %1<br>").arg(ax, 7, 'f', 2);
            info += QString("  Y: %1<br>").arg(ay, 7, 'f', 2);
            info += QString("  Z: %1<br>").arg(az, 7, 'f', 2);
            info += "<br>";

            info += QString("<b>Gyro (rad/s):</b><br>");
            info += QString("  X: %1<br>").arg(gx, 7, 'f', 3);
            info += QString("  Y: %1<br>").arg(gy, 7, 'f', 3);
            info += QString("  Z: %1<br>").arg(gz, 7, 'f', 3);
        } else {
            // Default units: g and °/s
            info += QString("<b>Accel (g):</b><br>");
            info += QString("  X: %1<br>").arg(lastIMUData_.accel_x, 7, 'f', 3);
            info += QString("  Y: %1<br>").arg(lastIMUData_.accel_y, 7, 'f', 3);
            info += QString("  Z: %1<br>").arg(lastIMUData_.accel_z, 7, 'f', 3);
            info += "<br>";

            info += QString("<b>Gyro (°/s):</b><br>");
            info += QString("  X: %1<br>").arg(lastIMUData_.gyro_x, 7, 'f', 1);
            info += QString("  Y: %1<br>").arg(lastIMUData_.gyro_y, 7, 'f', 1);
            info += QString("  Z: %1<br>").arg(lastIMUData_.gyro_z, 7, 'f', 1);
        }
    } else {
        info += "<i>IMU not connected</i><br>";
        info += QString("Port: %1").arg(serialPortName_);
    }

    infoPanel_->setHtml(info);
}

// ============================================================================
// Rosbag Recording Slots
// ============================================================================

void MainWindow::onRecordRosbagToggled(bool enabled)
{
    if (!rosbagRecorder_) {
        return;
    }

    if (enabled) {
        // Create rosbag_logs directory next to executable
        QString appDir = QCoreApplication::applicationDirPath();
        QString logDir = appDir + "/rosbag_logs";

        QDir dir(logDir);
        if (!dir.exists()) {
            if (!dir.mkpath(".")) {
                onRosbagError("Failed to create log directory: " + logDir);
                recordRosbagAction_->setChecked(false);
                return;
            }
        }

        // Create bag file with timestamp
        QString timestamp = QDateTime::currentDateTime().toString("yyyy-MM-dd_HH-mm-ss");
        QString bagPath = logDir + "/kalibr_data_" + timestamp + ".bag";

        if (!rosbagRecorder_->startRecording(bagPath)) {
            recordRosbagAction_->setChecked(false);
        }
    } else {
        rosbagRecorder_->stopRecording();
    }
}

void MainWindow::onRosbagRecordingStarted(const QString& filePath)
{
    if (recordRosbagAction_) {
        recordRosbagAction_->setText("Stop Bag");
        recordRosbagAction_->setChecked(true);
    }

    qDebug() << "Rosbag recording started:" << filePath;
    updateInfoPanel();
}

void MainWindow::onRosbagRecordingStopped(const QString& filePath, uint64_t frameCount, uint64_t imuCount)
{
    if (recordRosbagAction_) {
        recordRosbagAction_->setText("Record Bag");
        recordRosbagAction_->setChecked(false);
    }

    QString msg = QString("Rosbag saved:\n%1\n\nFrames: %2\nIMU samples: %3")
                  .arg(filePath)
                  .arg(frameCount)
                  .arg(imuCount);

    qDebug() << msg;

    QMessageBox::information(this, "Rosbag Recording Complete", msg);
    updateInfoPanel();
}

void MainWindow::onRosbagStatusUpdated(uint64_t frameCount, uint64_t imuCount, uint64_t fileSizeBytes)
{
    // Status updates happen during recording - just update the info panel
    updateInfoPanel();
}

void MainWindow::onRosbagError(const QString& error)
{
    qWarning() << "Rosbag error:" << error;

    if (recordRosbagAction_) {
        recordRosbagAction_->setText("Record Bag");
        recordRosbagAction_->setChecked(false);
    }

    QMessageBox::warning(this, "Rosbag Error", error);
}
