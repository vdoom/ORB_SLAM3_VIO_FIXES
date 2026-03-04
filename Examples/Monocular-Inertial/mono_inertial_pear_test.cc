/**
 * @file mono_inertial_pear_test.cc
 * @brief Monocular-Inertial VIO with PearAPI - Test/Visualization Only (No MAVLink)
 *
 * Same VIO pipeline as mono_inertial_pear.cc but without MAVLink/ArduPilot
 * integration. Useful for testing VIO performance with Pangolin visualization.
 *
 * Hardware:
 * - OV9281 global shutter camera (via libcamera / PearAPI)
 * - BMI160 IMU via Raspberry Pi Pico (hardware-triggered camera)
 *
 * Usage: ./mono_inertial_pear_test path_to_vocabulary path_to_settings
 *            [imu_serial_port] [visualization] [--autogain] [--record]
 *
 *   Defaults: /dev/ttyACM0, 1 (visualization ON)
 */

#include <signal.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <ctime>
#include <sstream>
#include <thread>
#include <atomic>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <Eigen/Dense>

// PearAPI for camera and IMU access
#include <PearAPI/PearAPI.h>

#include <System.h>
#include <FrameDrawer.h>

// I2C includes for OV9281 trigger mode control
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>
#include <errno.h>

using namespace std;

//=============================================================================
// Global state
//=============================================================================

atomic<bool> b_continue_session{true};

void exit_loop_handler(int s) {
    cout << "\nFinishing session..." << endl;
    b_continue_session = false;
}

//=============================================================================
// Main Function
//=============================================================================

int main(int argc, char** argv) {
    // Extract flags first
    bool enable_autogain = false;
    bool enable_video_recording = false;
    std::vector<std::string> positional_args;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--autogain") {
            enable_autogain = true;
        } else if (arg == "--record") {
            enable_video_recording = true;
        } else {
            positional_args.push_back(arg);
        }
    }

    int pargc = positional_args.size();
    if (pargc < 2 || pargc > 4) {
        cerr << endl
             << "Usage: ./mono_inertial_pear_test path_to_vocabulary path_to_settings"
             << endl
             << "           [imu_serial_port] [visualization] [--autogain] [--record]"
             << endl
             << endl
             << "  Defaults:"
             << endl
             << "    imu_serial_port: /dev/ttyACM0"
             << endl
             << "    visualization:   1 = ON (default), 0 = OFF"
             << endl
             << "    --autogain:      Run auto gain tuning on startup"
             << endl
             << "    --record:        Record feature tracking video to current directory"
             << endl;
        return 1;
    }

    string vocabularyPath = positional_args[0];
    string settingsPath = positional_args[1];
    string imuSerialPort = (pargc >= 3) ? positional_args[2] : "/dev/ttyACM0";

    // Visualization ON by default for testing
    bool enable_visualization = true;
    if (pargc >= 4) {
        int vis_val = atoi(positional_args[3].c_str());
        enable_visualization = (vis_val != 0);
    }

    // Setup signal handler
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = exit_loop_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);

    cout << "========================================" << endl;
    cout << "Monocular-Inertial VIO Test (PearAPI)" << endl;
    cout << "  (No MAVLink - visualization only)" << endl;
    cout << "========================================" << endl;
    cout << "Vocabulary:      " << vocabularyPath << endl;
    cout << "Settings:        " << settingsPath << endl;
    cout << "IMU Port:        " << imuSerialPort << endl;
    cout << "Visualization:   " << (enable_visualization ? "ON" : "OFF") << endl;
    cout << "Auto Gain:       " << (enable_autogain ? "ON" : "OFF") << endl;
    cout << "Video Record:    " << (enable_video_recording ? "ON" : "OFF") << endl;
    cout << "========================================" << endl;

    // ---- Initialize IMU reader using PearAPI ----
    pearvio::IMUReader imuReader;
    if (!imuReader.open(imuSerialPort)) {
        cerr << "Failed to open IMU serial port: " << imuSerialPort << endl;
        cerr << "Check that:" << endl;
        cerr << "  - Pico is connected and running" << endl;
        cerr << "  - User has dialout group access: sudo usermod -aG dialout $USER" << endl;
        return 1;
    }

    // ---- Initialize camera using PearAPI ----
    auto camera = pearvio::CameraBackend::create();
    if (!camera) {
        cerr << "Failed to create camera backend" << endl;
        return 1;
    }

    pearvio::CameraConfig camConfig;
    camConfig.cameraIndex = 0;
    camConfig.fps = 20;  // 20 FPS for triggered mode

    // Load camera settings from PearCameraApp config (resolution, exposure, gain, etc.)
    if (camConfig.loadFromIniFile()) {
        cout << "Loaded camera settings from config file" << endl;
        cout << "  Resolution: " << camConfig.width << "x" << camConfig.height << endl;
    } else {
        cout << "No config file found, using defaults ("
             << camConfig.width << "x" << camConfig.height << ")" << endl;
    }

    // VIO always needs trigger mode for hardware sync
    camConfig.triggerMode = true;

    if (!camera->initialize(camConfig)) {
        cerr << "Failed to initialize camera" << endl;
        return 1;
    }

    cout << "Camera initialized: " << camera->frameWidth() << "x" << camera->frameHeight() << endl;

    if (!camera->start()) {
        cerr << "Failed to start camera" << endl;
        return 1;
    }

    // Apply camera settings AFTER start() - start() reinitializes the sensor,
    // so V4L2 control writes must come after to avoid being reset.
    // This matches the sequence used in PearCameraApp (MainWindow.cpp).
    camera->setTriggerMode(camConfig.triggerMode);
    camera->setAutoExposure(camConfig.autoExposure);
    if (!camConfig.autoExposure) {
        camera->setGain(camConfig.gain);
        this_thread::sleep_for(chrono::milliseconds(50));
        camera->setExposureTime(camConfig.exposureTimeUs);
    }
    cout << "Camera settings applied: trigger=" << camConfig.triggerMode
         << " gain=" << camera->gain()
         << " exposure=" << camera->exposureTime() << "us" << endl;

    // ---- Auto-tune gain if requested via --autogain flag ----
    if (enable_autogain) {
        cout << "Running auto gain..." << endl;
        pearvio::AutoGainConfig agc;
        auto result = camera->autoGain(agc);
        if (result.success) {
            cout << "Auto gain: " << result.gain << " (brightness=" << result.brightness
                 << ", iterations=" << result.iterations << ")" << endl;
            camConfig.gain = result.gain;
            camConfig.autoExposure = false;
            camConfig.saveToIniFile();
        } else {
            cout << "Auto gain failed, using current gain: " << camera->gain() << endl;
        }
    }

    // ---- Wait for first IMU data to establish time base ----
    cout << "Waiting for IMU data..." << endl;
    uint64_t firstImuTimestamp = 0;
    while (b_continue_session && firstImuTimestamp == 0) {
        auto imuData = imuReader.getIMUData();
        if (!imuData.empty()) {
            firstImuTimestamp = imuData.front().timestamp_ms;
            cout << "First IMU timestamp: " << firstImuTimestamp << " ms" << endl;

            const auto& first = imuData.front();
            cout << "First IMU reading (raw):" << endl;
            cout << "  Accel: [" << first.accel_x() << ", " << first.accel_y() << ", " << first.accel_z() << "] g" << endl;
            cout << "  Gyro:  [" << first.gyro_x() << ", " << first.gyro_y() << ", " << first.gyro_z() << "] deg/s" << endl;

            float ax, ay, az;
            first.getAccelSI(ax, ay, az);
            cout << "  Accel (SI): [" << ax << ", " << ay << ", " << az << "] m/s^2" << endl;
        }
        this_thread::sleep_for(chrono::milliseconds(10));
    }

    if (!b_continue_session) {
        cout << "Interrupted before initialization" << endl;
        return 0;
    }

    // ---- Create SLAM system (video recording is handled directly in main loop) ----
    cout << "Creating ORB-SLAM3 system..." << endl;
    ORB_SLAM3::System SLAM(vocabularyPath, settingsPath, ORB_SLAM3::System::IMU_MONOCULAR, enable_visualization);
    float imageScale = SLAM.GetImageScale();

    // Half exposure time offset
    double halfExposureTimeSec = camera->exposureTime() / 2.0 / 1e6;
    cout << "Exposure time: " << camera->exposureTime() << " us" << endl;
    cout << "Half exposure time offset: " << halfExposureTimeSec * 1000.0 << " ms" << endl;

    // Read Camera.TimeOffset from Kalibr calibration (camera-IMU time offset)
    double cameraTimeOffset = 0.0;
    {
        cv::FileStorage fSettings(settingsPath, cv::FileStorage::READ);
        cv::FileNode node = fSettings["Camera.TimeOffset"];
        if (!node.empty()) {
            cameraTimeOffset = static_cast<double>(node);
            cout << "Camera.TimeOffset from Kalibr: " << cameraTimeOffset * 1000.0 << " ms" << endl;
        }
        fSettings.release();
    }
    cout << "Total frame time offset: " << (halfExposureTimeSec + cameraTimeOffset) * 1000.0 << " ms" << endl;

    // Clear IMU buffer accumulated during SLAM initialization (loading vocabulary takes time)
    cout << "Clearing IMU buffer accumulated during initialization..." << endl;
    imuReader.getIMUData();  // Discard accumulated IMU data
    imuReader.clearTriggerQueue();  // Discard ALL pending camera triggers

    // Wait for fresh IMU data to establish new time base
    firstImuTimestamp = 0;
    while (b_continue_session && firstImuTimestamp == 0) {
        auto imuData = imuReader.getIMUData();
        if (!imuData.empty()) {
            firstImuTimestamp = imuData.front().timestamp_ms;
            cout << "Reset time base to: " << firstImuTimestamp << " ms" << endl;
        }
        this_thread::sleep_for(chrono::milliseconds(5));
    }

    // Discard first few camera frames to ensure they have timestamps after our time base
    cout << "Waiting for camera frames with valid timestamps..." << endl;
    for (int i = 0; i < 5 && b_continue_session; i++) {
        cv::Mat frame;
        double ts;
        camera->getFrame(frame, ts);
        imuReader.getIMUData();  // Discard IMU
    }

    // Final cleanup: clear any remaining old triggers that accumulated during init
    imuReader.clearTriggerQueue();
    imuReader.getIMUData();  // Discard any remaining IMU data

    cout << "VIO system ready. Press Ctrl+C to exit." << endl;
    cout << "========================================" << endl;

    // ---- Initialize video recorder if requested ----
    cv::VideoWriter videoWriter;
    uint64_t videoFrameCount = 0;
    string videoFilePath;
    if (enable_video_recording) {
        // Generate timestamped filename
        auto now = chrono::system_clock::now();
        auto time_t_now = chrono::system_clock::to_time_t(now);
        struct tm tm_now;
        localtime_r(&time_t_now, &tm_now);
        ostringstream oss;
        oss << put_time(&tm_now, "%Y-%m-%d_%H-%M-%S");
        videoFilePath = "VIO_" + oss.str() + ".mkv";

        // VideoWriter is opened on first frame when we know the actual resolution
        cout << "Video recording enabled, will save to: " << videoFilePath << endl;
    }

    // Unit conversion constants
    constexpr float DEG_TO_RAD = 0.0174532925f;  // pi/180
    constexpr float G_TO_MS2 = 9.80665f;

    // IMU outlier rejection thresholds
    constexpr float MAX_GYRO_DEG_S = 300.0f;    // Reject gyro readings above 300 deg/s
    constexpr float MAX_ACCEL_G = 6.0f;          // Reject accel readings above 6g
    constexpr float MIN_ACCEL_G = 0.1f;          // Reject accel readings below 0.1g (corrupt data)
    uint64_t imuRejectedCount = 0;

    // IMU measurements for current frame
    vector<ORB_SLAM3::IMU::Point> vImuMeas;

    // Statistics
    uint64_t frameCount = 0;
    uint64_t imuCount = 0;
    double totalTrackTimeMs = 0;
    auto startTime = chrono::steady_clock::now();

    // Track previous IMU timestamp for gap detection
    double lastImuTimestamp = 0;
    int imuGapWarnings = 0;

    // Tracking state counters
    uint32_t tracking_ok_count = 0;
    uint32_t tracking_lost_count = 0;
    uint32_t tracking_recently_lost_count = 0;

    // ---- Main loop ----
    while (!SLAM.isShutDown() && b_continue_session) {
        // Get camera frame FIRST (this blocks until frame is ready)
        cv::Mat frame;
        double cameraTimestamp;

        if (!camera->getFrame(frame, cameraTimestamp)) {
            continue;
        }

        // Get Pico trigger timestamp
        // When processing is slower than camera rate, triggers accumulate in queue.
        // Drain ALL triggers and use only the LAST one to match the current frame.
        uint64_t triggerTimestamp = 0;
        int droppedTriggers = 0;
        while (imuReader.hasTriggerTimestamp()) {
            uint64_t ts = imuReader.getCameraTriggerTimestamp();
            if (triggerTimestamp > 0) {
                droppedTriggers++;
            }
            triggerTimestamp = ts;
        }
        if (droppedTriggers > 0 && frameCount < 50) {
            cout << "Note: Skipped " << droppedTriggers << " triggers (dropped frames during processing)" << endl;
        }

        // Check for invalid trigger (before our time base)
        if (triggerTimestamp > 0 && triggerTimestamp < firstImuTimestamp) {
            if (frameCount < 20) {
                cout << "Skipping frame with old trigger timestamp: " << triggerTimestamp
                     << " < " << firstImuTimestamp << endl;
            }
            continue;
        }

        // NOW get all IMU data (after camera frame, so we have data up to trigger time)
        auto imuData = imuReader.getIMUData();

        for (const auto& imu : imuData) {
            // Convert to SI units and create IMU::Point
            double t = (imu.timestamp_ms - firstImuTimestamp) / 1000.0;

            // Check for gaps in IMU stream
            if (lastImuTimestamp > 0 && t > lastImuTimestamp) {
                double gap = t - lastImuTimestamp;
                if (gap > 0.005 && imuGapWarnings < 10) {  // More than 5ms gap
                    cout << "WARNING: IMU gap detected: " << fixed << setprecision(1)
                         << gap * 1000 << "ms at t=" << setprecision(3) << t << endl;
                    imuGapWarnings++;
                }
            }
            lastImuTimestamp = t;

            // IMU outlier rejection: reject clearly corrupt samples
            float gyroMag = std::sqrt(imu.gyro_x() * imu.gyro_x() +
                                      imu.gyro_y() * imu.gyro_y() +
                                      imu.gyro_z() * imu.gyro_z());
            float accelMag = std::sqrt(imu.accel_x() * imu.accel_x() +
                                       imu.accel_y() * imu.accel_y() +
                                       imu.accel_z() * imu.accel_z());

            if (gyroMag > MAX_GYRO_DEG_S || accelMag > MAX_ACCEL_G || accelMag < MIN_ACCEL_G) {
                imuRejectedCount++;
                if (imuRejectedCount <= 20) {
                    cout << "WARNING: IMU outlier rejected at t=" << fixed << setprecision(3) << t
                         << " gyro=" << setprecision(1) << gyroMag << " deg/s"
                         << " accel=" << setprecision(2) << accelMag << " g" << endl;
                }
                continue;  // Skip this sample
            }

            // Convert: gyro from deg/s to rad/s, accel from g to m/s^2
            float ax = imu.accel_x() * G_TO_MS2;
            float ay = imu.accel_y() * G_TO_MS2;
            float az = imu.accel_z() * G_TO_MS2;
            float gx = imu.gyro_x() * DEG_TO_RAD;
            float gy = imu.gyro_y() * DEG_TO_RAD;
            float gz = imu.gyro_z() * DEG_TO_RAD;

            ORB_SLAM3::IMU::Point pt(ax, ay, az, gx, gy, gz, t);
            vImuMeas.push_back(pt);
            imuCount++;
        }

        // Determine frame time
        double frameTime;
        if (triggerTimestamp > 0 && triggerTimestamp >= firstImuTimestamp) {
            frameTime = ((triggerTimestamp - firstImuTimestamp) / 1000.0) + halfExposureTimeSec + cameraTimeOffset;
        } else {
            if (!vImuMeas.empty()) {
                frameTime = vImuMeas.back().t;
                if (frameCount < 20) {
                    cout << "No trigger, using last IMU time: " << fixed << setprecision(3) << frameTime << endl;
                }
            } else {
                if (frameCount < 20) {
                    cout << "Skipping frame: no trigger and no IMU data" << endl;
                }
                continue;
            }
        }

        // Filter IMU measurements: only keep those with timestamp <= frameTime
        vector<ORB_SLAM3::IMU::Point> vImuForFrame;
        vector<ORB_SLAM3::IMU::Point> vImuForNext;

        for (const auto& pt : vImuMeas) {
            if (pt.t <= frameTime) {
                vImuForFrame.push_back(pt);
            } else {
                vImuForNext.push_back(pt);
            }
        }

        // Debug: Print timing info for first 20 frames
        if (frameCount < 20) {
            cout << "Frame " << frameCount << ": t=" << fixed << setprecision(3) << frameTime
                 << " IMU=" << vImuForFrame.size() << "/" << vImuMeas.size();
            if (!vImuForFrame.empty()) {
                cout << " range=[" << vImuForFrame.front().t << "," << vImuForFrame.back().t << "]";
                double imuFrameGap = frameTime - vImuForFrame.back().t;
                if (imuFrameGap > 0.005) {
                    cout << " GAP=" << setprecision(0) << imuFrameGap * 1000 << "ms";
                }
            }
            cout << " trigger=" << (triggerTimestamp > 0 ? "yes" : "NO");
            if (!vImuForNext.empty()) {
                cout << " kept=" << vImuForNext.size();
            }
            cout << endl;
        }

        // Keep IMU data after frame time for next iteration
        vImuMeas = std::move(vImuForNext);

        // Need at least some IMU data to track
        if (vImuForFrame.empty()) {
            if (frameCount < 20) {
                cout << "  -> Skipping frame: no IMU data" << endl;
            }
            continue;
        }

        // Resize if needed
        if (imageScale != 1.0f) {
            int newWidth = static_cast<int>(frame.cols * imageScale);
            int newHeight = static_cast<int>(frame.rows * imageScale);
            cv::resize(frame, frame, cv::Size(newWidth, newHeight));
        }

        // Track with ORB-SLAM3 monocular-inertial
        auto t_track_start = chrono::steady_clock::now();
        Sophus::SE3f Tcw = SLAM.TrackMonocular(frame, frameTime, vImuForFrame);
        auto t_track_end = chrono::steady_clock::now();
        double trackTimeMs = chrono::duration<double, milli>(t_track_end - t_track_start).count();

        // Get velocity and tracking state from ORB-SLAM3
        Eigen::Vector3f velocity = SLAM.GetVelocity();
        auto tracking_state = SLAM.GetTrackingState();

        totalTrackTimeMs += trackTimeMs;

        // Record annotated frame (with tracking markers) to video
        if (enable_video_recording) {
            cv::Mat videoFrame = SLAM.GetFrameDrawer()->DrawFrame(imageScale);

            if (!videoFrame.empty()) {
                // Ensure even dimensions
                int vw = videoFrame.cols & ~1;
                int vh = videoFrame.rows & ~1;
                if (vw != videoFrame.cols || vh != videoFrame.rows)
                    videoFrame = videoFrame(cv::Rect(0, 0, vw, vh)).clone();

                if (!videoWriter.isOpened()) {
                    string gstPipeline = "appsrc ! videoconvert ! video/x-raw,format=I420 ! "
                                         "x264enc tune=zerolatency bitrate=2000 ! "
                                         "matroskamux ! filesink location=" + videoFilePath;
                    videoWriter.open(gstPipeline, cv::CAP_GSTREAMER, 0, 20.0,
                                     cv::Size(vw, vh), true);
                    if (videoWriter.isOpened()) {
                        cout << "Video recording started: " << videoFilePath
                             << " (" << vw << "x" << vh << ", GStreamer x264)" << endl;
                    } else {
                        videoFilePath = videoFilePath.substr(0, videoFilePath.rfind('.')) + ".avi";
                        int fourcc = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
                        videoWriter.open(videoFilePath, fourcc, 20.0,
                                         cv::Size(vw, vh), true);
                        if (videoWriter.isOpened()) {
                            cout << "Video recording started: " << videoFilePath
                                 << " (" << vw << "x" << vh << ", FFmpeg XVID)" << endl;
                        } else {
                            cerr << "WARNING: Could not open video writer, recording disabled" << endl;
                            enable_video_recording = false;
                        }
                    }
                }
                if (videoWriter.isOpened()) {
                    videoWriter.write(videoFrame);
                    videoFrameCount++;
                }
            }
        }

        // Count tracking states
        switch (tracking_state) {
            case ORB_SLAM3::Tracking::OK:
            case ORB_SLAM3::Tracking::OK_KLT:
                tracking_ok_count++;
                break;
            case ORB_SLAM3::Tracking::RECENTLY_LOST:
                tracking_recently_lost_count++;
                break;
            case ORB_SLAM3::Tracking::LOST:
                tracking_lost_count++;
                break;
        }

        // Log pose when tracking is OK
        if (tracking_state == ORB_SLAM3::Tracking::OK ||
            tracking_state == ORB_SLAM3::Tracking::OK_KLT) {

            Sophus::SE3f Twc = Tcw.inverse();
            Eigen::Vector3f position = Twc.translation();
            Eigen::Quaternionf q = Twc.unit_quaternion();

            // Log every 30 frames (roughly once per second at 30fps)
            if (frameCount % 30 == 0) {
                cout << "[VIO] pos=(" << fixed << setprecision(3)
                     << position.x() << ", " << position.y() << ", " << position.z()
                     << ") vel=(" << setprecision(2)
                     << velocity.x() << ", " << velocity.y() << ", " << velocity.z()
                     << ") q=(" << setprecision(3)
                     << q.w() << ", " << q.x() << ", " << q.y() << ", " << q.z()
                     << ")" << endl;
            }
        }

        frameCount++;

        // Print IMU diagnostics every 50 frames
        if (frameCount % 50 == 0 && !vImuForFrame.empty()) {
            float maxGyro = 0;
            float avgAccelMag = 0;
            for (const auto& pt : vImuForFrame) {
                float gyroMag = pt.w.norm();
                if (gyroMag > maxGyro) maxGyro = gyroMag;
                avgAccelMag += pt.a.norm();
            }
            avgAccelMag /= vImuForFrame.size();
            cout << "IMU check: maxGyro=" << fixed << setprecision(2) << maxGyro * 57.3 << " deg/s"
                 << " | accelMag=" << setprecision(2) << avgAccelMag << " m/s^2"
                 << " (expect ~9.8 stationary)" << endl;
        }

        // Print statistics every 100 frames
        if (frameCount % 100 == 0) {
            auto now = chrono::steady_clock::now();
            double elapsed = chrono::duration<double>(now - startTime).count();
            double avgFps = frameCount / elapsed;

            // Instantaneous FPS (over the last 100 frames)
            static auto lastStatTime = startTime;
            static uint64_t lastStatFrameCount = 0;
            double windowElapsed = chrono::duration<double>(now - lastStatTime).count();
            double instantFps = (windowElapsed > 0) ? (frameCount - lastStatFrameCount) / windowElapsed : avgFps;
            lastStatTime = now;
            lastStatFrameCount = frameCount;

            const char* stateStr = "UNKNOWN";
            switch (tracking_state) {
                case -1: stateStr = "NOT_READY"; break;
                case 0: stateStr = "NO_IMAGES"; break;
                case 1: stateStr = "INIT"; break;
                case 2: stateStr = "OK"; break;
                case 3: stateStr = "RECENTLY_LOST"; break;
                case 4: stateStr = "LOST"; break;
                case 5: stateStr = "OK_KLT"; break;
            }

            double avgTrackMs = totalTrackTimeMs / frameCount;
            cout << "Frames: " << frameCount
                 << " | IMU: " << imuCount
                 << " | FPS: " << fixed << setprecision(1) << instantFps
                 << " (avg: " << avgFps << ")"
                 << " | Track: " << setprecision(0) << avgTrackMs << "ms"
                 << " | State: " << stateStr
                 << " | OK: " << tracking_ok_count
                 << " Lost: " << tracking_lost_count;
            if (imuRejectedCount > 0)
                cout << " | IMU rejected: " << imuRejectedCount;
            cout << " | Time: " << fixed << setprecision(1) << elapsed << "s"
                 << endl;
        }
    }

    // ---- Cleanup ----
    cout << endl << "Shutting down..." << endl;

    // Close video writer
    if (videoWriter.isOpened()) {
        videoWriter.release();
        cout << "Video saved: " << videoFilePath << " (" << videoFrameCount << " frames)" << endl;
    }

    // Disable trigger mode before stopping camera
    camera->setTriggerMode(false);

    camera->stop();
    imuReader.close();

    SLAM.Shutdown();

    // Final statistics
    auto endTime = chrono::steady_clock::now();
    double totalTime = chrono::duration<double>(endTime - startTime).count();

    cout << "========================================" << endl;
    cout << "Session complete" << endl;
    cout << "Total frames:      " << frameCount << endl;
    cout << "Total IMU:         " << imuCount << endl;
    cout << "Duration:          " << fixed << setprecision(1) << totalTime << " s" << endl;
    cout << "Average FPS:       " << fixed << setprecision(1) << (frameCount / totalTime) << endl;
    cout << "Tracking OK:       " << tracking_ok_count << endl;
    cout << "Recently lost:     " << tracking_recently_lost_count << endl;
    cout << "Tracking lost:     " << tracking_lost_count << endl;
    cout << "IMU rejected:      " << imuRejectedCount << endl;
    cout << "========================================" << endl;

    // Save trajectory
    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
    cout << "Trajectories saved to CameraTrajectory.txt and KeyFrameTrajectory.txt" << endl;

    return 0;
}
