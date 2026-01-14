
/**
 * Monocular-Inertial ORB-SLAM3 for Custom Camera (OV9281 + BMI160)
 *
 * Based on mono_inertial_realsense_D435i.cc but adapted for:
 * - PearAPI for camera and IMU access
 * - BMI160 IMU via Raspberry Pi Pico over serial
 * - Hardware-synchronized camera trigger (Pico triggers camera every 10 IMU samples)
 * - Both accel and gyro at 200Hz (no interpolation needed)
 *
 * Usage: ./mono_inertial_custom_cam path_to_vocabulary path_to_settings [serial_port]
 *        Default serial port: /dev/ttyACM0
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

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// PearAPI for camera and IMU access
#include <PearAPI/PearAPI.h>

#include <System.h>

// I2C includes for OV9281 trigger mode control
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>
#include <cstring>
#include <errno.h>

using namespace std;

// ============================================================================
// Global state
// ============================================================================

atomic<bool> b_continue_session{true};

void exit_loop_handler(int s) {
    cout << "\nFinishing session..." << endl;
    b_continue_session = false;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        cerr << endl
             << "Usage: ./mono_inertial_custom_cam path_to_vocabulary path_to_settings [serial_port]"
             << endl
             << "  Default serial_port: /dev/ttyACM0"
             << endl;
        return 1;
    }

    string vocabularyPath = argv[1];
    string settingsPath = argv[2];
    string serialPort = (argc == 4) ? argv[3] : "/dev/ttyACM0";

    // Setup signal handler
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = exit_loop_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);

    cout << "========================================" << endl;
    cout << "Monocular-Inertial VIO (Custom Camera)" << endl;
    cout << "  Using PearAPI" << endl;
    cout << "========================================" << endl;
    cout << "Vocabulary: " << vocabularyPath << endl;
    cout << "Settings:   " << settingsPath << endl;
    cout << "IMU Port:   " << serialPort << endl;
    cout << "========================================" << endl;

    // Initialize IMU reader using PearAPI
    pearvio::IMUReader imuReader;
    if (!imuReader.open(serialPort)) {
        cerr << "Failed to open IMU serial port: " << serialPort << endl;
        cerr << "Check that:" << endl;
        cerr << "  - Pico is connected and running" << endl;
        cerr << "  - User has dialout group access: sudo usermod -aG dialout $USER" << endl;
        return 1;
    }

    // Initialize camera using PearAPI
    auto camera = pearvio::CameraBackend::create();
    if (!camera) {
        cerr << "Failed to create camera backend" << endl;
        return 1;
    }

    pearvio::CameraConfig camConfig;
    camConfig.cameraIndex = 0;
    camConfig.width = 640;
    camConfig.height = 400;
    camConfig.fps = 20;  // 20 FPS for triggered mode

    // Load camera settings from PearCameraApp config (exposure, gain, etc.)
    if (camConfig.loadFromIniFile()) {
        cout << "Loaded camera settings from config file" << endl;
    } else {
        cout << "No config file found, using default camera settings" << endl;
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

    // Wait for first IMU data to establish time base
    cout << "Waiting for IMU data..." << endl;
    uint64_t firstImuTimestamp = 0;
    while (b_continue_session && firstImuTimestamp == 0) {
        auto imuData = imuReader.getIMUData();
        if (!imuData.empty()) {
            firstImuTimestamp = imuData.front().timestamp_ms;
            cout << "First IMU timestamp: " << firstImuTimestamp << " ms" << endl;

            // Print first IMU reading to verify values (gravity should be ~9.8 m/s² on one axis)
            const auto& first = imuData.front();
            cout << "First IMU reading (raw):" << endl;
            cout << "  Accel: [" << first.accel_x() << ", " << first.accel_y() << ", " << first.accel_z() << "] g" << endl;
            cout << "  Gyro:  [" << first.gyro_x() << ", " << first.gyro_y() << ", " << first.gyro_z() << "] deg/s" << endl;

            // Use new getAccelSI() method
            float ax, ay, az;
            first.getAccelSI(ax, ay, az);
            cout << "  Accel (SI): [" << ax << ", " << ay << ", " << az << "] m/s²" << endl;
        }
        this_thread::sleep_for(chrono::milliseconds(10));
    }

    if (!b_continue_session) {
        cout << "Interrupted before initialization" << endl;
        return 0;
    }

    // Create SLAM system
    cout << "Creating ORB-SLAM3 system..." << endl;
    ORB_SLAM3::System SLAM(vocabularyPath, settingsPath, ORB_SLAM3::System::IMU_MONOCULAR, true);
    float imageScale = SLAM.GetImageScale();

    // Calculate camera-IMU time offset
    // Two components:
    // 1. Camera.TimeOffset from Kalibr: represents full delay from trigger to image center
    //    (Kalibr convention: Time_camera = Time_imu + td)
    // 2. Half exposure time: if Kalibr wasn't used, this approximates the delay
    //
    // Kalibr's timeshift already includes the half-exposure effect, so we use it directly
    // if available. If not, we fall back to just half-exposure as an approximation.
    double cameraTimeOffset = 0.0;
    {
        cv::FileStorage fSettings(settingsPath, cv::FileStorage::READ);
        if (fSettings.isOpened()) {
            cv::FileNode node = fSettings["Camera.TimeOffset"];
            if (!node.empty() && node.isReal()) {
                cameraTimeOffset = static_cast<double>(node);
                cout << "Camera.TimeOffset from Kalibr: " << cameraTimeOffset * 1000.0 << " ms" << endl;
            }
            fSettings.release();
        }
    }

    // Calculate half exposure time
    double halfExposureTimeSec = camera->exposureTime() / 2.0 / 1e6;
    cout << "Exposure time: " << camera->exposureTime() << " us" << endl;
    cout << "Half exposure time: " << halfExposureTimeSec * 1000.0 << " ms" << endl;

    // Use Kalibr offset if available, otherwise fall back to half exposure
    double frameTimeOffset = (cameraTimeOffset != 0.0) ? cameraTimeOffset : halfExposureTimeSec;
    cout << "Using frame time offset: " << frameTimeOffset * 1000.0 << " ms" << endl;

    // Clear IMU buffer accumulated during SLAM initialization (loading vocabulary takes time)
    // and reset time base to avoid huge initial backlog
    cout << "Clearing IMU buffer accumulated during initialization..." << endl;
    imuReader.getIMUData();  // Discard accumulated IMU data
    imuReader.clearTriggerQueue();  // Discard ALL pending camera triggers

    // Wait for fresh IMU data to establish new time base
    // IMPORTANT: Use front() not back() to match the timestamp of the first IMU sample
    // Using back() would shift the time base forward, causing IMU samples to appear
    // "in the future" relative to frame times, leading to accumulation issues
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

    // Unit conversion constants
    constexpr float DEG_TO_RAD = 0.0174532925f;  // pi/180
    constexpr float G_TO_MS2 = 9.80665f;

    // IMU measurements for current frame
    vector<ORB_SLAM3::IMU::Point> vImuMeas;

    // Statistics
    uint64_t frameCount = 0;
    uint64_t imuCount = 0;
    auto startTime = chrono::steady_clock::now();

    // Track previous IMU timestamp for gap detection
    double lastImuTimestamp = 0;
    int imuGapWarnings = 0;

    // Main loop
    while (!SLAM.isShutDown() && b_continue_session) {
        // Get camera frame FIRST (this blocks until frame is ready)
        cv::Mat frame;
        double cameraTimestamp;

        if (!camera->getFrame(frame, cameraTimestamp)) {
            continue;
        }

        // Get Pico trigger timestamp
        // IMPORTANT: When processing is slower than camera rate, triggers accumulate in queue.
        // Camera frames get dropped during processing, but ALL triggers get queued.
        // We need to drain ALL triggers and use only the LAST one to match the current frame.
        // Otherwise we'd use an old trigger (for a dropped frame) with the current frame.
        uint64_t triggerTimestamp = 0;
        int droppedTriggers = 0;
        while (imuReader.hasTriggerTimestamp()) {
            uint64_t ts = imuReader.getCameraTriggerTimestamp();
            if (triggerTimestamp > 0) {
                droppedTriggers++;  // Previous trigger was for a dropped frame
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

            // Check for gaps in IMU stream (should be ~5ms at 200Hz)
            if (lastImuTimestamp > 0 && t > lastImuTimestamp) {
                double gap = t - lastImuTimestamp;
                if (gap > 0.010 && imuGapWarnings < 10) {  // More than 10ms gap
                    cout << "WARNING: IMU gap detected: " << fixed << setprecision(1)
                         << gap * 1000 << "ms at t=" << setprecision(3) << t << endl;
                    imuGapWarnings++;
                }
            }
            lastImuTimestamp = t;

            // Convert: gyro from deg/s to rad/s, accel from g to m/s²
            // Using accessor methods from pearvio::IMUData
            ORB_SLAM3::IMU::Point pt(
                imu.accel_x() * G_TO_MS2,
                imu.accel_y() * G_TO_MS2,
                imu.accel_z() * G_TO_MS2,
                imu.gyro_x() * DEG_TO_RAD,
                imu.gyro_y() * DEG_TO_RAD,
                imu.gyro_z() * DEG_TO_RAD,
                t
            );
            vImuMeas.push_back(pt);
            imuCount++;
        }

        // Determine frame time
        // Apply time offset to align trigger timestamp with actual image capture time
        double frameTime;
        if (triggerTimestamp > 0 && triggerTimestamp >= firstImuTimestamp) {
            // Use Pico timestamp (relative to first IMU timestamp) + time offset
            frameTime = (triggerTimestamp - firstImuTimestamp) / 1000.0 + frameTimeOffset;
        } else {
            // No trigger - fall back to last IMU timestamp
            if (!vImuMeas.empty()) {
                frameTime = vImuMeas.back().t;
                if (frameCount < 20) {
                    cout << "No trigger, using last IMU time: " << fixed << setprecision(3) << frameTime << endl;
                }
            } else {
                // No IMU data yet, skip frame
                if (frameCount < 20) {
                    cout << "Skipping frame: no trigger and no IMU data" << endl;
                }
                continue;
            }
        }

        // Filter IMU measurements: only keep those with timestamp <= frameTime
        // Keep measurements after frameTime for next iteration
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
                // Show gap between last IMU and frame time
                double imuFrameGap = frameTime - vImuForFrame.back().t;
                if (imuFrameGap > 0.005) {  // More than 5ms
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

        // Track with filtered IMU data
        SLAM.TrackMonocular(frame, frameTime, vImuForFrame);

        frameCount++;

        // Print IMU diagnostics every 50 frames to verify motion detection
        if (frameCount % 50 == 0 && !vImuForFrame.empty()) {
            // Calculate max gyro from recent IMU data to see if motion is detected
            float maxGyro = 0;
            float avgAccelMag = 0;
            for (const auto& pt : vImuForFrame) {
                float gyroMag = pt.w.norm();  // Eigen vector norm
                if (gyroMag > maxGyro) maxGyro = gyroMag;
                avgAccelMag += pt.a.norm();  // Eigen vector norm
            }
            avgAccelMag /= vImuForFrame.size();
            cout << "IMU check: maxGyro=" << fixed << setprecision(2) << maxGyro * 57.3 << " deg/s"
                 << " | accelMag=" << setprecision(2) << avgAccelMag << " m/s²"
                 << " (expect ~9.8 stationary, varies with motion)" << endl;
        }

        // Print statistics every 100 frames
        if (frameCount % 100 == 0) {
            auto now = chrono::steady_clock::now();
            double elapsed = chrono::duration<double>(now - startTime).count();
            double fps = frameCount / elapsed;

            // Get tracking state
            int trackingState = SLAM.GetTrackingState();
            const char* stateStr = "UNKNOWN";
            switch (trackingState) {
                case -1: stateStr = "NOT_READY"; break;
                case 0: stateStr = "NO_IMAGES"; break;
                case 1: stateStr = "INIT"; break;
                case 2: stateStr = "OK"; break;
                case 3: stateStr = "RECENTLY_LOST"; break;
                case 4: stateStr = "LOST"; break;
                case 5: stateStr = "OK_KLT"; break;
            }

            cout << "Frames: " << frameCount
                 << " | IMU: " << imuCount
                 << " | FPS: " << fixed << setprecision(1) << fps
                 << " | State: " << stateStr
                 << " | Time: " << fixed << setprecision(1) << elapsed << "s"
                 << endl;
        }
    }

    // Cleanup
    cout << endl << "Shutting down..." << endl;

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
    cout << "Total frames:  " << frameCount << endl;
    cout << "Total IMU:     " << imuCount << endl;
    cout << "Duration:      " << fixed << setprecision(1) << totalTime << " s" << endl;
    cout << "Average FPS:   " << fixed << setprecision(1) << (frameCount / totalTime) << endl;
    cout << "========================================" << endl;

    // Save trajectory
    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
    cout << "Trajectories saved to CameraTrajectory.txt and KeyFrameTrajectory.txt" << endl;

    return 0;
}

