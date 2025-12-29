/**
 * Monocular-Inertial ORB-SLAM3 with Pear VIO Camera
 *
 * Clean implementation using the Pear VIO Camera API.
 *
 * Hardware:
 * - OV9281 global shutter camera (via libcamera)
 * - BMI160 IMU via Raspberry Pi Pico (200Hz, hardware-triggered camera)
 *
 * Usage: ./mono_inertial_pear path_to_vocabulary path_to_settings [serial_port]
 *        Default serial port: /dev/ttyACM0
 */

#include <signal.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <atomic>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <pear/pear.h>
#include <System.h>

using namespace std;

// Global flag for clean shutdown
atomic<bool> g_running{true};

void signalHandler(int s) {
    cout << "\nShutting down..." << endl;
    g_running = false;
}

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        cerr << endl
             << "Usage: ./mono_inertial_pear path_to_vocabulary path_to_settings [serial_port]"
             << endl
             << "  Default serial_port: /dev/ttyACM0"
             << endl;
        return 1;
    }

    string vocabularyPath = argv[1];
    string settingsPath = argv[2];
    string serialPort = (argc == 4) ? argv[3] : "/dev/ttyACM0";

    // Setup signal handler
    struct sigaction sigHandler;
    sigHandler.sa_handler = signalHandler;
    sigemptyset(&sigHandler.sa_mask);
    sigHandler.sa_flags = 0;
    sigaction(SIGINT, &sigHandler, NULL);

    cout << "========================================" << endl;
    cout << "Monocular-Inertial VIO (Pear Camera)" << endl;
    cout << "========================================" << endl;
    cout << "Vocabulary: " << vocabularyPath << endl;
    cout << "Settings:   " << settingsPath << endl;
    cout << "IMU Port:   " << serialPort << endl;
    cout << "========================================" << endl;

    // Initialize Pear VIO Camera
    pear::PearVIOCamera camera;

    if (!camera.open(serialPort)) {
        cerr << "Failed to open Pear camera" << endl;
        return 1;
    }

    // Load camera settings from viewer_app config (if available)
    camera.loadSettings();

    if (!camera.start()) {
        cerr << "Failed to start camera" << endl;
        return 1;
    }

    // Enable hardware trigger mode on OV9281 via I2C
    // This must be done AFTER camera.start() so the driver is initialized
    if (!camera.setTriggerMode(true)) {
        cerr << "Warning: Failed to enable trigger mode" << endl;
        cerr << "  Camera may not sync properly with hardware trigger" << endl;
    }

    // Wait for first IMU data to establish time base
    uint64_t firstImuTimestamp = camera.waitForIMU();
    if (firstImuTimestamp == 0) {
        cerr << "Failed to receive IMU data" << endl;
        return 1;
    }

    if (!g_running) {
        cout << "Interrupted before initialization" << endl;
        return 0;
    }

    // Create SLAM system
    cout << "Creating ORB-SLAM3 system..." << endl;
    ORB_SLAM3::System SLAM(vocabularyPath, settingsPath, ORB_SLAM3::System::IMU_MONOCULAR, true);
    float imageScale = SLAM.GetImageScale();

    // Load camera-IMU time offset from settings
    double frameTimeOffset = 0.0;
    {
        cv::FileStorage fSettings(settingsPath, cv::FileStorage::READ);
        if (fSettings.isOpened()) {
            cv::FileNode node = fSettings["Camera.TimeOffset"];
            if (!node.empty() && node.isReal()) {
                frameTimeOffset = static_cast<double>(node);
                cout << "Camera.TimeOffset from Kalibr: " << frameTimeOffset * 1000.0 << " ms" << endl;
            }
            fSettings.release();
        }
    }

    // If no Kalibr offset, use half exposure time as approximation
    if (frameTimeOffset == 0.0) {
        frameTimeOffset = camera.exposureTimeUs() / 2.0 / 1e6;
        cout << "Using half exposure time offset: " << frameTimeOffset * 1000.0 << " ms" << endl;
    }

    // Clear buffers accumulated during SLAM initialization
    cout << "Clearing buffers accumulated during initialization..." << endl;
    camera.clearBuffers();

    // Re-establish time base
    firstImuTimestamp = camera.waitForIMU();
    if (firstImuTimestamp == 0 || !g_running) {
        cerr << "Failed to re-establish time base" << endl;
        return 1;
    }

    // Discard first few frames to ensure valid timestamps
    cout << "Discarding initial frames..." << endl;
    for (int i = 0; i < 5 && g_running; i++) {
        pear::VIOFrame frame;
        camera.getFrame(frame);
    }
    camera.clearBuffers();

    cout << "VIO system ready. Press Ctrl+C to exit." << endl;
    cout << "========================================" << endl;

    // Unit conversion constants
    constexpr float DEG_TO_RAD = 0.0174532925f;
    constexpr float G_TO_MS2 = 9.80665f;

    // IMU measurements buffer
    vector<ORB_SLAM3::IMU::Point> vImuMeas;

    // Statistics
    uint64_t frameCount = 0;
    uint64_t imuCount = 0;
    auto startTime = chrono::steady_clock::now();

    // Gap detection
    double lastImuTimestamp = 0;
    int imuGapWarnings = 0;

    // Main loop
    while (!SLAM.isShutDown() && g_running) {
        pear::VIOFrame frame;
        if (!camera.getFrame(frame)) {
            continue;
        }

        // Skip frames with invalid trigger timestamps
        if (frame.hasTrigger && frame.triggerTimestampMs < firstImuTimestamp) {
            if (frameCount < 20) {
                cout << "Skipping frame with old trigger timestamp" << endl;
            }
            continue;
        }

        // Convert IMU data to ORB-SLAM3 format
        for (const auto& imu : frame.imuData) {
            double t = static_cast<double>(imu.timestamp_ms - firstImuTimestamp) / 1000.0;

            // Check for gaps in IMU stream
            if (lastImuTimestamp > 0 && t > lastImuTimestamp) {
                double gap = t - lastImuTimestamp;
                if (gap > 0.010 && imuGapWarnings < 10) {
                    cout << "WARNING: IMU gap: " << fixed << setprecision(1)
                         << gap * 1000 << "ms at t=" << setprecision(3) << t << endl;
                    imuGapWarnings++;
                }
            }
            lastImuTimestamp = t;

            // Convert units: gyro deg/s -> rad/s, accel g -> m/s²
            ORB_SLAM3::IMU::Point pt(
                imu.accel_x * G_TO_MS2,
                imu.accel_y * G_TO_MS2,
                imu.accel_z * G_TO_MS2,
                imu.gyro_x * DEG_TO_RAD,
                imu.gyro_y * DEG_TO_RAD,
                imu.gyro_z * DEG_TO_RAD,
                t
            );
            vImuMeas.push_back(pt);
            imuCount++;
        }

        // Calculate frame timestamp
        double frameTime;
        if (frame.hasTrigger) {
            frameTime = static_cast<double>(frame.triggerTimestampMs - firstImuTimestamp) / 1000.0 + frameTimeOffset;
        } else {
            // Fallback to last IMU timestamp
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

        // Split IMU data: keep measurements after frame time for next iteration
        vector<ORB_SLAM3::IMU::Point> vImuForFrame;
        vector<ORB_SLAM3::IMU::Point> vImuForNext;

        for (const auto& pt : vImuMeas) {
            if (pt.t <= frameTime) {
                vImuForFrame.push_back(pt);
            } else {
                vImuForNext.push_back(pt);
            }
        }

        // Debug output for first 20 frames
        if (frameCount < 20) {
            cout << "Frame " << frameCount << ": t=" << fixed << setprecision(3) << frameTime
                 << " IMU=" << vImuForFrame.size() << "/" << vImuMeas.size();
            if (!vImuForFrame.empty()) {
                cout << " range=[" << vImuForFrame.front().t << "," << vImuForFrame.back().t << "]";
            }
            cout << " trigger=" << (frame.hasTrigger ? "yes" : "NO") << endl;
        }

        vImuMeas = std::move(vImuForNext);

        // Need IMU data to track
        if (vImuForFrame.empty()) {
            if (frameCount < 20) {
                cout << "  -> Skipping: no IMU data" << endl;
            }
            continue;
        }

        // Resize image if needed
        cv::Mat image = frame.image;
        if (imageScale != 1.0f) {
            int newWidth = static_cast<int>(image.cols * imageScale);
            int newHeight = static_cast<int>(image.rows * imageScale);
            cv::resize(image, image, cv::Size(newWidth, newHeight));
        }

        // Track frame
        SLAM.TrackMonocular(image, frameTime, vImuForFrame);
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
            cout << "IMU: maxGyro=" << fixed << setprecision(2) << maxGyro * 57.3 << " deg/s"
                 << " | accelMag=" << setprecision(2) << avgAccelMag << " m/s²" << endl;
        }

        // Print statistics every 100 frames
        if (frameCount % 100 == 0) {
            auto now = chrono::steady_clock::now();
            double elapsed = chrono::duration<double>(now - startTime).count();
            double fps = frameCount / elapsed;

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
    camera.setTriggerMode(false);

    camera.stop();
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
    cout << "Trajectories saved" << endl;

    return 0;
}
