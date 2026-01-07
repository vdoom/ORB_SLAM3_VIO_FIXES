/**
 * This file is part of ORB-SLAM3
 *
 * Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gomez Rodriguez,
 * Jose M.M. Montiel and Juan D. Tardos, University of Zaragoza.
 * Copyright (C) 2014-2016 Raul Mur-Artal, Jose M.M. Montiel and Juan D. Tardos,
 * University of Zaragoza.
 *
 * ORB-SLAM3 is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later version.
 *
 * ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * ORB-SLAM3. If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * Monocular ORB-SLAM3 for Custom Camera (OV9281 via PearAPI)
 *
 * Uses PearAPI CameraBackend for camera access (supports libcamera on RPi,
 * V4L2+OpenCV on Jetson).
 *
 * Usage: ./mono_custom_cam path_to_vocabulary path_to_settings [trajectory_file_name]
 */

#include <signal.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <atomic>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <PearAPI/PearAPI.h>
#include <System.h>

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
             << "Usage: ./mono_custom_cam path_to_vocabulary path_to_settings [trajectory_file_name]"
             << endl;
        return 1;
    }

    string vocabularyPath = argv[1];
    string settingsPath = argv[2];
    string trajectoryFile;

    if (argc == 4) {
        trajectoryFile = argv[3];
    }

    // Setup signal handler for graceful shutdown
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = exit_loop_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);

    cout << "========================================" << endl;
    cout << "Monocular ORB-SLAM3 (Custom Camera)" << endl;
    cout << "========================================" << endl;
    cout << "Vocabulary: " << vocabularyPath << endl;
    cout << "Settings:   " << settingsPath << endl;
    if (!trajectoryFile.empty()) {
        cout << "Trajectory: " << trajectoryFile << endl;
    }
    cout << "========================================" << endl;

    // Initialize camera using PearAPI
    auto camera = pearvio::CameraBackend::create();
    if (!camera) {
        cerr << "Failed to create camera backend" << endl;
        return 1;
    }

    pearvio::CameraConfig config;
    config.cameraIndex = 0;
    config.width = 640;
    config.height = 400;
    config.fps = 30;

    // Load camera settings from PearCameraApp config (exposure, gain, etc.)
    if (config.loadFromIniFile()) {
        cout << "Loaded camera settings from config file" << endl;
        // Monocular mode doesn't use trigger - disable it even if config has it enabled
        config.triggerMode = false;
    }

    if (!camera->initialize(config)) {
        cerr << "Failed to initialize camera" << endl;
        cerr << "Check that:" << endl;
        cerr << "  - Camera is connected (libcamera-hello --list-cameras)" << endl;
        cerr << "  - User has video group access: sudo usermod -aG video $USER" << endl;
        return 1;
    }

    if (!camera->start()) {
        cerr << "Failed to start camera" << endl;
        return 1;
    }

    // Print camera info
    cout << "Camera resolution: " << camera->frameWidth() << "x" << camera->frameHeight() << endl;

    // Create SLAM system
    cout << "Creating ORB-SLAM3 system..." << endl;
    ORB_SLAM3::System SLAM(vocabularyPath, settingsPath, ORB_SLAM3::System::MONOCULAR,
                           true, 0, trajectoryFile);
    float imageScale = SLAM.GetImageScale();

    cout << "SLAM system ready. Press Ctrl+C to exit." << endl;
    cout << "========================================" << endl;

    // Statistics
    auto startTime = chrono::steady_clock::now();
    auto lastStatsTime = startTime;
    uint64_t frameCount = 0;
    uint64_t droppedFrames = 0;

    double t_resize = 0.0;
    double t_track = 0.0;

    // Main loop
    while (!SLAM.isShutDown() && b_continue_session) {
        cv::Mat frame;
        double timestamp;

        if (!camera->getFrame(frame, timestamp, 1000)) {
            continue;
        }

        frameCount++;

        // Resize if needed
        if (imageScale != 1.0f) {
#ifdef REGISTER_TIMES
    #ifdef COMPILEDWITHC11
            auto t_Start_Resize = chrono::steady_clock::now();
    #else
            auto t_Start_Resize = chrono::monotonic_clock::now();
    #endif
#endif
            int newWidth = static_cast<int>(frame.cols * imageScale);
            int newHeight = static_cast<int>(frame.rows * imageScale);
            cv::resize(frame, frame, cv::Size(newWidth, newHeight));

#ifdef REGISTER_TIMES
    #ifdef COMPILEDWITHC11
            auto t_End_Resize = chrono::steady_clock::now();
    #else
            auto t_End_Resize = chrono::monotonic_clock::now();
    #endif
            t_resize = chrono::duration_cast<chrono::duration<double, milli>>(
                t_End_Resize - t_Start_Resize).count();
            SLAM.InsertResizeTime(t_resize);
#endif
        }

#ifdef REGISTER_TIMES
    #ifdef COMPILEDWITHC11
        auto t_Start_Track = chrono::steady_clock::now();
    #else
        auto t_Start_Track = chrono::monotonic_clock::now();
    #endif
#endif

        // Track monocular frame
        SLAM.TrackMonocular(frame, timestamp);

#ifdef REGISTER_TIMES
    #ifdef COMPILEDWITHC11
        auto t_End_Track = chrono::steady_clock::now();
    #else
        auto t_End_Track = chrono::monotonic_clock::now();
    #endif
        t_track = t_resize + chrono::duration_cast<chrono::duration<double, milli>>(
            t_End_Track - t_Start_Track).count();
        SLAM.InsertTrackTime(t_track);
#endif

        // Print statistics every 5 seconds
        auto now = chrono::steady_clock::now();
        double statsDelta = chrono::duration<double>(now - lastStatsTime).count();

        if (statsDelta >= 5.0) {
            double fps = frameCount / chrono::duration<double>(now - startTime).count();
            double totalTime = chrono::duration<double>(now - startTime).count();

            cout << "Frames: " << frameCount
                 << " | FPS: " << fixed << setprecision(1) << fps
                 << " | Time: " << fixed << setprecision(1) << totalTime << "s"
                 << endl;

            lastStatsTime = now;
        }
    }

    // Cleanup
    cout << endl << "Shutting down..." << endl;

    camera->stop();
    SLAM.Shutdown();

    // Final statistics
    auto endTime = chrono::steady_clock::now();
    double totalTime = chrono::duration<double>(endTime - startTime).count();

    cout << "========================================" << endl;
    cout << "Session complete" << endl;
    cout << "Total frames:   " << frameCount << endl;
    cout << "Duration:       " << fixed << setprecision(1) << totalTime << " s" << endl;
    cout << "Average FPS:    " << fixed << setprecision(1) << (frameCount / totalTime) << endl;
    cout << "========================================" << endl;

    // Save trajectory
    if (trajectoryFile.empty()) {
        SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
        SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
        cout << "Trajectories saved to CameraTrajectory.txt and KeyFrameTrajectory.txt" << endl;
    } else {
        SLAM.SaveTrajectoryTUM("f_" + trajectoryFile + ".txt");
        SLAM.SaveKeyFrameTrajectoryTUM("kf_" + trajectoryFile + ".txt");
        cout << "Trajectories saved to f_" << trajectoryFile << ".txt and kf_"
             << trajectoryFile << ".txt" << endl;
    }

    return 0;
}
