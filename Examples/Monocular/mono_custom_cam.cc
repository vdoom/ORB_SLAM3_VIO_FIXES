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
 * Monocular ORB-SLAM3 for Custom Camera (OV9281 via libcamera)
 *
 * Based on mono_realsense_D435i.cc but adapted for:
 * - libcamera API instead of librealsense (for Raspberry Pi cameras)
 * - OV9281 global shutter monochrome camera at 1280x800
 * - Camera settings loaded from viewer_app config file
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
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <map>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <libcamera/libcamera.h>
#include <libcamera/control_ids.h>

#include <System.h>

// For mmap
#include <sys/mman.h>
#include <unistd.h>

using namespace std;
using namespace libcamera;

// ============================================================================
// Global state
// ============================================================================

atomic<bool> b_continue_session{true};

void exit_loop_handler(int s) {
    cout << "\nFinishing session..." << endl;
    b_continue_session = false;
}

// ============================================================================
// Camera Settings (loaded from viewer_app INI file)
// ============================================================================

struct CameraSettings {
    bool autoExposure = true;
    int exposureTimeUs = 10000;   // microseconds
    float gain = 8.0f;

    // Load settings from INI file (same format as viewer_app)
    bool loadFromIniFile(const string& path) {
        ifstream file(path);
        if (!file.is_open()) {
            return false;
        }

        string line;
        string currentSection;

        while (getline(file, line)) {
            // Trim whitespace
            size_t start = line.find_first_not_of(" \t\r\n");
            if (start == string::npos) continue;
            size_t end = line.find_last_not_of(" \t\r\n");
            line = line.substr(start, end - start + 1);

            // Skip empty lines and comments
            if (line.empty() || line[0] == '#' || line[0] == ';') continue;

            // Section header
            if (line[0] == '[' && line.back() == ']') {
                currentSection = line.substr(1, line.size() - 2);
                continue;
            }

            // Key=value pair
            size_t eqPos = line.find('=');
            if (eqPos == string::npos) continue;

            string key = line.substr(0, eqPos);
            string value = line.substr(eqPos + 1);

            // Trim key and value
            start = key.find_first_not_of(" \t");
            end = key.find_last_not_of(" \t");
            if (start != string::npos) key = key.substr(start, end - start + 1);

            start = value.find_first_not_of(" \t");
            end = value.find_last_not_of(" \t");
            if (start != string::npos) value = value.substr(start, end - start + 1);

            // Parse camera section
            if (currentSection == "camera") {
                if (key == "auto_exposure") {
                    autoExposure = (value == "true" || value == "1");
                } else if (key == "manual_exposure_us") {
                    exposureTimeUs = stoi(value);
                } else if (key == "manual_gain") {
                    gain = stof(value);
                }
            }
        }

        return true;
    }

    // Get default config path (~/.config/rpi_camera_viewer.ini)
    static string getDefaultConfigPath() {
        const char* home = getenv("HOME");
        if (home) {
            return string(home) + "/.config/rpi_camera_viewer.ini";
        }
        return "";
    }
};

// ============================================================================
// Camera Controller (libcamera)
// ============================================================================

class CameraController {
public:
    CameraController() = default;
    ~CameraController() { stop(); }

    bool initialize(int cameraIndex = 0) {
        cameraManager_ = make_unique<CameraManager>();
        int ret = cameraManager_->start();
        if (ret < 0) {
            cerr << "Failed to start camera manager: " << strerror(-ret) << endl;
            return false;
        }

        auto cameras = cameraManager_->cameras();
        if (cameras.empty()) {
            cerr << "No cameras found" << endl;
            return false;
        }

        if (cameraIndex >= static_cast<int>(cameras.size())) {
            cerr << "Camera index " << cameraIndex << " out of range (found "
                 << cameras.size() << " cameras)" << endl;
            return false;
        }

        camera_ = cameras[cameraIndex];
        ret = camera_->acquire();
        if (ret < 0) {
            cerr << "Failed to acquire camera: " << strerror(-ret) << endl;
            return false;
        }

        cout << "Acquired camera: " << camera_->id() << endl;

        // Configure for 1280x800 R8 (grayscale) - native resolution of OV9281
        config_ = camera_->generateConfiguration({StreamRole::Viewfinder});
        if (!config_) {
            cerr << "Failed to generate camera configuration" << endl;
            return false;
        }

        StreamConfiguration& streamConfig = config_->at(0);
        streamConfig.size = Size(640, 400);//Size(1280, 800);
        streamConfig.pixelFormat = formats::R8;

        cout << "Requesting format: R8 (8-bit grayscale) at 1280x800" << endl;

        auto status = config_->validate();
        if (status == CameraConfiguration::Invalid) {
            cerr << "Invalid camera configuration" << endl;
            return false;
        }

        if (status == CameraConfiguration::Adjusted) {
            cout << "Camera configuration adjusted to: "
                 << streamConfig.size.width << "x" << streamConfig.size.height
                 << " " << streamConfig.pixelFormat.toString() << endl;
        }

        ret = camera_->configure(config_.get());
        if (ret < 0) {
            cerr << "Failed to configure camera: " << strerror(-ret) << endl;
            return false;
        }

        frameWidth_ = streamConfig.size.width;
        frameHeight_ = streamConfig.size.height;
        frameStride_ = streamConfig.stride;
        pixelFormat_ = streamConfig.pixelFormat;
        stream_ = streamConfig.stream();

        // Allocate buffers
        allocator_ = make_unique<FrameBufferAllocator>(camera_);
        ret = allocator_->allocate(stream_);
        if (ret < 0) {
            cerr << "Failed to allocate buffers: " << strerror(-ret) << endl;
            return false;
        }

        cout << "Allocated " << allocator_->buffers(stream_).size() << " buffers" << endl;

        // Map buffers for CPU access
        for (const auto& buffer : allocator_->buffers(stream_)) {
            for (const auto& plane : buffer->planes()) {
                size_t length = plane.length;
                if (length == 0) {
                    off_t fdSize = lseek(plane.fd.get(), 0, SEEK_END);
                    if (fdSize < 0) {
                        cerr << "Failed to get buffer size" << endl;
                        return false;
                    }
                    length = static_cast<size_t>(fdSize);
                    lseek(plane.fd.get(), 0, SEEK_SET);
                }

                void* mem = mmap(nullptr, length, PROT_READ, MAP_SHARED, plane.fd.get(), 0);
                if (mem == MAP_FAILED) {
                    cerr << "Failed to mmap buffer: " << strerror(errno) << endl;
                    return false;
                }
                mappedBuffers_[buffer.get()] = {static_cast<uint8_t*>(mem), length};
            }
        }

        // Create requests
        for (const auto& buffer : allocator_->buffers(stream_)) {
            auto request = camera_->createRequest();
            if (!request) {
                cerr << "Failed to create request" << endl;
                return false;
            }
            request->addBuffer(stream_, buffer.get());
            requests_.push_back(move(request));
        }

        // Connect callback
        camera_->requestCompleted.connect(this, &CameraController::requestComplete);

        cout << "Camera initialized: " << frameWidth_ << "x" << frameHeight_
             << " stride: " << frameStride_
             << " format: " << pixelFormat_.toString() << endl;

        // Check format support
        if (pixelFormat_ == formats::R8) {
            cout << "  -> Using R8 (8-bit grayscale) - optimal" << endl;
        } else if (pixelFormat_ == formats::R16) {
            cout << "  -> Using R16 (16-bit grayscale) - will convert 10-bit to 8-bit" << endl;
        } else if (pixelFormat_ == formats::XRGB8888 || pixelFormat_ == formats::XBGR8888) {
            cout << "  -> Using XRGB8888 (32-bit) - will extract grayscale channel" << endl;
        } else {
            cout << "  -> Warning: Format " << pixelFormat_.toString()
                 << " may not be fully supported" << endl;
        }

        return true;
    }

    // Load camera settings from INI file
    bool loadSettings(const string& configPath = "") {
        string path = configPath.empty() ? CameraSettings::getDefaultConfigPath() : configPath;

        if (path.empty()) {
            cout << "No config path available, using default camera settings" << endl;
            return false;
        }

        if (settings_.loadFromIniFile(path)) {
            cout << "Loaded camera settings from: " << path << endl;
            cout << "  Auto exposure: " << (settings_.autoExposure ? "enabled" : "disabled") << endl;
            if (!settings_.autoExposure) {
                cout << "  Manual exposure: " << settings_.exposureTimeUs << " us" << endl;
                cout << "  Manual gain: " << settings_.gain << endl;
            }
            return true;
        } else {
            cout << "Config file not found: " << path << endl;
            cout << "Using default camera settings (auto exposure)" << endl;
            return false;
        }
    }

    bool start() {
        if (!camera_) return false;
        if (running_) return true;

        // Set frame duration for consistent FPS (33ms = ~30 FPS)
        ControlList controls;
        int64_t frameDurationMin = 33333;  // 30 FPS in microseconds
        int64_t frameDurationMax = 33333;
        controls.set(controls::FrameDurationLimits,
                     Span<const int64_t, 2>({frameDurationMin, frameDurationMax}));

        // Apply exposure settings from loaded config
        if (settings_.autoExposure) {
            controls.set(controls::AeEnable, true);
            cout << "Camera using auto exposure" << endl;
        } else {
            // Manual exposure mode - must set BOTH exposure time AND gain
            controls.set(controls::AeEnable, false);
            controls.set(controls::ExposureTime, settings_.exposureTimeUs);
            controls.set(controls::AnalogueGain, settings_.gain);
            cout << "Camera using manual exposure: " << settings_.exposureTimeUs
                 << " us, gain: " << settings_.gain << endl;
        }

        int ret = camera_->start(&controls);
        if (ret < 0) {
            cerr << "Failed to start camera: " << strerror(-ret) << endl;
            return false;
        }

        running_ = true;
        frameCount_ = 0;
        droppedFrames_ = 0;

        // Queue all requests with exposure controls
        for (auto& request : requests_) {
            applyExposureControls(request.get());
            camera_->queueRequest(request.get());
        }

        cout << "Camera started at ~30 FPS" << endl;
        return true;
    }

    void stop() {
        if (!running_) return;
        running_ = false;

        if (camera_) {
            camera_->stop();
        }

        // Cleanup mapped buffers
        for (auto& [buffer, info] : mappedBuffers_) {
            if (info.data) {
                munmap(info.data, info.size);
            }
        }
        mappedBuffers_.clear();

        if (camera_) {
            camera_->release();
            camera_.reset();
        }
        if (cameraManager_) {
            cameraManager_->stop();
            cameraManager_.reset();
        }

        cout << "Camera stopped" << endl;
    }

    // Wait for next frame (blocking)
    bool getFrame(cv::Mat& frame, double& timestamp) {
        unique_lock<mutex> lock(frameMtx_);
        if (!frameReady_) {
            frameCond_.wait(lock, [this] { return frameReady_ || !running_; });
        }

        if (!running_) return false;

        frame = currentFrame_.clone();
        timestamp = currentTimestamp_;
        frameReady_ = false;
        return true;
    }

    int width() const { return frameWidth_; }
    int height() const { return frameHeight_; }
    bool isRunning() const { return running_; }
    uint64_t frameCount() const { return frameCount_; }
    uint64_t droppedFrames() const { return droppedFrames_; }

private:
    void applyExposureControls(Request* request) {
        ControlList& controls = request->controls();

        if (settings_.autoExposure) {
            controls.set(controls::AeEnable, true);
        } else {
            controls.set(controls::AeEnable, false);
            controls.set(controls::ExposureTime, settings_.exposureTimeUs);
            controls.set(controls::AnalogueGain, settings_.gain);
        }
    }

    void requestComplete(Request* request) {
        if (!running_) return;
        if (request->status() == Request::RequestCancelled) return;

        // Get buffer
        FrameBuffer* buffer = request->buffers().begin()->second;
        auto it = mappedBuffers_.find(buffer);
        if (it == mappedBuffers_.end()) {
            request->reuse(Request::ReuseBuffers);
            applyExposureControls(request);
            camera_->queueRequest(request);
            return;
        }

        const auto& info = it->second;

        // Get timestamp (SensorTimestamp is in nanoseconds)
        auto tsOpt = request->metadata().get(controls::SensorTimestamp);
        double timestamp = tsOpt.value_or(0) / 1e9;  // Convert to seconds

        // Convert to cv::Mat based on pixel format
        cv::Mat gray;

        if (pixelFormat_ == formats::R8) {
            // Direct 8-bit grayscale
            gray = cv::Mat(frameHeight_, frameWidth_, CV_8UC1, info.data, frameStride_);
        } else if (pixelFormat_ == formats::R16) {
            // 16-bit grayscale - OV9281 outputs 10-bit packed into 16-bit
            // Data is in upper bits, shift right by 8 to get 8-bit value
            gray = cv::Mat(frameHeight_, frameWidth_, CV_8UC1);
            const uint16_t* src = reinterpret_cast<const uint16_t*>(info.data);
            int srcStride = frameStride_ / 2;  // Stride is in bytes, convert to uint16 elements

            for (int y = 0; y < frameHeight_; y++) {
                const uint16_t* srcLine = src + y * srcStride;
                uint8_t* dstLine = gray.ptr<uint8_t>(y);
                for (int x = 0; x < frameWidth_; x++) {
                    dstLine[x] = static_cast<uint8_t>(srcLine[x] >> 8);
                }
            }
        } else if (pixelFormat_ == formats::XRGB8888 || pixelFormat_ == formats::XBGR8888) {
            // 32-bit XRGB - for mono camera, all channels are same, just take one
            gray = cv::Mat(frameHeight_, frameWidth_, CV_8UC1);
            for (int y = 0; y < frameHeight_; y++) {
                const uint8_t* srcLine = info.data + y * frameStride_;
                uint8_t* dstLine = gray.ptr<uint8_t>(y);
                for (int x = 0; x < frameWidth_; x++) {
                    // Take blue channel (first byte in BGRX layout)
                    dstLine[x] = srcLine[x * 4];
                }
            }
        } else if (pixelFormat_ == formats::RGB888) {
            cv::Mat rgb(frameHeight_, frameWidth_, CV_8UC3, info.data, frameStride_);
            cv::cvtColor(rgb, gray, cv::COLOR_RGB2GRAY);
        } else if (pixelFormat_ == formats::BGR888) {
            cv::Mat bgr(frameHeight_, frameWidth_, CV_8UC3, info.data, frameStride_);
            cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
        } else {
            // Unsupported format - try to use as grayscale anyway
            cerr << "Warning: Unknown pixel format, attempting raw grayscale interpretation" << endl;
            gray = cv::Mat(frameHeight_, frameWidth_, CV_8UC1, info.data, frameStride_);
        }

        {
            lock_guard<mutex> lock(frameMtx_);

            // Check for dropped frames
            if (frameReady_) {
                droppedFrames_++;
            }

            gray.copyTo(currentFrame_);
            currentTimestamp_ = timestamp;
            frameReady_ = true;
            frameCount_++;
        }
        frameCond_.notify_one();

        // Requeue request with exposure controls
        request->reuse(Request::ReuseBuffers);
        applyExposureControls(request);
        camera_->queueRequest(request);
    }

    unique_ptr<CameraManager> cameraManager_;
    shared_ptr<Camera> camera_;
    unique_ptr<CameraConfiguration> config_;
    unique_ptr<FrameBufferAllocator> allocator_;
    vector<unique_ptr<Request>> requests_;
    Stream* stream_ = nullptr;
    PixelFormat pixelFormat_;

    struct BufferInfo {
        uint8_t* data = nullptr;
        size_t size = 0;
    };
    map<FrameBuffer*, BufferInfo> mappedBuffers_;

    int frameWidth_ = 0;
    int frameHeight_ = 0;
    int frameStride_ = 0;

    atomic<bool> running_{false};
    atomic<uint64_t> frameCount_{0};
    atomic<uint64_t> droppedFrames_{0};

    mutex frameMtx_;
    condition_variable frameCond_;
    cv::Mat currentFrame_;
    double currentTimestamp_ = 0;
    bool frameReady_ = false;

    // Camera settings loaded from INI file
    CameraSettings settings_;
};

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

    // Initialize camera
    CameraController camera;
    if (!camera.initialize(0)) {
        cerr << "Failed to initialize camera" << endl;
        cerr << "Check that:" << endl;
        cerr << "  - Camera is connected (libcamera-hello --list-cameras)" << endl;
        cerr << "  - User has video group access: sudo usermod -aG video $USER" << endl;
        return 1;
    }

    // Load camera settings from viewer_app config (if available)
    camera.loadSettings();

    if (!camera.start()) {
        cerr << "Failed to start camera" << endl;
        return 1;
    }

    // Print camera info
    cout << "Camera resolution: " << camera.width() << "x" << camera.height() << endl;

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
    uint64_t lastFrameCount = 0;

    double t_resize = 0.0;
    double t_track = 0.0;

    // Main loop
    while (!SLAM.isShutDown() && b_continue_session) {
        cv::Mat frame;
        double timestamp;

        if (!camera.getFrame(frame, timestamp)) {
            continue;
        }

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
            uint64_t currentFrameCount = camera.frameCount();
            uint64_t framesDelta = currentFrameCount - lastFrameCount;
            double fps = framesDelta / statsDelta;
            double totalTime = chrono::duration<double>(now - startTime).count();

            cout << "Frames: " << currentFrameCount
                 << " | Dropped: " << camera.droppedFrames()
                 << " | FPS: " << fixed << setprecision(1) << fps
                 << " | Time: " << fixed << setprecision(1) << totalTime << "s"
                 << endl;

            lastStatsTime = now;
            lastFrameCount = currentFrameCount;
        }
    }

    // Cleanup
    cout << endl << "Shutting down..." << endl;

    camera.stop();
    SLAM.Shutdown();

    // Final statistics
    auto endTime = chrono::steady_clock::now();
    double totalTime = chrono::duration<double>(endTime - startTime).count();
    uint64_t totalFrames = camera.frameCount();

    cout << "========================================" << endl;
    cout << "Session complete" << endl;
    cout << "Total frames:   " << totalFrames << endl;
    cout << "Dropped frames: " << camera.droppedFrames() << endl;
    cout << "Duration:       " << fixed << setprecision(1) << totalTime << " s" << endl;
    cout << "Average FPS:    " << fixed << setprecision(1) << (totalFrames / totalTime) << endl;
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
