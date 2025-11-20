#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include <iomanip>
#include <Eigen/Dense>
#include <limits>
#include <csignal>
#include <atomic>
#include "System.h"

// Global flag for clean shutdown
std::atomic<bool> keep_running(true);

void signal_handler(int signal) {
    std::cout << "\nShutdown signal received. Printing final statistics..." << std::endl;
    keep_running = false;
}

// Class to track min/max values for axis testing
class AxisStatistics {
private:
    // Position min/max
    float pos_x_min, pos_x_max;
    float pos_y_min, pos_y_max;
    float pos_z_min, pos_z_max;

    // Velocity min/max (from ORB-SLAM3)
    float vel_x_min, vel_x_max;
    float vel_y_min, vel_y_max;
    float vel_z_min, vel_z_max;

    // Angular velocity min/max (from gyro)
    float gyro_x_min, gyro_x_max;
    float gyro_y_min, gyro_y_max;
    float gyro_z_min, gyro_z_max;

    int update_counter;
    std::chrono::steady_clock::time_point last_print_time;
    std::chrono::steady_clock::time_point last_reset_time;
    int reset_count;

public:
    AxisStatistics()
        : pos_x_min(std::numeric_limits<float>::max())
        , pos_x_max(std::numeric_limits<float>::lowest())
        , pos_y_min(std::numeric_limits<float>::max())
        , pos_y_max(std::numeric_limits<float>::lowest())
        , pos_z_min(std::numeric_limits<float>::max())
        , pos_z_max(std::numeric_limits<float>::lowest())
        , vel_x_min(std::numeric_limits<float>::max())
        , vel_x_max(std::numeric_limits<float>::lowest())
        , vel_y_min(std::numeric_limits<float>::max())
        , vel_y_max(std::numeric_limits<float>::lowest())
        , vel_z_min(std::numeric_limits<float>::max())
        , vel_z_max(std::numeric_limits<float>::lowest())
        , gyro_x_min(std::numeric_limits<float>::max())
        , gyro_x_max(std::numeric_limits<float>::lowest())
        , gyro_y_min(std::numeric_limits<float>::max())
        , gyro_y_max(std::numeric_limits<float>::lowest())
        , gyro_z_min(std::numeric_limits<float>::max())
        , gyro_z_max(std::numeric_limits<float>::lowest())
        , update_counter(0)
        , last_print_time(std::chrono::steady_clock::now())
        , last_reset_time(std::chrono::steady_clock::now())
        , reset_count(0)
    {
        std::cout << "\n======================================" << std::endl;
        std::cout << "   VIO AXIS TESTING TOOL" << std::endl;
        std::cout << "======================================" << std::endl;
        std::cout << "Move the camera in different directions to see which axis responds." << std::endl;
        std::cout << "Press Ctrl+C to see final statistics." << std::endl;
        std::cout << "Statistics update every 2 seconds..." << std::endl;
        std::cout << "Statistics RESET every 10 seconds..." << std::endl;
        std::cout << std::endl;
    }

    void reset() {
        pos_x_min = pos_y_min = pos_z_min = std::numeric_limits<float>::max();
        pos_x_max = pos_y_max = pos_z_max = std::numeric_limits<float>::lowest();

        vel_x_min = vel_y_min = vel_z_min = std::numeric_limits<float>::max();
        vel_x_max = vel_y_max = vel_z_max = std::numeric_limits<float>::lowest();

        gyro_x_min = gyro_y_min = gyro_z_min = std::numeric_limits<float>::max();
        gyro_x_max = gyro_y_max = gyro_z_max = std::numeric_limits<float>::lowest();

        update_counter = 0;
        reset_count++;
        last_reset_time = std::chrono::steady_clock::now();

        std::cout << "\n*** STATISTICS RESET #" << reset_count << " ***" << std::endl;
        std::cout << "Move camera in a different direction to test another axis.\n" << std::endl;
    }

    void checkAndReset() {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_reset_time).count();

        if (elapsed >= 10) {
            reset();
        }
    }

    void updatePosition(float x, float y, float z) {
        pos_x_min = std::min(pos_x_min, x);
        pos_x_max = std::max(pos_x_max, x);
        pos_y_min = std::min(pos_y_min, y);
        pos_y_max = std::max(pos_y_max, y);
        pos_z_min = std::min(pos_z_min, z);
        pos_z_max = std::max(pos_z_max, z);
    }

    void updateVelocity(float vx, float vy, float vz) {
        vel_x_min = std::min(vel_x_min, vx);
        vel_x_max = std::max(vel_x_max, vx);
        vel_y_min = std::min(vel_y_min, vy);
        vel_y_max = std::max(vel_y_max, vy);
        vel_z_min = std::min(vel_z_min, vz);
        vel_z_max = std::max(vel_z_max, vz);
    }

    void updateGyro(float gx, float gy, float gz) {
        gyro_x_min = std::min(gyro_x_min, gx);
        gyro_x_max = std::max(gyro_x_max, gx);
        gyro_y_min = std::min(gyro_y_min, gy);
        gyro_y_max = std::max(gyro_y_max, gy);
        gyro_z_min = std::min(gyro_z_min, gz);
        gyro_z_max = std::max(gyro_z_max, gz);
    }

    void printStatistics(bool force = false) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_print_time).count();

        update_counter++;

        // Print every 2 seconds or when forced
        if (elapsed >= 2 || force) {
            auto time_since_reset = std::chrono::duration_cast<std::chrono::seconds>(now - last_reset_time).count();
            int time_until_reset = 10 - time_since_reset;
            if (time_until_reset < 0) time_until_reset = 0;

            std::cout << "\n========== AXIS STATISTICS (Updates: " << update_counter
                     << " | Reset in: " << time_until_reset << "s) ==========" << std::endl;
            std::cout << std::fixed << std::setprecision(4);

            std::cout << "\n--- POSITION (meters) ---" << std::endl;
            std::cout << "X-axis: MIN = " << std::setw(10) << pos_x_min
                     << "   MAX = " << std::setw(10) << pos_x_max
                     << "   RANGE = " << std::setw(10) << (pos_x_max - pos_x_min) << std::endl;
            std::cout << "Y-axis: MIN = " << std::setw(10) << pos_y_min
                     << "   MAX = " << std::setw(10) << pos_y_max
                     << "   RANGE = " << std::setw(10) << (pos_y_max - pos_y_min) << std::endl;
            std::cout << "Z-axis: MIN = " << std::setw(10) << pos_z_min
                     << "   MAX = " << std::setw(10) << pos_z_max
                     << "   RANGE = " << std::setw(10) << (pos_z_max - pos_z_min) << std::endl;

            std::cout << "\n--- LINEAR VELOCITY (m/s) - FROM ORB-SLAM3 ---" << std::endl;
            std::cout << "X-axis: MIN = " << std::setw(10) << vel_x_min
                     << "   MAX = " << std::setw(10) << vel_x_max
                     << "   RANGE = " << std::setw(10) << (vel_x_max - vel_x_min) << std::endl;
            std::cout << "Y-axis: MIN = " << std::setw(10) << vel_y_min
                     << "   MAX = " << std::setw(10) << vel_y_max
                     << "   RANGE = " << std::setw(10) << (vel_y_max - vel_y_min) << std::endl;
            std::cout << "Z-axis: MIN = " << std::setw(10) << vel_z_min
                     << "   MAX = " << std::setw(10) << vel_z_max
                     << "   RANGE = " << std::setw(10) << (vel_z_max - vel_z_min) << std::endl;

            std::cout << "\n--- ANGULAR VELOCITY (rad/s) - FROM GYRO ---" << std::endl;
            std::cout << "X-axis: MIN = " << std::setw(10) << gyro_x_min
                     << "   MAX = " << std::setw(10) << gyro_x_max
                     << "   RANGE = " << std::setw(10) << (gyro_x_max - gyro_x_min) << std::endl;
            std::cout << "Y-axis: MIN = " << std::setw(10) << gyro_y_min
                     << "   MAX = " << std::setw(10) << gyro_y_max
                     << "   RANGE = " << std::setw(10) << (gyro_y_max - gyro_y_min) << std::endl;
            std::cout << "Z-axis: MIN = " << std::setw(10) << gyro_z_min
                     << "   MAX = " << std::setw(10) << gyro_z_max
                     << "   RANGE = " << std::setw(10) << (gyro_z_max - gyro_z_min) << std::endl;

            std::cout << "\nTIP: Move camera along ONE axis at a time to see which VIO axis responds." << std::endl;
            std::cout << "     The axis with the largest RANGE is the one responding to your movement." << std::endl;
            std::cout << "================================================\n" << std::endl;

            last_print_time = now;
        }
    }
};

int main(int argc, char **argv) {
    if(argc != 3) {
        std::cerr << "Usage: ./stereo_inertial_realsense_axis_test path_to_vocabulary path_to_settings" << std::endl;
        return 1;
    }

    // Setup signal handler for clean shutdown
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    // Create SLAM system
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::IMU_STEREO, false);

    // Create axis statistics tracker
    AxisStatistics stats;

    // Configure RealSense
    rs2::pipeline pipe;
    rs2::config cfg;

    cfg.enable_stream(RS2_STREAM_INFRARED, 1, 640, 480, RS2_FORMAT_Y8, 30);
    cfg.enable_stream(RS2_STREAM_INFRARED, 2, 640, 480, RS2_FORMAT_Y8, 30);
    cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F, 250);
    cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F, 400);

    rs2::pipeline_profile profile = pipe.start(cfg);

    std::vector<ORB_SLAM3::IMU::Point> vImuMeas;
    int frame_count = 0;

    // Variables to store latest IMU data
    Eigen::Vector3f latest_accel(0, 0, 0);
    Eigen::Vector3f latest_gyro(0, 0, 0);

    std::cout << "Starting VIO axis testing..." << std::endl;
    std::cout << "Move the camera to initialize the system." << std::endl;

    while(keep_running) {
        rs2::frameset frames = pipe.wait_for_frames();
        frame_count++;

        // Collect IMU data using RealSense's synchronized frames (same as working VIO)
        auto motion_frames = frames.first_or_default(RS2_STREAM_ACCEL);
        auto gyro_frames = frames.first_or_default(RS2_STREAM_GYRO);

        if (motion_frames && gyro_frames) {
            auto accel_data = motion_frames.as<rs2::motion_frame>().get_motion_data();
            auto gyro_data = gyro_frames.as<rs2::motion_frame>().get_motion_data();

            double imu_timestamp = motion_frames.get_timestamp() * 1e-3;

            // Store latest IMU data
            latest_accel = Eigen::Vector3f(accel_data.x, accel_data.y, accel_data.z);
            latest_gyro = Eigen::Vector3f(gyro_data.x, gyro_data.y, gyro_data.z);

            // Create properly paired IMU measurement
            ORB_SLAM3::IMU::Point imu_point(accel_data.x, accel_data.y, accel_data.z,
                                          gyro_data.x, gyro_data.y, gyro_data.z, imu_timestamp);
            vImuMeas.push_back(imu_point);
        }

        // Get stereo images
        rs2::video_frame ir_frame_left = frames.get_infrared_frame(1);
        rs2::video_frame ir_frame_right = frames.get_infrared_frame(2);

        if (!ir_frame_left || !ir_frame_right) continue;

        cv::Mat left(cv::Size(640, 480), CV_8UC1, (void*)ir_frame_left.get_data());
        cv::Mat right(cv::Size(640, 480), CV_8UC1, (void*)ir_frame_right.get_data());

        double timestamp = ir_frame_left.get_timestamp() * 1e-3;

        // Track with stereo-inertial
        Sophus::SE3f Tcw = SLAM.TrackStereo(left, right, timestamp, vImuMeas);

        // Check tracking status
        auto tracking_state = SLAM.GetTrackingState();
        bool tracking_good = (tracking_state == ORB_SLAM3::Tracking::OK ||
                             tracking_state == ORB_SLAM3::Tracking::OK_KLT);

        if (tracking_good) {
            // Get camera pose (inverse of Tcw gives camera position in world frame)
            Sophus::SE3f Twc = Tcw.inverse();
            Eigen::Vector3f position = Twc.translation();

            // Get velocity from ORB-SLAM3
            Eigen::Vector3f velocity = SLAM.GetVelocity();

            // Convert from ORB-SLAM3 camera frame to NED frame
            // ORB-SLAM3: X-right, Y-down, Z-forward
            // NED: X-north, Y-east, Z-down
            float ned_x = position.z();   // Forward -> North
            float ned_y = -position.x();  // Right -> West, so negate for East
            float ned_z = position.y();   // Down -> Down

            // Transform velocity to NED frame (same transformation as position)
            float ned_vx = velocity.z();   // Forward -> North
            float ned_vy = -velocity.x();  // Right -> West, so negate for East
            float ned_vz = velocity.y();   // Down -> Down

            // Update statistics
            stats.updatePosition(ned_x, ned_y, ned_z);
            stats.updateVelocity(ned_vx, ned_vy, ned_vz);
            stats.updateGyro(latest_gyro.x(), latest_gyro.y(), latest_gyro.z());

            // Check if 10 seconds elapsed and reset statistics
            stats.checkAndReset();

            // Print statistics periodically
            stats.printStatistics();
        } else {
            // Display tracking status occasionally
            if (frame_count % 30 == 0) {
                switch(tracking_state) {
                    case ORB_SLAM3::Tracking::SYSTEM_NOT_READY:
                        std::cout << "System not ready..." << std::endl;
                        break;
                    case ORB_SLAM3::Tracking::NO_IMAGES_YET:
                        std::cout << "No images yet..." << std::endl;
                        break;
                    case ORB_SLAM3::Tracking::NOT_INITIALIZED:
                        std::cout << "Not initialized - move camera with rotation!" << std::endl;
                        break;
                    case ORB_SLAM3::Tracking::RECENTLY_LOST:
                    case ORB_SLAM3::Tracking::LOST:
                        std::cout << "Tracking lost!" << std::endl;
                        break;
                    default:
                        break;
                }
            }
        }

        // Clear IMU measurements after use (like the working VIO)
        vImuMeas.clear();

        // Check for ESC key
        if (cv::waitKey(1) == 27) break;
    }

    // Print final statistics
    std::cout << "\n\n========== FINAL STATISTICS ==========" << std::endl;
    stats.printStatistics(true);

    std::cout << "\nShutting down..." << std::endl;
    SLAM.Shutdown();

    return 0;
}
