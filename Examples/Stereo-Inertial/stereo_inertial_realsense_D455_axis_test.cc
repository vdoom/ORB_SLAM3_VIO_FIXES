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
#include <condition_variable>
#include "System.h"

// Global flag for clean shutdown
std::atomic<bool> keep_running(true);

void signal_handler(int signal) {
    std::cout << "\nShutdown signal received. Printing final statistics..." << std::endl;
    keep_running = false;
}

// Interpolate IMU measurements to match gyro timestamps
rs2_vector interpolateMeasure(const double target_time,
                              const rs2_vector current_data, const double current_time,
                              const rs2_vector prev_data, const double prev_time)
{
    // If there are not previous information, the current data is propagated
    if(prev_time == 0)
    {
        return current_data;
    }

    rs2_vector increment;
    rs2_vector value_interp;

    if(target_time > current_time) {
        value_interp = current_data;
    }
    else if(target_time > prev_time)
    {
        increment.x = current_data.x - prev_data.x;
        increment.y = current_data.y - prev_data.y;
        increment.z = current_data.z - prev_data.z;

        double factor = (target_time - prev_time) / (current_time - prev_time);

        value_interp.x = prev_data.x + increment.x * factor;
        value_interp.y = prev_data.y + increment.y * factor;
        value_interp.z = prev_data.z + increment.z * factor;

        // Use current data (zero-order hold)
        value_interp = current_data;
    }
    else
    {
        value_interp = prev_data;
    }

    return value_interp;
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
    cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
    cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F);

    // IMU callback variables
    std::mutex imu_mutex;
    std::condition_variable cond_image_rec;

    std::vector<double> v_accel_timestamp;
    std::vector<rs2_vector> v_accel_data;
    std::vector<double> v_gyro_timestamp;
    std::vector<rs2_vector> v_gyro_data;

    double prev_accel_timestamp = 0;
    rs2_vector prev_accel_data;
    double current_accel_timestamp = 0;
    rs2_vector current_accel_data;
    std::vector<double> v_accel_timestamp_sync;
    std::vector<rs2_vector> v_accel_data_sync;

    cv::Mat imCV, imRightCV;
    int width_img = 640, height_img = 480;
    double timestamp_image = -1.0;
    bool image_ready = false;
    int count_im_buffer = 0;

    double offset = 0; // Timestamp offset in ms

    // IMU callback
    auto imu_callback = [&](const rs2::frame& frame)
    {
        std::unique_lock<std::mutex> lock(imu_mutex);

        if(rs2::frameset fs = frame.as<rs2::frameset>())
        {
            count_im_buffer++;

            double new_timestamp_image = fs.get_timestamp()*1e-3;
            if(std::abs(timestamp_image-new_timestamp_image)<0.001){
                count_im_buffer--;
                return;
            }

            rs2::video_frame ir_frameL = fs.get_infrared_frame(1);
            rs2::video_frame ir_frameR = fs.get_infrared_frame(2);

            imCV = cv::Mat(cv::Size(width_img, height_img), CV_8U, (void*)(ir_frameL.get_data()), cv::Mat::AUTO_STEP);
            imRightCV = cv::Mat(cv::Size(width_img, height_img), CV_8U, (void*)(ir_frameR.get_data()), cv::Mat::AUTO_STEP);

            timestamp_image = fs.get_timestamp()*1e-3;
            image_ready = true;

            while(v_gyro_timestamp.size() > v_accel_timestamp_sync.size())
            {
                int index = v_accel_timestamp_sync.size();
                double target_time = v_gyro_timestamp[index];

                v_accel_data_sync.push_back(current_accel_data);
                v_accel_timestamp_sync.push_back(target_time);
            }

            lock.unlock();
            cond_image_rec.notify_all();
        }
        else if (rs2::motion_frame m_frame = frame.as<rs2::motion_frame>())
        {
            if (m_frame.get_profile().stream_name() == "Gyro")
            {
                v_gyro_data.push_back(m_frame.get_motion_data());
                v_gyro_timestamp.push_back((m_frame.get_timestamp()+offset)*1e-3);
            }
            else if (m_frame.get_profile().stream_name() == "Accel")
            {
                prev_accel_timestamp = current_accel_timestamp;
                prev_accel_data = current_accel_data;

                current_accel_data = m_frame.get_motion_data();
                current_accel_timestamp = (m_frame.get_timestamp()+offset)*1e-3;

                while(v_gyro_timestamp.size() > v_accel_timestamp_sync.size())
                {
                    int index = v_accel_timestamp_sync.size();
                    double target_time = v_gyro_timestamp[index];

                    rs2_vector interp_data = interpolateMeasure(target_time, current_accel_data, current_accel_timestamp,
                                                                prev_accel_data, prev_accel_timestamp);

                    v_accel_data_sync.push_back(interp_data);
                    v_accel_timestamp_sync.push_back(target_time);
                }
            }
        }
    };

    rs2::pipeline_profile profile = pipe.start(cfg, imu_callback);

    std::vector<ORB_SLAM3::IMU::Point> vImuMeas;
    int frame_count = 0;

    std::cout << "Starting VIO axis testing..." << std::endl;
    std::cout << "Move the camera to initialize the system." << std::endl;

    while(keep_running) {
        std::vector<rs2_vector> vAccel, vGyro;
        std::vector<double> vGyro_times;
        cv::Mat im, imRight;
        double timestamp;

        {
            std::unique_lock<std::mutex> lk(imu_mutex);
            if(!image_ready)
                cond_image_rec.wait(lk);

            if(count_im_buffer > 1)
                std::cout << count_im_buffer - 1 << " dropped frs\n";
            count_im_buffer = 0;

            while(v_gyro_timestamp.size() > v_accel_timestamp_sync.size())
            {
                int index = v_accel_timestamp_sync.size();
                double target_time = v_gyro_timestamp[index];

                v_accel_data_sync.push_back(current_accel_data);
                v_accel_timestamp_sync.push_back(target_time);
            }

            vAccel = v_accel_data_sync;
            vGyro = v_gyro_data;
            vGyro_times = v_gyro_timestamp;
            timestamp = timestamp_image;
            im = imCV.clone();
            imRight = imRightCV.clone();

            v_accel_data.clear();
            v_accel_timestamp.clear();
            v_gyro_data.clear();
            v_gyro_timestamp.clear();
            v_accel_data_sync.clear();
            v_accel_timestamp_sync.clear();

            image_ready = false;
        }

        frame_count++;

        // Build IMU measurements from synchronized data
        for(size_t i=0; i<vGyro.size(); ++i)
        {
            ORB_SLAM3::IMU::Point lastPoint(vAccel[i].x, vAccel[i].y, vAccel[i].z,
                                  vGyro[i].x, vGyro[i].y, vGyro[i].z,
                                  vGyro_times[i]);
            vImuMeas.push_back(lastPoint);
        }

        // Track with stereo-inertial (images come from callback)
        Sophus::SE3f Tcw = SLAM.TrackStereo(im, imRight, timestamp, vImuMeas);

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

            // Use latest gyro data for angular velocity stats
            if (!vGyro.empty()) {
                rs2_vector latest_gyro = vGyro.back();
                stats.updateGyro(latest_gyro.x, latest_gyro.y, latest_gyro.z);
            }

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
