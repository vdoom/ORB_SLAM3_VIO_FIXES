#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include "System.h"

int main(int argc, char **argv) {
    if(argc != 3) {
        std::cerr << "Usage: ./stereo_inertial_realsense path_to_vocabulary path_to_settings" << std::endl;
        return 1;
    }

    // Create SLAM system - Fixed: Use IMU_STEREO instead of STEREO_INERTIAL
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::IMU_STEREO, true);

    // Configure RealSense
    rs2::pipeline pipe;
    rs2::config cfg;
    
    cfg.enable_stream(RS2_STREAM_INFRARED, 1, 640, 480, RS2_FORMAT_Y8, 30);
    cfg.enable_stream(RS2_STREAM_INFRARED, 2, 640, 480, RS2_FORMAT_Y8, 30);
    cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F, 250);
    cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F, 400);
    
    rs2::pipeline_profile profile = pipe.start(cfg);
    
    std::vector<ORB_SLAM3::IMU::Point> vImuMeas;
    
    // IMU data collection variables
    double last_timestamp = 0;
    
    while(true) {
        rs2::frameset frames = pipe.wait_for_frames();
        
        // Collect IMU data first
        auto motion_frames = frames.first_or_default(RS2_STREAM_ACCEL);
        auto gyro_frames = frames.first_or_default(RS2_STREAM_GYRO);
        
        if (motion_frames && gyro_frames) {
            auto accel_data = motion_frames.as<rs2::motion_frame>().get_motion_data();
            auto gyro_data = gyro_frames.as<rs2::motion_frame>().get_motion_data();
            
            double imu_timestamp = motion_frames.get_timestamp() * 1e-3;
            
            // Create IMU measurement
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
        
        double timestamp = ir_frame_left.get_timestamp() * 1e-3; // Convert to seconds
        
        // Track with stereo-inertial
        Sophus::SE3f Tcw = SLAM.TrackStereo(left, right, timestamp, vImuMeas);
        
        vImuMeas.clear(); // Clear after use
        
        // Fixed: Check if tracking was successful differently
        // For Sophus::SE3f, we can't use .empty(), check the translation norm instead
        Eigen::Vector3f translation = Tcw.translation();
        if(translation.norm() > 0 || !std::isnan(translation.norm())) {
            std::cout << "Tracking successful - Position: " 
                      << translation.x() << ", " << translation.y() << ", " << translation.z() << std::endl;
        } else {
            std::cout << "Tracking lost" << std::endl;
        }
        
        // Optional: Display images
        cv::imshow("Left", left);
        cv::imshow("Right", right);
        if(cv::waitKey(1) == 27) break; // ESC to exit
    }
    
    SLAM.Shutdown();
    return 0;
}
