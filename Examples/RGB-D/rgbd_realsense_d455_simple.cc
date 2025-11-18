#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>

#include"../../include/System.h"
#include<librealsense2/rs.hpp>

using namespace std;

int main(int argc, char **argv) {
    if (argc != 3) {
        cerr << endl << "Usage: ./rgbd_realsense_d455_simple path_to_vocabulary path_to_settings" << endl;
        return 1;
    }

    // Create SLAM system
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::RGBD, true);

    cout << "Starting RealSense..." << endl;

    // RealSense pipeline
    rs2::pipeline pipe;
    rs2::config cfg;

    // Configure streams
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);

    // Start pipeline
    rs2::pipeline_profile profile = pipe.start(cfg);

    // Get depth scale
    auto depth_sensor = profile.get_device().first<rs2::depth_sensor>();
    float depth_scale = depth_sensor.get_depth_scale();
    
    cout << "Depth scale: " << depth_scale << endl;
    cout << "Starting main loop. Press ESC to quit." << endl;

    // Create align object to align depth to color
    rs2::align align(RS2_STREAM_COLOR);

    while(true) {
        rs2::frameset frames = pipe.wait_for_frames();
        
        // Align frames
        frames = align.process(frames);
        
        rs2::video_frame color_frame = frames.get_color_frame();
        rs2::depth_frame depth_frame = frames.get_depth_frame();

        if (!color_frame || !depth_frame) {
            cout << "Missing frames, skipping..." << endl;
            continue;
        }

        // Convert to OpenCV
        cv::Mat color(cv::Size(640, 480), CV_8UC3, (void*)color_frame.get_data());
        cv::Mat depth(cv::Size(640, 480), CV_16UC1, (void*)depth_frame.get_data());

        // Convert depth to meters
        cv::Mat depth_meters;
        depth.convertTo(depth_meters, CV_32FC1, depth_scale);

        double timestamp = color_frame.get_timestamp() * 1e-3;

        cout << "Processing frame at timestamp: " << timestamp << endl;

        // Pass to SLAM - don't store return value if not needed
        SLAM.TrackRGBD(color, depth_meters, timestamp);

        // Display images for debugging
        cv::imshow("Color", color);
        cv::Mat depth_display;
        depth.convertTo(depth_display, CV_8UC1, 255.0/10000.0);
        cv::imshow("Depth", depth_display);

        if(cv::waitKey(1) == 27) break; // ESC
    }

    cout << "Shutting down..." << endl;
    SLAM.Shutdown();
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    return 0;
}
