/**
* This file is part of ORB-SLAM3
*
* Video recorder for feature tracking visualization.
* Records FrameDrawer output to a video file without opening any GUI windows.
* Video filename is auto-generated with date and time of the start.
*/

#ifndef VIDEORECORDER_H
#define VIDEORECORDER_H

#include "FrameDrawer.h"
#include "Tracking.h"
#include "Settings.h"

#include <opencv2/videoio.hpp>
#include <string>
#include <mutex>
#include <thread>

namespace ORB_SLAM3
{

class Tracking;
class FrameDrawer;
class Settings;

class VideoRecorder
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VideoRecorder(FrameDrawer* pFrameDrawer, Tracking* pTracking,
                  const string &strSettingPath, Settings* settings,
                  const string &strOutputDir);

    // Main thread function. Grabs frames from FrameDrawer and writes to video file.
    void Run();

    void RequestFinish();
    bool isFinished();

    bool both;

    // Get the generated video file path
    string GetVideoFilePath() const { return mVideoFilePath; }

private:
    void LoadParameters(Settings* settings);
    bool ParseParamFile(cv::FileStorage &fSettings);

    // Generate unique filename with date and time
    static string GenerateFilename(const string &outputDir);

    bool CheckFinish();
    void SetFinish();

    FrameDrawer* mpFrameDrawer;
    Tracking* mpTracker;

    // 1/fps in ms
    double mT;
    float mImageViewerScale;

    string mVideoFilePath;
    string mOutputDir;

    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;
};

} // namespace ORB_SLAM3

#endif // VIDEORECORDER_H
