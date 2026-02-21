/**
* This file is part of ORB-SLAM3
*
* Video recorder for feature tracking visualization.
* Records FrameDrawer output to a video file without opening any GUI windows.
*/

#include "VideoRecorder.h"

#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <unistd.h>

namespace ORB_SLAM3
{

VideoRecorder::VideoRecorder(FrameDrawer* pFrameDrawer, Tracking* pTracking,
                             const string &strSettingPath, Settings* settings,
                             const string &strOutputDir):
    both(false), mpFrameDrawer(pFrameDrawer), mpTracker(pTracking),
    mbFinishRequested(false), mbFinished(true),
    mImageViewerScale(1.f), mOutputDir(strOutputDir)
{
    if(settings){
        LoadParameters(settings);
    }
    else{
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
        ParseParamFile(fSettings);
    }

    mVideoFilePath = GenerateFilename(mOutputDir);
}

void VideoRecorder::LoadParameters(Settings *settings) {
    mImageViewerScale = 1.f;

    float fps = settings->fps();
    if(fps < 1)
        fps = 30;
    mT = 1e3/fps;

    mImageViewerScale = settings->imageViewerScale();
}

bool VideoRecorder::ParseParamFile(cv::FileStorage &fSettings)
{
    mImageViewerScale = 1.f;

    float fps = fSettings["Camera.fps"];
    if(fps < 1)
        fps = 30;
    mT = 1e3/fps;

    cv::FileNode node = fSettings["Viewer.imageViewScale"];
    if(!node.empty())
    {
        mImageViewerScale = node.real();
    }

    return true;
}

string VideoRecorder::GenerateFilename(const string &outputDir)
{
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    struct tm tm_now;
    localtime_r(&time_t_now, &tm_now);

    std::ostringstream oss;
    oss << std::put_time(&tm_now, "%Y-%m-%d_%H-%M-%S");

    string dir = outputDir;
    if(!dir.empty() && dir.back() != '/')
        dir += '/';

    return dir + "VIO_" + oss.str() + ".avi";
}

void VideoRecorder::Run()
{
    mbFinished = false;

    float trackedImageScale = mpTracker->GetImageScale();
    float fps = 1000.0f / mT;

    cv::VideoWriter videoWriter;
    bool writerOpened = false;

    cout << "VideoRecorder: recording to " << mVideoFilePath << " at " << fps << " fps" << endl;

    while(1)
    {
        cv::Mat toShow;
        cv::Mat im = mpFrameDrawer->DrawFrame(trackedImageScale);

        if(im.empty()){
            usleep(static_cast<useconds_t>(mT * 1000));
            if(CheckFinish())
                break;
            continue;
        }

        if(both){
            cv::Mat imRight = mpFrameDrawer->DrawRightFrame(trackedImageScale);
            cv::hconcat(im, imRight, toShow);
        }
        else{
            toShow = im;
        }

        if(mImageViewerScale != 1.f)
        {
            int width = toShow.cols * mImageViewerScale;
            int height = toShow.rows * mImageViewerScale;
            cv::resize(toShow, toShow, cv::Size(width, height));
        }

        if(!writerOpened)
        {
            int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
            videoWriter.open(mVideoFilePath, fourcc, fps,
                             cv::Size(toShow.cols, toShow.rows), true);
            if(!videoWriter.isOpened())
            {
                cerr << "VideoRecorder: ERROR - could not open video file: " << mVideoFilePath << endl;
                break;
            }
            writerOpened = true;
            cout << "VideoRecorder: video writer opened (" << toShow.cols << "x" << toShow.rows << ")" << endl;
        }

        videoWriter.write(toShow);

        usleep(static_cast<useconds_t>(mT * 1000));

        if(CheckFinish())
            break;
    }

    if(writerOpened)
    {
        videoWriter.release();
        cout << "VideoRecorder: video saved to " << mVideoFilePath << endl;
    }

    mbFinished = true;
}

void VideoRecorder::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool VideoRecorder::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void VideoRecorder::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool VideoRecorder::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

} // namespace ORB_SLAM3
