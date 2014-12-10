
#ifndef VIDEO_DETECTION_V8_PROC_UTILITIES
#define VIDEO_DETECTION_V8_PROC_UTILITIES
#include <string>
#include <opencv2/opencv.hpp>

#include "video_det_param.h"

namespace vd
{

    int exactFeat(std::string videoPath, Video_det_param param, cv::Mat &colorFeat, bool saveData);
    int loadVideoFeat(std::string videoName, cv::Mat &colorFeat);


}
#endif