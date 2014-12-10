
#ifndef VIDEO_DETECTION_V8_DET_ENGINE
#define VIDEO_DETECTION_V8_DET_ENGINE

#include "video_det_param.h"
#include <string>
#include <vector>
#include <memory>

#include <opencv2/opencv.hpp>

#include <flann/flann.hpp>

#include "nonCopyable.h"
#include "colorFeat.h"
namespace vd
{
    class DetEngine:public nonCopyable
    {
        //typedef cv::flann::Index FeatIndexType;
        //typedef ::flann::Index<::flann::HistIntersectionDistance<float>>  FeatIndexType;
        typedef ::flann::Index<::flann::L1<float>> FeatIndexType;
    private:
        Video_det_param param_;
        ColorHist colorHist;
        std::vector<std::string> videoNames_;
        std::vector<int> frameCnts_;
        cv::Mat colorFeat_;
        std::vector<int> index2VideoId_;
        std::vector<int> framePosMapG2L_;
        std::shared_ptr< FeatIndexType > featIndex_;
        bool changed_;
        std::string dataPath_;
    public:
        DetEngine(std::string dataPath) :dataPath_(dataPath), changed_(false)
        {
        }
        ~DetEngine()
        {
            saveState();
        }
        int addVideos(std::vector<std::string> &videoNames);
        int train();

        int detectSingleVideo(std::string videoName, std::vector<std::string> &jsonStr);
        int readState();
    private:
        int procVideos();
        int trainFlannIndex();

        int saveState();        
        int saveIndex();

        int genFrameMap();

    };
}
#endif