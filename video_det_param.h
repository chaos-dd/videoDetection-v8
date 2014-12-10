#ifndef  VIDEO_DETECTION_V8_DET_PARAM
#define VIDEO_DETECTION_V8_DET_PARAM

#include <string>

namespace vd
{
    class Video_det_param
    {
    public:
        Video_det_param() :maxFrameSize(500), timeInterval(5), colorThres(0.2f), orbThres(50), used_fps(1)
        {
        }
        int loadFromFile(std::string configFilePath);
    public:

        int maxFrameSize;
        int timeInterval;
        float colorThres;
        float orbThres;
        float used_fps;
    };

}
#endif // ! VIDEO_DETECTION_V8_DET_PARAM
