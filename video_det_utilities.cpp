
#ifndef VIDEO_DETECTION_V8_DET_UTILITIES
#define VIDEO_DETECTION_V8_DET_UTILITIES

#include <string>
#include <vector>

#include "video_proc_utilities.h"
namespace vd
{
    using std::vector;
    using std::string;

    int trainIndex(string configFilePath, vector<string> &videoNames)
    {
        Video_det_param param;
        if (param.loadFromFile(configFilePath) !=0)
        {
            param = Video_det_param();
        }
        
        for (size_t i = 0; i != videoNames.size();++i)
        {

        }

        return 0;
    }
}


#endif