

#include<string>
#include<vector>
#include<boost/filesystem.hpp>

#include"video_det_engine.h"

using namespace std;
using namespace boost::filesystem;

int loadTarVideoNames(string videoPath, vector<string> &tarVideoNames)
{
    vector<boost::filesystem::path>  vecPath_tar;
    try
    {
        boost::filesystem::path videoPathObj(videoPath);

        if (!exists(videoPathObj) || !is_directory(videoPathObj))
        {
            cout << string("vd::loadTarVideoNames: ""videoPaht error ");
        }
        std::copy_if(directory_iterator(videoPathObj), directory_iterator(), std::back_inserter(vecPath_tar), [](boost::filesystem::path p){return is_regular_file(p); });
    }
    catch (boost::filesystem::filesystem_error &e)
    {
        cout << string("vd::loadTarVideoNames: boostfilesystem:" + string(e.what()));
        return -1;
    }

    tarVideoNames.reserve(vecPath_tar.size());
    for_each(vecPath_tar.begin(), vecPath_tar.end(), [&tarVideoNames](boost::filesystem::path &p){tarVideoNames.push_back(p.string()); });
    return 0;

}

int main()
{

    string videoName = "/home/ufo/data/videos/det/det2.mp4";
    string videoPath = "/home/ufo/data/videos/tar";
    string dataPath = "/home/ufo/data/videos/data";
    vector<string> videoNames;
    if(-1==loadTarVideoNames(videoPath, videoNames))
        return 0;

    vd::DetEngine detEngine(dataPath);

    if (detEngine.readState() == -1)
    {
        detEngine.addVideos(videoNames);
        detEngine.train();
    }
    vector<string> json;
    detEngine.detectSingleVideo(videoName, json);
}
