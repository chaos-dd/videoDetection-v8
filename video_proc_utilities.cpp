
#include "video_proc_utilities.h"

#include <iostream>
#include <tuple>
#include <numeric>
#include <stdio.h>
#include <time.h>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/opencv.hpp>

#include <opencv2/nonfree/features2d.hpp>
#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <opencv2/flann/flann.hpp>

#include "colorFeat.h"
#include "video_det_param.h"


namespace vd
{
    using std::cout;
    using std::endl;
    using std::to_string;
    using std::tuple;
    using std::make_tuple;
    using std::get;
    using std::string;
    using std::vector;
    using std::shared_ptr;

    using cv::Point;
    using cv::Vec3b;
    using cv::Mat;
    using cv::imread;
    using cv::KeyPoint;
    using cv::DMatch;
    using cv::Size;
    using cv::ORB;
    using cv::waitKey;
    using cv::RNG;
    using cv::Point2f;
    using cv::BFMatcher;
    using cv::NORM_HAMMING;
    using cv::Range;
    using cv::VideoCapture;

    using boost::filesystem::exists;
    using boost::filesystem::create_directory;
    using boost::filesystem::directory_iterator;
    using boost::property_tree::ptree;


    static int sum_image(Mat &img, int dim)
    {
        int sum = 0;
        for (int i = 0; i != img.rows; ++i)
        {
            for (int j = 0; j != img.cols; ++j)
            {
                sum += img.at<Vec3b>(i, j)[dim];
            }
        }
        return sum;
    }

    static  int calcPixel(Mat img, int dim)
    {
        int num = 0;
        for (int i = 0; i != img.rows; ++i)
        {
            for (int j = 0; j != img.cols; ++j)
            {
                if (img.at<Vec3b>(i, j)[0] + img.at<Vec3b>(i, j)[1] + img.at<Vec3b>(i, j)[2] > 30)
                {
                    num++;
                }
            }
        }
        return num;
    }


    static tuple<int, int, int, int>
        find_boundry(VideoCapture &cap)
    {
            float thres = 0.01;
            int piexl_thres = 5;


            vector<int> row_start_rec;
            vector<int> row_end_rec;
            vector<int> col_start_rec;
            vector<int> col_end_rec;

            int cnt = 0;
            while (cnt++ < 200)
            {
                Mat frame;
                cap.read(frame);
                if (!cap.isOpened())
                {
                    break;
                }
                int rows = frame.rows;
                int cols = frame.cols;

                int i = 1;

                while (i<rows)
                {
                    if (calcPixel(frame(Range(0, i), Range(0, cols)), 0)>thres *i *cols)
                        break;

                    i += 1;
                }
                row_start_rec.push_back(i);


                i = rows - 1;
                while (i >= 0)
                {
                    if (calcPixel(frame(Range(i, rows), Range(0, cols)), 0) > thres * (rows - i)*cols)
                        break;
                    i--;
                }
                row_end_rec.push_back(i);

                i = 1;
                while (i<cols)
                {
                    if (calcPixel(frame(Range(0, rows), Range(0, i)), 0) >thres*i*rows)
                        break;
                    i++;
                }
                col_start_rec.push_back(i);

                i = cols - 1;
                while (i >= 0)
                {
                    if (calcPixel(frame(Range(0, rows), Range(i, cols)), 0) > thres*(cols - i)*rows)
                        break;
                    i--;
                }
                col_end_rec.push_back(i);
            }
            int start_row = std::accumulate(row_start_rec.begin(), row_start_rec.end(), 0) / row_start_rec.size();
            int end_row = std::accumulate(row_end_rec.begin(), row_end_rec.end(), 0) / row_end_rec.size();

            int start_col = std::accumulate(col_start_rec.begin(), col_start_rec.end(), 0) / col_start_rec.size();
            int end_col = std::accumulate(col_end_rec.begin(), col_end_rec.end(), 0) / col_end_rec.size();;

            tuple<int, int, int, int> ret_tuple(0, 0, 0, 0);

            ret_tuple = make_tuple(start_row, end_row, start_col, end_col);
            cap.set(CV_CAP_PROP_POS_FRAMES, 0);

            return ret_tuple;
        }

    static void scaleFrame(Mat &img, int maxFrameSize)
    {
        int r, c;
        if (img.rows > img.cols)
        {
            r = maxFrameSize;
            c = (float)maxFrameSize / img.rows * img.cols;
        }
        else
        {
            c = maxFrameSize;
            r = (float)maxFrameSize / img.cols * img.rows;
        }
        resize(img, img, Size(c, r));
    }
    int exactFeat(string videoPath, Video_det_param param, Mat &colorFeat, bool saveData)
    {
        boost::filesystem::path vn(videoPath);
        string nm = vn.string();
        VideoCapture cap(nm);

        if (!cap.isOpened())
        {
            cout << string("exactFeatAndKFrame: ") + nm + "  ´ò¿ªÊ§°Ü";
            return -1;
        }


        double fps = cap.get(CV_CAP_PROP_FPS);
        double totalFrameNum = cap.get(CV_CAP_PROP_FRAME_COUNT);
        int frameWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH);
        int frameHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT);

        //image has frame count of 1
        if (totalFrameNum < 2)
        {
            return -1;
        }

        //erase  back margin in video 
        //tuple<int, int, int, int> ranges = find_boundry(cap);

        tuple<int, int, int, int> ranges(0, frameHeight, 0, frameWidth);

        vector<Mat> vec_colorFeat;
        vec_colorFeat.reserve(totalFrameNum / fps*param.used_fps);
        ColorHist colorHist;
        //ColorCoherenceVec colorHist;
        CEdgeHist edgeHist;

        Mat orginalFrame;
        int validFrameInd = 0;

        
        for (int frameInd = 0; frameInd < totalFrameNum; frameInd += fps / param.used_fps)
        {

            cap.set(CV_CAP_PROP_POS_FRAMES, frameInd);
            if (!cap.read(orginalFrame) || orginalFrame.empty())
            {
                continue;
            }

            Mat frame = orginalFrame(Range(get<0>(ranges), get<1>(ranges)), Range(get<2>(ranges), get<3>(ranges)));

            //scaleFrame(frame, param.maxFrameSize);
            resize(frame, frame, Size(param.maxFrameSize, param.maxFrameSize));

            //char fileName[100] = { 0 };
            //sprintf(fileName, "%s_%d.jpg", videoPath.c_str(), validFrameInd);
            //imwrite(fileName, frame);
            
            Mat hist;
            colorHist.computeFeat(frame, hist, COLOR_SPACE_HLS, cv::NORM_L1);
            //Mat ehist;
            //edgeHist.computeFeat(frame, ehist, COLOR_SPACE_HLS, cv::NORM_L1);

            vec_colorFeat.push_back(hist);
            validFrameInd++;

        }


        if (vec_colorFeat.size() <1 )
        {
            return -1;
        }

        colorFeat.push_back(vec_colorFeat[0]);
        colorFeat.reserve(validFrameInd-1);

        for (int i = 1; i != validFrameInd; ++i)
        {
            colorFeat.push_back(vec_colorFeat[i]);
        }

        if (saveData == true)
        {
            string dir = string(vn.parent_path().string() + "_data");
            if (!exists(dir))
            {
                create_directory(dir);
            }

            string fileName = string(dir + '/' + vn.filename().string() + ".data");
            FILE *file = fopen(fileName.c_str(), "wb");

            if (!file)
            {
                cout << "exactFeatAndKFrame £º can not open out put file " + fileName;
                return -1;
            }

            //save color feat
            fwrite(&colorFeat.rows, sizeof(colorFeat.rows), 1, file);
            fwrite(&colorFeat.cols, sizeof(colorFeat.cols), 1, file);
            fwrite(colorFeat.data, sizeof(float), colorFeat.rows*colorFeat.cols, file);

            fclose(file);
        }

        return 0;
    }

    int loadVideoFeat(string videoName, Mat &colorFeat)
    {
        boost::filesystem::path p(videoName);
        string dir = string(p.parent_path().string() + "_data");
        if (!exists(dir))
        {
            return -1;
        }

        string fileName = string(dir + '/' + p.filename().string() + ".data");
        FILE *file = fopen(fileName.c_str(), "rb");

        if (!file)
        {
            cout << "loadVideoFeat : can not open file " + fileName<<endl;
            return -1;
        }

        //load color feature
        int r, c;
        fread(&r, sizeof(r), 1, file);
        fread(&c, sizeof(c), 1, file);
        colorFeat = Mat::zeros(r, c, CV_32FC1);
        fread(colorFeat.data, sizeof(float), r*c, file);

        fclose(file);
        return 0;
    }




}
