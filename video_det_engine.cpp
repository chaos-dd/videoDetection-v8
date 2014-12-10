
#include "video_det_engine.h"
#include "video_proc_utilities.h"

#include <stdio.h>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/filesystem.hpp>

namespace vd
{
    using namespace std;
    using namespace cv;
    using namespace boost::filesystem;

    using boost::property_tree::ptree;
    using boost::property_tree::write_json;

    void saveResult(string det_vn,string tar_vn, string &json, Mat resultImg)
    {
        //save results
        boost::filesystem::path det_path(det_vn);

        string dir = string(det_path.parent_path().string() + "_data");
        if (!exists(dir))
        {
            create_directory(dir);
        }

        boost::filesystem::path tar_path(tar_vn);


        string resultImgFileName = string(dir + '/' + det_path.filename().string() + tar_path.filename().string() + ".jpg");
        imwrite(resultImgFileName, resultImg);

        ptree pt;
        stringstream strstream(json);
        read_json(strstream, pt);
        write_json(string(dir + '/' + det_path.filename().string() + tar_path.filename().string() + ".json"), pt);
    }
    Mat drawDetResult(int frameNum1, vector<pair<int, int>> matchResult_tar, int frameNum2, vector<pair<int, int>> matchResult_det)
    {

        //show result

        int max_frames = frameNum1 > frameNum2 ? frameNum1 : frameNum2;

        int height = 400;
        int width = 1000;
        Mat showImg(height, width, CV_8UC3, Scalar(30, 30, 30));


        RNG  rng(getTickCount());

        //Scalar line_color = Scalar(rng(255), rng(255), rng(255));
        Scalar line_color = Scalar(170, 75, 17);
        int lineType = CV_AA;

        line(showImg, Point(0, 100), Point(static_cast<int>(width*(float)frameNum1 / max_frames), 100), line_color, 3, lineType);
        line(showImg, Point(0, 300), Point(static_cast<int>(width * (float)frameNum2 / max_frames), 300), line_color, 3, lineType);

        int fontFace = CV_FONT_HERSHEY_DUPLEX;

        double fontScale = 1;
        const int bufSize = 100;
        int thickness = 1;

        int baseline = 0;

        Scalar font_color = Scalar(rng(255), rng(255), rng(255));
        char text[bufSize] = { 0 };
        Point outPt;
        Size sz;

        //original video information
        snprintf(text, bufSize,"detect:%ds ", frameNum1);
        sz = getTextSize(text, fontFace, fontScale, thickness, &baseline);

        outPt = Point(static_cast<int>(width - sz.width), 0 + sz.height*1.2);
        putText(showImg, text, outPt, fontFace, fontScale, font_color, thickness, lineType);
        memset(text, 0, bufSize);

        //detecting video information
        snprintf(text,bufSize, "target:%ds ", frameNum2);
        sz = getTextSize(text, fontFace, fontScale, thickness, &baseline);
        outPt = Point(width - sz.width, height - sz.height);
        putText(showImg, text, outPt, fontFace, fontScale, font_color, thickness, lineType);
        memset(text, 0, bufSize);


        fontFace = CV_FONT_HERSHEY_COMPLEX_SMALL;
        fontScale = 0.7;
        lineType = 4;
        for (int i = 0; i != matchResult_tar.size(); ++i)
        {
            Scalar color(rng(255), rng(255), rng(255));

            Point pt[4];
            pt[0] = Point(static_cast<int>(width * (float)matchResult_tar[i].first / max_frames), 100);

            snprintf(text,bufSize, "%d", matchResult_tar[i].first);
            sz = getTextSize(text, fontFace, fontScale, thickness, &baseline);
            outPt = Point(pt[0].x - sz.width / 2, pt[0].y - sz.height);
            putText(showImg, text, outPt, fontFace, fontScale, font_color, thickness, lineType);
            memset(text, 0, bufSize);


            pt[1] = Point(static_cast<int>(width * (float)matchResult_det[i].first / max_frames), 300);

            snprintf(text, bufSize,"%d", matchResult_det[i].first);
            sz = getTextSize(text, fontFace, fontScale, thickness, &baseline);
            outPt = Point(pt[1].x - sz.width / 2, pt[1].y + sz.height * 2);
            putText(showImg, text, outPt, fontFace, fontScale, font_color, thickness, lineType);
            memset(text, 0, bufSize);


            pt[2] = Point(static_cast<int>(width * (float)matchResult_det[i].second / max_frames), 300);

            snprintf(text, bufSize,"%d", matchResult_det[i].second);
            sz = getTextSize(text, fontFace, fontScale, thickness, &baseline);
            outPt = Point(pt[2].x - sz.width / 2, pt[2].y + sz.height * 2);
            putText(showImg, text, outPt, fontFace, fontScale, font_color, thickness, lineType);
            memset(text, 0, bufSize);


            pt[3] = Point(static_cast<int>(width * (float)matchResult_tar[i].second / max_frames), 100);

            snprintf(text,bufSize, "%d", matchResult_tar[i].second);
            sz = getTextSize(text, fontFace, fontScale, thickness, &baseline);
            outPt = Point(pt[3].x - sz.width / 2, pt[3].y - sz.height);
            putText(showImg, text, outPt, fontFace, fontScale, font_color, thickness, lineType);
            memset(text, 0, bufSize);

            fillConvexPoly(showImg, pt, 4, color, CV_AA);

            waitKey();

        }

        return showImg;
    }

    int genJsonStr(string &targetVideoName, vector<pair<int, int>> & finalMatchResult_det,
                   vector<pair<int, int>> &finalMatchResult_tar, string &strJsonResult)
    {
        ptree json_result;

        json_result.put("filename", targetVideoName);

        ptree json_array;
        for (size_t i = 0; i != finalMatchResult_det.size(); ++i)
        {
            ptree temp_pt;
            pair<int, int> & tempTar = finalMatchResult_tar[i];
            pair<int, int> & tempDet = finalMatchResult_det[i];

            temp_pt.put("det_beg", tempDet.first);
            temp_pt.put("det_end", tempDet.second);
            temp_pt.put("tar_beg", tempTar.first);
            temp_pt.put("tar_end", tempTar.second);


            json_array.push_back(std::make_pair("", temp_pt));
        }
        json_result.put_child("time", json_array);

        //json_result.put("pic_path", resultImgFileName);
        stringstream strstream;
        write_json(strstream, json_result);
        strJsonResult = strstream.str();

        return 0;
    }
    static tuple<int, int> go_bottom_right(Mat &recordMap, SparseMat & flagMat, int r, int c, int interval)
    {
        Mat flag;
        flagMat.copyTo(flag);
        //int gap_len = 0;

        int valid_r = r;
        int valid_c = c;

        //while loop avoid recursion
        while (true)
        {
            //given a (valid_r,valid_c) find for next one, if success,break out for, 
            //use the found (valid_r,valid_c) to start with
            //if for loop end normally, then no next one, so return
            // in the for loop visit the element in diagonal line first,the method found on stackoverflow.com
            bool next = false;
            for (int diag = 1; diag < interval; ++diag)
            {
                auto pred = [&](int pos_r, int pos_c)
                {
                    return pos_r < recordMap.rows && pos_c
                        && recordMap.data[pos_r * recordMap.cols + pos_c] == 0
                        && flagMat.ptr(pos_r, pos_c, false) != NULL;
                };

                int test_row = valid_r + diag;
                int test_col = valid_c + diag;

                if (pred(test_row, test_col))
                {
                    next = true;
                    valid_r = test_row;
                    valid_c = test_col;
                }
                else
                {
                    for (int delta = 1; delta <= diag; ++delta)
                    {

                        test_row = valid_r + diag;
                        test_col = valid_c + diag - delta;
                        if (pred(test_row, test_col))
                        {
                            next = true;
                            valid_r = test_row;
                            valid_c = test_col;
                            break;
                        }

                        test_row = valid_r + diag - delta;
                        test_col = valid_c + diag;
                        if (pred(test_row, test_col))
                        {
                            next = true;
                            valid_r = test_row;
                            valid_c = test_col;
                            break;
                        }
                    }//end of for delta


                }
                if (next == true)
                {
                    break;
                }
            }//end of for diag

            if (next == false)
            {
                return make_tuple(valid_r, valid_c);
            }

        }//end of while

    };
    static void assembleMatches(Video_det_param &param, SparseMat &flagMat, int beg0, int end0, int beg1, int end1
                                , vector<pair<int, int>> &matches0, vector<pair<int, int>> &matches1)
    {
        Mat flag;
        flagMat.copyTo(flag);
        Mat recordMap = Mat::zeros(end0 - beg0, end1 - beg1, CV_8UC1);
        const int interval = static_cast<int>(param.timeInterval*param.used_fps);

        for (int i = beg0; i < end0; ++i)
        {

            for (int j = beg1; j < end1; ++j)
            {
                if (recordMap.data[i*recordMap.cols + j] !=0 || flagMat.ptr(i, j, false) == NULL)
                {
                    continue;
                }
                int valid_r, valid_c;
                std::tie(valid_r, valid_c) = go_bottom_right(recordMap, flagMat, i, j, interval);

                //record visited location
                for (int m = i; m < valid_r + 1; ++m)
                {
                    for (int n = j; n < valid_c + 1; ++n)
                    {
                        recordMap.data[m*recordMap.cols + n] = 255;
                    }
                }
                if (valid_r - i + 1 < interval || valid_c - j + 1 < interval)
                {
                    continue;
                }
                matches0.push_back(make_pair(i, valid_r));
                matches1.push_back(make_pair(j, valid_c));
            }
        }
    }
  
    bool my_cmp(const pair<int, int> &v1,const pair<int, int> &v2)
    {

        return v1.second > v2.second;
    }
    int DetEngine::detectSingleVideo(std::string videoName,vector<string> &jsonStr)
    {
        Mat videoFeats;
        if (vd::loadVideoFeat(videoName, videoFeats) == -1)
        {
            if (vd::exactFeat(videoName, param_, videoFeats, true) == -1)
            {
                return -1;
            }
        }
        
        Mat colorFeat = colorFeat_;
        int knn_num = 100;
        int vFrameCnt = videoFeats.rows;
        Mat knn_indices = Mat::zeros(vFrameCnt,knn_num,CV_32SC1);
        Mat knn_dists=Mat::zeros(vFrameCnt,knn_num,CV_32FC1);
        
        ::flann::Matrix<float> flann_queries((float*)videoFeats.data, videoFeats.rows, videoFeats.cols);
        ::flann::Matrix<int> flann_indices((int*)knn_indices.data, knn_indices.rows, knn_indices.cols);
        ::flann::Matrix<float> flann_dist((float*)knn_dists.data, knn_dists.rows, knn_dists.cols);

        featIndex_->knnSearch(flann_queries, flann_indices, flann_dist, knn_num, ::flann::SearchParams());
       //featIndex_->knnSearch(videoFeats, knn_indices, knn_dists, knn_num, cv::flann::SearchParams());
        
        map<int, int> votes;
        map<int,vector<tuple<int, int,int>>> match_frames;
        for (int i = 0; i != knn_indices.rows; ++i)
        {
            for (int j = 0; j != knn_indices.cols; ++j)
            {
                int pos_inTotalFeat = ((int*)knn_indices.data)[i*knn_num + j];
                float ratio= 1-((float*)knn_dists.data)[i*knn_num + j] /2;

                //if (j >= colorFeat_.rows ||  ratio  < 0.75)
                 //   continue;
                if (j>=colorFeat_.rows || ratio <0.7)
                {
                    continue;
                }
                int videoId = index2VideoId_[static_cast<size_t>(pos_inTotalFeat)];
                votes[videoId] += 1;

                //matches_maps[frameIndex].push_back(DMatch(i, GPos2LPos[pos_inTotalFeat],0.0));
                //match_frames[videoId].emplace_back(i, framePosMapG2L_[pos_inTotalFeat]);
                match_frames[videoId].emplace_back(i, framePosMapG2L_[pos_inTotalFeat], ratio *100);
            }
        }
        vector<pair<int, int>> votesMap(votes.begin(), votes.end());
        std::sort(votesMap.begin(), votesMap.end(),my_cmp );

        for (size_t i = 0; i < votesMap.size() && i < knn_num;++i)
        {
            int frameCnt_tar= frameCnts_[static_cast<int>(votesMap[i].first)];
            int sparseMatSize[2] = { vFrameCnt, frameCnt_tar };
            SparseMat flagMat(2, sparseMatSize, CV_8UC1);

            vector<tuple<int, int,int>> &matches = match_frames[votesMap[i].first];
            for (size_t j = 0; j != matches.size(); ++j)
            {
                flagMat.ref<uchar>(std::get<0>(matches[j]),std::get<1>(matches[j])) = std::get<2>(matches[j]);
            }

            Mat flag;
            flagMat.copyTo(flag);

            //typedef pair<int, int> FrameDuration;
            vector<pair<int, int> > det_initFrameMatches;
            vector<pair<int, int> > tar_initFrameMatches;

            assembleMatches(param_, flagMat, 0, videoFeats.rows, 0, frameCnt_tar, det_initFrameMatches, tar_initFrameMatches);

            vector<pair<int, int>> finalMatchResult_det(det_initFrameMatches.size());
            vector<pair<int, int>> finalMatchResult_tar(tar_initFrameMatches.size());

            for (int j = 0; j != det_initFrameMatches.size(); ++j)
            {
                finalMatchResult_det[j].first = static_cast<int>(det_initFrameMatches[j].first / param_.used_fps);
                finalMatchResult_det[j].second = static_cast<int>(det_initFrameMatches[j].second / param_.used_fps);
                finalMatchResult_tar[j].first = static_cast<int>(tar_initFrameMatches[j].first / param_.used_fps);
                finalMatchResult_tar[j].second = static_cast<int>(tar_initFrameMatches[j].second / param_.used_fps);
            }


            if (finalMatchResult_tar.size() == 0)
            {
                return 0;
            }

            string result;
            genJsonStr(videoName, finalMatchResult_det, finalMatchResult_tar, result);

            jsonStr.push_back(result);


            Mat resultImg = drawDetResult(vFrameCnt/param_.used_fps, finalMatchResult_det, frameCnt_tar/param_.used_fps, finalMatchResult_tar);
            saveResult(videoName, videoNames_[votesMap[i].first], result, resultImg);
        }
        
        return 0;
    }

    int DetEngine::procVideos()
    {
        vector<int> invaildNamesInd;

        int validNames = 0;
        vector<string> videoNames;

        for (size_t i = 0; i != videoNames_.size();++i)
        {
            Mat hists;

            if (vd::loadVideoFeat(videoNames_[i], hists) == -1)
            {
                if (vd::exactFeat(videoNames_[i], param_, hists, true) == -1)
                {
                    continue;
                }
            }

            colorFeat_.push_back(hists);
            frameCnts_.push_back(hists.rows);
            index2VideoId_.insert(index2VideoId_.end(), hists.rows, validNames);
            validNames++;
            videoNames.push_back(videoNames_[i]);
        }
        videoNames_.swap(videoNames);

        return 0;
    }

    int DetEngine::trainFlannIndex()
    {
        if (colorFeat_.rows == 0)
        {
            return -1;
        }
        //const int ORB_FEAT_DIM = 32;

        //FeatIndexType * index = new FeatIndexType(::flann::KDTreeIndexParams());;
        FeatIndexType *index = new FeatIndexType(::flann::LinearIndexParams());
        //index->build(colorFeat_, cv::flann::HierarchicalClusteringIndexParams(), cvflann::FLANN_DIST_L1);

        Mat temp = colorFeat_;

        ::flann::Matrix<float> featMat((float*)colorFeat_.data, colorFeat_.rows, colorFeat_.cols);
        index->buildIndex(featMat);
        featIndex_.reset(index);

        return 0;
    }
    int DetEngine::saveIndex()
    {
        if (changed_)
        {
            featIndex_->save(string(dataPath_ + "/featIndex"));
        }

        return 0;
    }

    int DetEngine::saveState()
    {
        if (!changed_)
        {
            return 0;
        }

        //save other state
        shared_ptr<FILE> file(fopen(string(dataPath_ + "/eng_state").c_str(), "wb"),fclose);

        int n = 0;
        n = static_cast<int>(videoNames_.size());
        fwrite(&n, sizeof(n), 1, file.get());

        for (size_t i = 0; i != videoNames_.size();++i)
        {
            n = static_cast<int>(videoNames_[i].size());
            fwrite(&n, sizeof(n), 1, file.get());
            fwrite(videoNames_[i].data(), 1, n, file.get());
            fwrite(&frameCnts_[i], sizeof(frameCnts_[i]), 1, file.get());
        }

        //n = colorFeat_.total() * colorFeat_.elemSize();
        n = colorFeat_.rows;
        fwrite(&n, sizeof(n), 1, file.get());

        n = colorFeat_.cols;
        fwrite(&n, sizeof(n), 1, file.get());

        //n = colorFeat_.elemSize();
        //fwrite(&n, sizeof(n), 1, file.get()); fwrite(&n, sizeof(n), 1, file.get());

        fwrite(colorFeat_.data, sizeof(float), colorFeat_.total(),file.get());

        //save index
        featIndex_->save(string(dataPath_ + "/featIndex"));

        return 0;
    }

    int DetEngine::readState()
    {
        shared_ptr<FILE> file(fopen(string(dataPath_ + "/eng_state").c_str(), "rb"), [](FILE *fp){if (fp != NULL) fclose(fp); });

        if (!file.get())
        {
            cout << "can not open file /eng_state" << endl;
            return -1;
        }
        const int bufSize = 511;
        char buf[bufSize] ;
        int n = 0; 
        
        fread(&n, sizeof(n), 1, file.get());
        videoNames_.resize(n);
        frameCnts_.resize(n);
        for (size_t i = 0; i < videoNames_.size(); i++)
        {
            memset(buf, 0, bufSize);
            fread(&n, sizeof(n), 1, file.get());
            fread(buf, 1, n, file.get());
            videoNames_[i] = string(buf);
            fread(&frameCnts_[i], sizeof(frameCnts_[i]), 1,file.get());
            index2VideoId_.insert(index2VideoId_.end(), static_cast<int>(frameCnts_[i]), static_cast<int>(i));
        }

        int r, c;
        fread(&r, sizeof(n), 1, file.get());
        fread(&c, sizeof(n), 1, file.get());
        colorFeat_ = Mat::zeros(r, c, CV_32FC1);
        fread(colorFeat_.data, sizeof(float),r*c, file.get());

        genFrameMap();

        try
        {
            ::flann::Matrix<float> flann_feat((float*)colorFeat_.data, colorFeat_.rows, colorFeat_.cols);
            featIndex_.reset(new FeatIndexType(flann_feat, ::flann::SavedIndexParams(string(dataPath_ + "/featIndex"))));
        }
        catch (cv::Exception e)
        {
            cout << string("loadVideoIndex : ") + e.what();
            return -1;
        }

        return 0;
    }

    int DetEngine::train()
    {
        clock_t t;
        t = clock();
        if(procVideos()==-1)
            return -1;
        printf("exact feature time :%lf \n", (double)(clock() - t) / CLOCKS_PER_SEC);
        genFrameMap();

        clock_t t1;
        t1 = clock();
        if (trainFlannIndex())
            return -1;
        printf("train feature index time: %lf \n", (double)(clock_t() - 1 / CLOCKS_PER_SEC));
      

        return 0;
    }

    int DetEngine::addVideos(vector<string> &videoNames)
    {
        changed_ = true;
        videoNames_.insert(videoNames_.end(),videoNames.begin(), videoNames.end());

        return 0;
    }

    int DetEngine::genFrameMap()
    {
        //prepare a map of features in target video from  global position to local position
        framePosMapG2L_=vector<int>(index2VideoId_.size());
        int LPos = 0;
        for (int i = 0; i< index2VideoId_.size(); ++i)
        {
            if (i>0 && index2VideoId_[i] != index2VideoId_[i - 1])
            {
                LPos = 0;
            }
            framePosMapG2L_[i] = LPos++;
        }

        return 0;
    }
}
