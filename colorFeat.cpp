

#include "colorFeat.h"

#include <stack>

namespace vd
{

    using namespace cv;
    using namespace std;
    void ColorHist::initSteps()
    {
        m_steps[0] = static_cast<int>(std::ceil((float)256 / m_binsNum[0]));
        m_steps[1] = static_cast<int>(std::ceil((float)256 / m_binsNum[1]));
        m_steps[2] = static_cast<int>(std::ceil((float)256 / m_binsNum[2]));
    }

    void ColorHist::computeFeat(Mat &src, Mat &hist, int colorSpaceType, int normType)
    {
        assert(src.channels() == 3);
        assert(src.type() == CV_8UC3);

        hist = Mat::zeros(1, getFeatDim(), CV_32FC1);

        Mat img;

        if (colorSpaceType == COLOR_SPACE_HLS)
        {
            cvtColor(src, img, CV_BGR2HLS);
            for (int r = 0; r < img.rows; ++r)
            {
                for (int c = 0; c < img.cols; ++c)
                {
                    int tmp = img.at<cv::Vec3b>(r, c)[0];
                    img.at<cv::Vec3b>(r, c)[0] = static_cast<uchar>((float)tmp * 2 / 360 * 255);
                }
            }
        }
        else
        {
            img = src;
        }

        initSteps();


        for (int r = 0; r < img.rows; ++r)
        {
            for (int c = 0; c < img.cols; ++c)
            {

                cv::Vec3b index = img.at<cv::Vec3b>(r, c);
                index[0] = index[0] / m_steps[0];
                index[1] = index[1] / m_steps[1];
                index[2] = index[2] / m_steps[2];

                //int location = index[0] * m_binsNum[1] * m_binsNum[2] + index[1] * m_binsNum[2] + index[2];
                int location = index[0] * m_binsNum[1] + index[1];
                hist.at<float>(location) += 1;
            }
        }

        //L1 normalization

        if (normType == -1)
        {
            return;
        }
        normalize(hist, hist, 1.0, 0, normType);
    }


    int ColorHist::getFeatDim()
    {
        //return m_binsNum[0] * m_binsNum[1] * m_binsNum[2];
        return m_binsNum[0] * m_binsNum[1];
    }


    void CEdgeHist::computeFeat(Mat &src, Mat &edgeHist, int colorSpaceType, int normType)
    {


        edgeHist = Mat::zeros(1, getFeatDim(), CV_32FC1);

        Mat img;

        if (src.channels() == 3)
        {
            cvtColor(src, img, CV_BGR2GRAY);
        }
        else
        {
            img = src;
        }
        IplImage image = IplImage(img);
        //IplImage dst = IplImage(edgeHist);
        calEdgeHistogram(&image, (float*)edgeHist.data);


        if (normType == -1)
        {
            return;
        }
        normalize(edgeHist, edgeHist, 1.0, 0, normType);

    }

    int CEdgeHist::calEdgeHistogram(IplImage *image,/*IplImage * dst,*/float * edge_hist_feature)
    {
        if (NULL == image || NULL == edge_hist_feature)
            return 1;

        CvHistogram *hist = 0; // 直方图
        IplImage* canny;//边缘图像
        IplImage* gradient_im;
        IplImage* dx; // x方向的sobel差分
        IplImage* dy; // y方向的sobel差分 

        CvMat* canny_m;
        CvMat* gradient; // 梯度值
        CvMat* gradient_dir; //梯度的方向
        CvMat* dx_m; // 
        CvMat* dy_m;
        CvMat* mask;
        CvSize  size;

        int i, j;
        float theta;
        float max_val;

        int hdims = EDGE_HIST_SIZE;     // 划分HIST的个数，越高越精确
        float hranges_arr[] = { -PI / 2, PI / 2 }; // 直方图的上界和下界
        float* hranges = hranges_arr;

        size = cvGetSize(image);
        canny = cvCreateImage(cvGetSize(image), 8, 1);//边缘图像
        dx = cvCreateImage(cvGetSize(image), 32, 1);//x方向上的差分
        dy = cvCreateImage(cvGetSize(image), 32, 1);
        gradient_im = cvCreateImage(cvGetSize(image), 32, 1);//梯度图像

        if (NULL == canny || NULL == dx || NULL == dy || NULL == gradient_im)
            return 2;

        canny_m = cvCreateMat(size.height, size.width, CV_32FC1);//边缘矩阵
        dx_m = cvCreateMat(size.height, size.width, CV_32FC1);
        dy_m = cvCreateMat(size.height, size.width, CV_32FC1);

        gradient = cvCreateMat(size.height, size.width, CV_32FC1);//梯度矩阵
        gradient_dir = cvCreateMat(size.height, size.width, CV_32FC1);//梯度方向矩阵
        mask = cvCreateMat(size.height, size.width, CV_32FC1);//掩码
        if (NULL == canny_m || NULL == dx_m || NULL == dy_m || NULL == gradient || NULL == gradient_dir || NULL == mask)
            return 2;


        cvSmooth(image, image, CV_GAUSSIAN, 7);
        //GaussianBlur(image,image,Size(9,9),1);
        cvCanny(image, canny, 50, 170, 3);//边缘检测
        //cvCopy(canny,dst);

        cvConvert(canny, canny_m);//把图像转换为矩阵
        cvSobel(image, dx, 1, 0, 3);// 一阶X方向的图像差分 :dx
        cvSobel(image, dy, 0, 1, 3);// 一阶Y方向的图像差分 :dy
        cvConvert(dx, dx_m);
        cvConvert(dy, dy_m);

        //用cvAdd近似计算梯度
        cvAdd(dx_m, dy_m, gradient); // 梯度值
        cvDiv(dx_m, dy_m, gradient_dir); // 梯度方向
        for (i = 0; i < size.height; i++){
            for (j = 0; j < size.width; j++){
                if (cvmGet(canny_m, i, j) != 0 && cvmGet(dx_m, i, j) != 0){
                    theta = cvmGet(gradient_dir, i, j);
                    theta = atan(theta);
                    cvmSet(gradient_dir, i, j, theta);
                }
                else{
                    cvmSet(gradient_dir, i, j, 0);
                }
            }
        }

        hist = cvCreateHist(1, &hdims, CV_HIST_ARRAY, &hranges, 1);  // 创建一个指定尺寸的直方图，并返回创建的直方图指针
        //cvZero(hist); // 清0；
        cvConvert(gradient_dir, gradient_im);//把梯度方向矩阵转化为图像
        cvCalcHist(&gradient_im, hist, 0, canny); // 计算直方图
        cvGetMinMaxHistValue(hist, 0, &max_val, 0, 0);  // 只找最大值
        cvConvertScale(hist->bins, hist->bins, max_val ? 255. / max_val : 0., 0); // 缩放 bin 到区间 [0,255] ，比例系数

        for (int i = 0; i < hdims; i++){
            edge_hist_feature[i] = cvGetReal1D(hist->bins, i) / 255.0;
        }

        cvReleaseHist(&hist);
        cvReleaseImage(&gradient_im);
        cvReleaseImage(&canny);
        cvReleaseImage(&dx);
        cvReleaseImage(&dy);

        cvReleaseMat(&canny_m);
        cvReleaseMat(&gradient);
        cvReleaseMat(&gradient_dir);
        cvReleaseMat(&dx_m);
        cvReleaseMat(&dy_m);
        cvReleaseMat(&mask);

        return 0;
    }


    /************************************************************************/
    /*  class implementation of color coherence vector                                                  */
    /************************************************************************/

    int ColorCoherenceVec::getFeatDim()
    {
        return m_binsNum[0] * m_binsNum[1] * m_binsNum[2] * 2;
    }

    void ColorCoherenceVec::initSteps()
    {
        m_steps[0] = std::ceil((float)256 / m_binsNum[0]);
        m_steps[1] = std::ceil((float)256 / m_binsNum[1]);
        m_steps[2] = std::ceil((float)256 / m_binsNum[2]);
    }


    void ColorCoherenceVec::computeFeat(Mat &src, Mat &ccv, int colorSpaceType, int normType)
    {
        assert(src.channels() == 3);
        assert(src.type() == CV_8UC3);
        //if (ccv.data==NULL)
        //{
        //	ccv = Mat::zeros(1,getFeatDim(),CV_32FC1);
        //}
        //else
        //{
        //	assert(ccv.type() == CV_32F);
        //	assert(ccv.channels() == 1);
        //}

        ccv = Mat::zeros(1, getFeatDim(), CV_32FC1);

        Mat img;

        if (colorSpaceType == COLOR_SPACE_HLS)
        {
            cvtColor(src, img, CV_BGR2HLS);
            for (int r = 0; r < img.rows; ++r)
            {
                for (int c = 0; c < img.cols; ++c)
                {
                    int tmp = img.at<Vec3b>(r, c)[0];
                    img.at<Vec3b>(r, c)[0] = (float)tmp * 2 / 360 * 255;
                }
            }
        }
        else
        {
            img = src;
        }

        initSteps();

        Mat bluredImg;
        //blur(img,bluredImg,Size(3,3));
        bluredImg = img;


        Mat flag = Mat::zeros(bluredImg.rows, bluredImg.cols, CV_8UC1);

        std::stack<Point> *pStack = new std::stack<Point>();

        for (int r = 0; r < bluredImg.rows; ++r)
        {
            for (int c = 0; c < bluredImg.cols; ++c)
            {


                if (flag.at<uchar>(r, c) == 0)
                {
                    int connectedPixNum = 0;


                    Point pt(c, r);
                    pStack->push(pt);
                    flag.at<uchar>(pt) = 1;


                    RegionGrow(bluredImg, flag, pStack, connectedPixNum);

                    Vec3b index = img.at<Vec3b>(pt);
                    index[0] = index[0] / m_steps[0];
                    index[1] = index[1] / m_steps[1];
                    index[2] = index[2] / m_steps[2];

                    //根据颜色量化bin确定在结果直方图中的位置
                    int location = index[0] * m_binsNum[1] * m_binsNum[2] + index[1] * m_binsNum[2] + index[2];
                    location *= 2;


                    //这里根据位置在把新得到的聚合像素或者非聚合像素加到相应位置，需要原始ccv矩阵是0初始化的
                    if (connectedPixNum >= int(m_thresholdFactor*bluredImg.total()))
                        ccv.at<float>(location) += connectedPixNum;
                    else
                        ccv.at<float>(location + 1) += connectedPixNum;
                }
            }
        }

        if (normType == -1)
        {
            return;
        }
        normalize(ccv, ccv, 1.0, 0, normType);

        delete pStack;
    }

    //四连通,img 原图像，flag标记是否访问过的图像
    //访问下一个结点满足两个条件，一是没超边界并且每访问过，二是满足颜色相似

    void ColorCoherenceVec::RegionGrow(Mat &img, Mat &flag, std::stack<Point>*& pStack, int &connectedPixNum)
    {

        while (pStack->size() != 0)
        {
            Point pt = pStack->top();
            connectedPixNum += 1;
            //flag.at<uchar>(pt)=1;

            pStack->pop();

            Vec3b index = img.at<Vec3b>(pt);
            index[0] = index[0] / m_steps[0];
            index[1] = index[1] / m_steps[1];
            index[2] = index[2] / m_steps[2];

            //go up
            if (pt.y - 1 >= 0 && flag.at<uchar>(pt.y - 1, pt.x) == 0)
            {
                Vec3b indexNext = img.at<Vec3b>(pt.y - 1, pt.x);
                indexNext[0] = indexNext[0] / m_steps[0];
                indexNext[1] = indexNext[1] / m_steps[1];
                indexNext[2] = indexNext[2] / m_steps[2];
                if (index[0] == indexNext[0] && index[1] == indexNext[1] && index[2] == indexNext[2])
                {
                    pStack->push(Point(pt.x, pt.y - 1));
                    flag.at<uchar>(Point(pt.x, pt.y - 1)) = 1;
                }
            }
            //go down
            if (pt.y + 1 < img.rows && flag.at<uchar>(pt.y + 1, pt.x) == 0)
            {
                Vec3b indexNext = img.at<Vec3b>(pt.y + 1, pt.x);
                indexNext[0] = indexNext[0] / m_steps[0];
                indexNext[1] = indexNext[1] / m_steps[1];
                indexNext[2] = indexNext[2] / m_steps[2];
                if (index[0] == indexNext[0] && index[1] == indexNext[1] && index[2] == indexNext[2])
                {
                    pStack->push(Point(pt.x, pt.y + 1));
                    flag.at<uchar>(Point(pt.x, pt.y + 1)) = 1;
                }
            }
            //go right
            if (pt.x + 1 < img.cols && flag.at<uchar>(pt.y, pt.x + 1) == 0)
            {
                Vec3b indexNext = img.at<Vec3b>(pt.y, pt.x + 1);
                indexNext[0] = indexNext[0] / m_steps[0];
                indexNext[1] = indexNext[1] / m_steps[1];
                indexNext[2] = indexNext[2] / m_steps[2];
                if (index[0] == indexNext[0] && index[1] == indexNext[1] && index[2] == indexNext[2])
                {
                    pStack->push(Point(pt.x + 1, pt.y));
                    flag.at<uchar>(Point(pt.x + 1, pt.y)) = 1;
                }
            }
            //go left
            if (pt.x - 1 >= 0 && flag.at<uchar>(pt.y, pt.x - 1) == 0)
            {
                Vec3b indexNext = img.at<Vec3b>(pt.y, pt.x - 1);
                indexNext[0] = indexNext[0] / m_steps[0];
                indexNext[1] = indexNext[1] / m_steps[1];
                indexNext[2] = indexNext[2] / m_steps[2];
                if (index[0] == indexNext[0] && index[1] == indexNext[1] && index[2] == indexNext[2])
                {
                    pStack->push(Point(pt.x - 1, pt.y));
                    flag.at<uchar>(Point(pt.x - 1, pt.y)) = 1;
                }
            }
        }
    }

}
