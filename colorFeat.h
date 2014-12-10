
#include <opencv2/opencv.hpp>

#include <stack>
#define COLOR_SPACE_BGR 1
#define COLOR_SPACE_HLS 2


namespace vd
{
    class ColorHist
    {
    public:
        ColorHist()
        {
            m_binsNum[0] = 6;
            m_binsNum[1] = 6;
            m_binsNum[2] = 6;
        }
        ColorHist(int binsNum1, int binsNum2, int binsNum3)
        {
            m_binsNum[0] = binsNum1;
            m_binsNum[1] = binsNum2;
            m_binsNum[2] = binsNum3;

        }
        void computeFeat(cv::Mat &src, cv::Mat &hist, int colorSpaceType = COLOR_SPACE_HLS, int normType = cv::NORM_L1);
        int getFeatDim();

    private:
        void initSteps();

    public:
        int m_binsNum[3];
        int m_steps[3];
    };


#define PI 3.141569254
    class CEdgeHist 
    {
    public:
        CEdgeHist()
        {
            EDGE_HIST_SIZE = 64;
        }
        void computeFeat(cv::Mat &src, cv::Mat &edgeHist, int colorSpaceType = COLOR_SPACE_HLS, int normType = cv::NORM_L1);
        int getFeatDim()
        {
            return EDGE_HIST_SIZE;
        }
    private:
        int calEdgeHistogram(IplImage *image,/*IplImage * dst,*/float * edge_hist_feature);
        int EDGE_HIST_SIZE;
    };

    /************************************************************************/
    /* 颜色聚合向量                                                                     */
    /************************************************************************/

    class ColorCoherenceVec 
    {
    public:
        ColorCoherenceVec()
        {
            m_binsNum[0] = 6;
            m_binsNum[1] = 3;
            m_binsNum[2] = 4;
            m_thresholdFactor = 0.01f;
        }
        ColorCoherenceVec(int binsNum1, int binsNum2, int binsNum3, float threshold = 0.01)
        {
            m_binsNum[0] = binsNum1;
            m_binsNum[1] = binsNum2;
            m_binsNum[2] = binsNum3;
            m_thresholdFactor = threshold;
        }
        void computeFeat(cv::Mat &src, cv::Mat &ccv, int colorSpaceType = COLOR_SPACE_HLS, int normType = cv::NORM_L1);
        int getFeatDim();

    private:
        void initSteps();
        void RegionGrow(cv::Mat &img, cv::Mat &flag, std::stack<cv::Point>* &pStack, int &connectedPixNum);

    public:
        int m_binsNum[3];
        int m_steps[3];
        float m_thresholdFactor;    //连通区域像素计算比例，默认图像总像素的1%
    };

}// namespace vd