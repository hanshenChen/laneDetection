#ifndef VLD_IMG_TRANSFORM_H_
#define VLD_IMG_TRANSFORM_H_

#include <opencv2/core/core.hpp>


class GaussianFilter {
public:
    void Init(int _sigma_x, int _sigma_y);
    void Filter(cv::Mat src, cv::Mat &dst);
private:
    cv::Mat gaussian_kernel_x_;
    cv::Mat gaussian_kernel_y_;
    int sigma_x_;
    int sigma_y_;
    void onGaussianChange();
};

class Parabola;

class ToBin{
public:
	bool threshold(cv::Mat &src, cv::Mat &dst, int blocksize, int threshold, bool debug_enable);

	cv::Mat blendingImag(cv::Mat &src, int frameCount);
	bool averageBinImage(cv::Mat &src, cv::Mat &dst, int num);
	bool multiFrameBlendingWithCalc(cv::Mat &src, cv::Mat &dst, Parabola &leftParabola, Parabola &rightParabola, int num);
private:
	cv::Mat hsv_threshold(cv::Mat &img, int thresh[][3]);
	void blackwb_thresh(cv::Mat& img, cv::Mat& out, int blocksize, int threshold);
	int normal_thresh(cv::Mat &inImage, cv::Mat &outImage, int lowerThresh, int upperThresh);
	cv::Mat averageImagWithOffset(cv::Mat &src, int offsetX, int offsetY, int num);
};

#endif
