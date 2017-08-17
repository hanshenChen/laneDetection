#pragma once

#include <opencv2/core/core.hpp>

class visualConfig;

class Cipm {
public:

	void Init(visualConfig &vc);
	void IPM(cv::Mat &src, cv::Mat &dst);
	void IPMInv(cv::Mat &src, cv::Mat &dst);
private:
	cv::Mat tsf_ipm;
	cv::Mat tsf_ipm_inv;
};

class Transformer {
public:
	void Init(visualConfig &cfg);
	void getDetectRoi(cv::Mat &src, cv::Mat &dst);
	void dispDetectRoi(cv::Mat &dst);
	void mapToRealImg(cv::Mat &src, cv::Mat &dst);
private:
	Cipm ipm;
	cv::Rect ipmRoi;
	cv::Rect detectRoi;
};