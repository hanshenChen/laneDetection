#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include "transformer.h"
#include "config.h"

using  namespace cv;


void Transformer::Init(visualConfig &cfg) {
	ipm.Init(cfg);
	ipmRoi = Rect(cfg.roiStartX, cfg.roiStartY, cfg.roiWidth, cfg.roiHeight);
	detectRoi = Rect(cfg.decRoiStartX, cfg.decRoiStartY, cfg.decRoiWidth, cfg.decRoiHeight);
}

void Transformer::getDetectRoi(Mat &src, Mat &dst) {
	Mat srcRoi;
	Mat dstRoi;
	srcRoi = src(ipmRoi);
	//imshowWithShot("srcRoi", srcRoi);
	ipm.IPM(srcRoi, dstRoi);
	//imshowWithShot("dstRoi", dstRoi);
	dst = dstRoi(detectRoi);
}

//srcRoi can be color or gray
void Transformer::mapToRealImg(cv::Mat &srcRoi, cv::Mat &dst) {

	Mat panelImg(Size(ipmRoi.width, ipmRoi.height), srcRoi.type());
	panelImg = Scalar::all(0);
	Mat detectImg = panelImg(detectRoi);

	srcRoi.copyTo(detectImg);
	ipm.IPMInv(panelImg, panelImg);

	Mat ipmImg = dst(ipmRoi);
	Mat temp;

	double alpha = 0.8;
	cv::addWeighted(panelImg, alpha, ipmImg, 1, 0, ipmImg);
}

//input orginal image
void Transformer::dispDetectRoi(Mat &dst) {
	Mat colorEdge(Size(detectRoi.width, detectRoi.height), dst.type());
	colorEdge = Scalar::all(0);
	rectangle(colorEdge, Point(0, 0), Point(colorEdge.cols - 1, colorEdge.rows - 1),
		Scalar(0, 0, 255), 1, 8, 0);
	mapToRealImg(colorEdge, dst);
}


void Cipm::Init(visualConfig &vc) {

	Point2f src[4];
	Point2f dst[4];

	int roi_width = vc.roiWidth;

	src[0].x = 0;
	src[0].y = 0;
	src[1].x = vc.ipmStartX;
	src[1].y = vc.roiHeight;
	src[2].x = vc.ipmEndX;
	src[2].y = vc.roiHeight;
	src[3].x = roi_width;//+30
	src[3].y = 0;

	dst[0].x = 0;
	dst[0].y = 0;
	dst[1].x = 0;
	dst[1].y = vc.roiHeight;
	dst[2].x = roi_width;
	dst[2].y = vc.roiHeight;
	dst[3].x = roi_width;
	dst[3].y = 0;

	tsf_ipm = getPerspectiveTransform(dst, src);
	tsf_ipm_inv = tsf_ipm.inv();
}

void Cipm::IPM(Mat &src, Mat &dst) {
	warpPerspective(src, dst, tsf_ipm, src.size());
}


void Cipm::IPMInv(Mat &src, Mat &dst) {
	warpPerspective(src, dst, tsf_ipm_inv, src.size());
}