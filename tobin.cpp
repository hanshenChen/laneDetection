#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>  
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>  
#include <stdio.h>
#include "tobin.h"
#include "config.h"
#include "utils.h"

#define GAUSSIANSIZE 25
using  namespace cv;
using  namespace std;

void ToBin::blackwb_thresh(Mat& img, Mat& out, int blocksize, int threshold)
{
	unsigned char *raw = (unsigned char*)(img.data);
	out.setTo(0);

	int aux = 0;
	int x = 0, y = 0;
	const int w = img.cols - blocksize - 1;

	/*for (y = 0; y < img.rows; ++y)
	{
		auto raw = img.ptr(y);
		for (x = blocksize; x < w; ++x)
		{
			aux = 2 * raw[x];
			aux -= raw[x - blocksize];
			aux -= raw[x + blocksize];

			aux -= abs(raw[x - blocksize] - raw[x + blocksize]);

			aux *= 4;// more contrast

			if (aux >= threshold)
			{
				out.at<unsigned char>(y, x) = 255;
			}
		}
	}*/

	int aux1, aux2;
	for (y = 0; y < img.rows; ++y)
	{
		auto raw = img.ptr(y);
		for (x = blocksize; x < w; ++x)
		{
			aux1 = raw[x] -raw[x + blocksize];
			aux2 = raw[x] -raw[x - blocksize];
			int d= aux1+ aux2 - abs(raw[x - blocksize] - raw[x + blocksize]);

			//aux *= 4;// more contrast

			if ((aux1 > 0)&& (aux1 > 0)&& d>(0.25*raw[x]))
			{
				out.at<unsigned char>(y, x) = 255;
			}
			else {
				out.at<unsigned char>(y, x) = 0;
			}
		}
	}
	//erode(out, out, getStructuringElement(MORPH_RECT, Size(5, 5)));
	//dilate(out, out, getStructuringElement(MORPH_RECT, Size(3, 3)));

	////Mat line = Mat::ones(10, 1, CV_8UC1);
	////now apply the morphology open operation
	Mat line2 = Mat::ones(1, 3, CV_8UC1);
	morphologyEx(out, out, MORPH_OPEN, line2, Point(-1, -1));
}


int ToBin::normal_thresh(Mat &inImage, Mat &outImage,int lowerThresh,int upperThresh)
{
	for (int i = 0; i < inImage.rows; i++) {
		for (int j = 0; j < inImage.cols; j++)
		{
			if ((inImage.at<uchar>(i, j)>lowerThresh) && (inImage.at<uchar>(i, j) <= upperThresh))
				outImage.at<uchar>(i, j) = 255;// inImage.at<uchar>(i, j);
			else
				outImage.at<uchar>(i, j) = 0;
		}
	}
	return 1;
}


Mat ToBin::hsv_threshold(Mat &img, int thresh[][3]) {
	//Find white and yellow based on range and apply threshold
	//Output array of the same size as the input image
	//Convert to HSV color space
	Mat hsv;
	cvtColor(img, hsv, COLOR_RGB2HSV);
    //define hsv ranges for yellow and white
	_InputArray lower_yellow(thresh[0], 3);
	_InputArray upper_yellow(thresh[1], 3);
	_InputArray lower_white(thresh[2], 3);
	_InputArray upper_white(thresh[3], 3);
	//threshold hsv_img with defined ranges
	Mat yellow_hsv;
	inRange(hsv, lower_yellow, upper_yellow, yellow_hsv);
	Mat white_hsv;
	inRange(hsv, lower_white, upper_white, white_hsv);
	Mat binary_output =Mat(img.rows, img.cols, CV_8UC1);//可优化 np.zeros_like(hsv[:, : , 0])
	binary_output= Scalar::all(0);
	normal_thresh(yellow_hsv, binary_output, 0, 255);
	normal_thresh(white_hsv, binary_output, 0, 255);
	return binary_output;
}

#define METHOD_ONE   
#define OVERLY_FRAME   2
bool ToBin::threshold(Mat &src, Mat &dst,int blocksize, int threshold, bool debug_enable) {
	Mat grayRoi;
	cvtColor(src, grayRoi, CV_RGB2GRAY);

#ifdef METHOD_ONE
	Mat inImg = grayRoi;
	static int count = 0;
	count++;
	int framecount = count % OVERLY_FRAME + 1;//1,..,OVERLY_FRAME

	if (debug_enable == true) {
		char cbuf[50];
		memset(cbuf, 0, sizeof(cbuf));
		sprintf(cbuf, "frame%d", framecount);
		imshowWithShot(cbuf, inImg);
	}

	Mat result = blendingImag(inImg, framecount);
	if (framecount >= OVERLY_FRAME)
	{
		if(debug_enable==true)
			imshowWithShot("combile img", result);
		blackwb_thresh(result, dst, blocksize, threshold);
		//int thresh3[4][3] = { { 30, 30, 125 },{ 50, 255, 255 },{ 0, 0, 230 },{ 180, 25, 255 } };
		int thresh3[4][3] = { { 20, 100, 100 },{ 35, 255, 255 },{ 0, 0, 230 },{ 180, 25, 255 } };
		Mat hsv = hsv_threshold(src, thresh3);
		//imshowWithShot("open image", hsv);
		Mat line2 = Mat::ones(5, 1, CV_8UC1);
		morphologyEx(dst, dst, MORPH_OPEN, line2, Point(-1, -1));
		return true;
	}

#else

	// Perform a Gaussian blur
	GaussianBlur(src, src, Size(3, 3), 0, 0);
	// 初始化自适应阈值参数  
	const int maxVal = 255;
	/* 自适应阈值算法
	0：ADAPTIVE_THRESH_MEAN_C
	1: ADAPTIVE_THRESH_GAUSSIAN_C
	阈值类型
	0: THRESH_BINARY
	1: THRESH_BINARY_INV */
	int adaptiveMethod = 0;
	int thresholdType = 0;
	// 图像自适应阈值操作  
	adaptiveThreshold(src, dst,
		maxVal, adaptiveMethod,
		thresholdType, blocksize,
		threshold*-1);
	//GaussianBlur(frameEdge, frameEdge, Size(5, 5), 0, 0);
	//imshow("frameRoi", frameEdge);
	int minThreshold = 150;
	int maxThreshold = 255;
	//normal_thresh(frameEdge, frameEdge,minThreshold,maxThreshold);

#endif
	//averageBinImage(temp, dst, 3);
	//imshow("open image", gray_image);
	//Sobel(frameEdge, frameDection, CV_8U, 1, 0, 3, 1, 1);
	/*blur(gray_image, gray_image, Size(5, 5));	// 对图像进行均值滤波
	Mat closeImage;
	int structElemnetSize = 1;
	Mat element = getStructuringElement(MORPH_RECT, Size( 3,2*structElemnetSize + 1));	//闭操作
	erode(gray_image, closeImage, element);
	dilate(closeImage, closeImage, element);
	//Mat element = getStructuringElement(MORPH_ELLIPSE, Size(1, 2 * structElemnetSize + 1));
	//morphologyEx(gray_image, closeImage, MORPH_OPEN, element);

	Mat element = getStructuringElement(MORPH_RECT, Size( 3,2*structElemnetSize + 1));	//膨胀
	line2 = Mat::ones(5, 2, CV_8UC1);
	//erode(gray_image, closeImage, element);
	dilate(gray_image, gray_image, line2);*/

	/// 使用Laplace函数
	/*	Mat abs_dst;
	Mat src, src_gray, dst;
	int kernel_size = 3;
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	Laplacian(gray_image, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT);
	//ddepth: 输出图像的深度。 因为输入图像的深度是 CV_8U ，这里我们必须定义 ddepth = CV_16S 以避免外溢。
	convertScaleAbs(dst, abs_dst);//求绝对值；
	imshowWithShot("dst", abs_dst);*/
	return false;
}

Mat ToBin::blendingImag(Mat &src, int frameCount) {
	static Mat dst;
	if (dst.empty())
	{
		dst.create(Size(src.cols, src.rows), src.type());
		dst = Scalar::all(0);
	}

	if (frameCount == 1) {   //1
		dst = Scalar::all(0);
		for (int i = 0; i < src.rows; i++) {//行
			uchar *pDst = dst.ptr<uchar>(i);
			for (int j = 0; j < src.cols; j++) {
				pDst[j] = src.ptr<uchar>(i)[j];

			}
		}
	}
	else {                  //2,3
		for (int i = 0; i < src.rows; i++) {//行
			uchar *pDst = dst.ptr<uchar>(i);
			for (int j = 0; j < src.cols; j++) {//列
				int temp = (src.ptr<uchar>(i)[j] - pDst[j]);
				if (temp > 10)
					pDst[j] = src.ptr<uchar>(i)[j];
			}
		}
	}

	return dst;
}

Mat ToBin::averageImagWithOffset(Mat &src, int offsetX, int offsetY, int num) {
	offsetX = 0;
	offsetY = 0;
	static int count = 0;
	static Mat dst;
	if (dst.empty())
	{
		dst.create(Size(src.cols * 2, src.rows * 2), src.type());
		dst = Scalar::all(0);
	}

	if (count >= num) {
		count = 0;
		dst = Scalar::all(0);
	}

	count++;
	if (count == 1) {
		for (int i = 0; i < src.rows; i++) {//行
			uchar *pDst = dst.ptr<uchar>(i + offsetY);
			for (int j = 0; j < src.cols; j++) {
				pDst[j + offsetX] = src.ptr<uchar>(i)[j];

			}
		}
	}
	else {
		for (int i = 0; i < src.rows; i++) {//行
			uchar *pDst = dst.ptr<uchar>(i + offsetY);
			for (int j = 0; j < src.cols; j++) {//列
												/*	int temp = (pDst[j + offsetX] + src.ptr<uchar>(i)[j])*3/4;
												if (temp > 255)
												pDst[j + offsetX] = 255;
												else
												pDst[j + offsetX] = temp;*/
				int temp = (src.ptr<uchar>(i)[j] - pDst[j + offsetX]);
				if (temp > 20)
					pDst[j + offsetX] = src.ptr<uchar>(i)[j];


			}
		}
	}

	return dst;
}

bool ToBin::averageBinImage(Mat &src, Mat &dst, int num) {
	static vector<Mat> vsMat;
	vsMat.push_back(src);
	if (vsMat.size() >= num)
	{
		dst.create(Size(src.cols, src.rows), src.type());
		for (int i = 0; i < src.cols; i++) {
			uchar *pDst = dst.ptr<uchar>(i);
			for (int j = 0; j < src.rows; j++) {
				int sum = 0;
				for (int k = 0; k < num; k++) {
					sum += vsMat[k].ptr<uchar>(i)[j];
				}
				if (sum >= 255)
					pDst[j] = 255;//sum / num;
				else
					pDst[j] = 0;
			}
		}
		vsMat.clear();
		return true;
	}
	return false;
}

bool ToBin::multiFrameBlendingWithCalc(Mat &src, Mat &dst, Parabola &leftParabola, Parabola &rightParabola, int num) {
	static int count = 1;
	static float angle1 = 0;
	static float angle2 = 0;
	static float dx1 = 0;
	static float dy1 = 0;
	static float dx2 = 0;
	static float dy2 = 0;

	int xm_per_pix = 1;

	if (count == 1) {
		count++;
		dy2 = 20;
		int pos_leftx1 = leftParabola.value(src.rows - 5);
		int pos_rightx1 = rightParabola.value(src.rows - 5);

		int position1 = ((pos_leftx1 + pos_rightx1) / 2) * xm_per_pix;//当作中心

		int pos_leftx2 = leftParabola.value(src.rows - 5 - dy2 / 2);
		int pos_rightx2 = rightParabola.value(src.rows - 5 - dy2 / 2);

		int position2 = ((pos_leftx2 + pos_rightx2) / 2) * xm_per_pix;

		int pos_leftx3 = leftParabola.value(src.rows - 5 - dy2);
		int pos_rightx3 = rightParabola.value(src.rows - 5 - dy2);

		int position3 = ((pos_leftx3 + pos_rightx3) / 2) * xm_per_pix;

		dx1 = position2 - position1;
		dy1 = dy2 / 2;
		angle1 = fastAtan2(dx1, dy1);

		dx2 = position3 - position1;

		angle2 = fastAtan2(abs(dx2), abs(dy2));
		cout << "pos1=" << position1 << " pos2=" << position2 << " pos3=" << position3 << " angle1=" << angle1 << " angle2=" << angle2 << endl;
		//result=averageImagWithOffset(img, 20, dy2, 2);
	}
	else if (count == 2) {
		Mat tmat;
		src.copyTo(tmat);
		//rotationAndMove(src, tmat, -1*angle1/2, 0);
		//result=averageImagWithOffset(tmat, 20+ dx1, dy2 -dy1, 2);
		count = 1;
		imshowWithShot("combile", dst);
		//缓冲
	}
	else if (count == 3) {
		//合并显示
		Mat tmat;
		src.copyTo(tmat);
		//rotationAndMove(img, src, -1*angle2/2, 0);
		dst = averageImagWithOffset(tmat, 20 + dx2, dy2 - dy2, 3);;
		//rotationAndMove(src, tmat, angle2, 0);
		count = 1;
	}
}

void GaussianFilter::Init(int _sigma_x, int _sigma_y) {
    sigma_x_ = _sigma_x;
    sigma_y_ = _sigma_y;
    onGaussianChange();
}

void GaussianFilter::onGaussianChange() {
    int i;
    float x;
    float xs[GAUSSIANSIZE] = {0};
    float ys[GAUSSIANSIZE] = {0};
    float sumx = 0, sumy = 0;

    //fprintf(stderr, "\\sigam_{x}: %d, \\sigma_{y}: %d, Gaussian Size: %d\n", sigma_x_, sigma_y_, GAUSSIANSIZE);    
    for (i = 0; i < GAUSSIANSIZE; i++) {
        x = 1.0 * i + 0.5 - GAUSSIANSIZE * 0.5;
        xs[i] = (1.0 / sigma_x_ / sigma_x_) * exp(-1.0 * x * x / 2 / sigma_x_ / sigma_x_) * (1 - x * x / sigma_x_ / sigma_x_);     
        ys[i] = exp(-1.0 * x * x / 2 / sigma_y_ / sigma_y_);

        sumx += xs[i];
        sumy += ys[i];
    }       
    for (i = 0; i < GAUSSIANSIZE; i++) {
        xs[i] /= sumx;
        ys[i] /= sumy;
    }       


    gaussian_kernel_x_ = Mat(1, GAUSSIANSIZE, CV_32F, xs).clone();
    gaussian_kernel_y_ = Mat(1, GAUSSIANSIZE, CV_32F, ys).clone();
}

void GaussianFilter::Filter(Mat src, Mat &dst) {
    sepFilter2D(src, dst, src.depth(), gaussian_kernel_x_, gaussian_kernel_y_);
}


