void Transform::Init(visualConfig &cfg) {
	perspective.Init(cfg);
	ipmRoi = Rect(cfg.roiStartX, cfg.roiStartY, cfg.roiWidth, cfg.roiHeight);
	detectRoi = Rect(cfg.decRoiStartX, cfg.decRoiStartY, cfg.decRoiWidth, cfg.decRoiHeight);
}

void Transform::getDetectRoi(Mat &src, Mat &dst) {
	Mat srcRoi;
	Mat dstRoi;
	srcRoi = src(ipmRoi);
	//imshowWithShot("srcRoi", srcRoi);
	perspective.IPM(srcRoi, dstRoi);
	//imshowWithShot("dstRoi", dstRoi);
	//IPM变换
	dst = dstRoi(detectRoi);
}

//支持彩色或者黑白
void Transform::mapToRealImg(cv::Mat &srcRoi, cv::Mat &dst) {

	Mat panelImg(Size(ipmRoi.width, ipmRoi.height), srcRoi.type());
	panelImg = Scalar::all(0);
	Mat detectImg = panelImg(detectRoi);

	srcRoi.copyTo(detectImg);
	perspective.IPMInv(panelImg, panelImg);

	Mat ipmImg = dst(ipmRoi);
	Mat temp;

	double alpha = 0.8;
	cv::addWeighted(panelImg, alpha, ipmImg, 1, 0, ipmImg);
}

void Transform::dispDetectRoi(Mat &dst) {//输入是原始图像
	Mat colorEdge(Size(detectRoi.width, detectRoi.height), dst.type());
	colorEdge = Scalar::all(0);
	rectangle(colorEdge, Point(0, 0), Point(colorEdge.cols - 1, colorEdge.rows - 1),
		Scalar(0, 0, 255), 1, 8, 0);
	mapToRealImg(colorEdge, dst);
}