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
	//IPM�任
	dst = dstRoi(detectRoi);
}

//֧�ֲ�ɫ���ߺڰ�
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

void Transform::dispDetectRoi(Mat &dst) {//������ԭʼͼ��
	Mat colorEdge(Size(detectRoi.width, detectRoi.height), dst.type());
	colorEdge = Scalar::all(0);
	rectangle(colorEdge, Point(0, 0), Point(colorEdge.cols - 1, colorEdge.rows - 1),
		Scalar(0, 0, 255), 1, 8, 0);
	mapToRealImg(colorEdge, dst);
}