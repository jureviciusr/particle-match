#include <opencv2/core.hpp>
#include <opencv2/imgproc/types_c.h>
#include "templmatch_ci.h"


double zeroIfNan(double x)
{
    if (x * 0.0 == 0.0)
        return x;
    return 0;
}

void UnifyImageFormats(cv::Mat& scene, cv::Mat& templ, cv::Mat& mask, int& type, int& depth, int& cn)
{
    type = scene.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);

    CV_Assert((depth == CV_8U || depth == CV_32F) && type == templ.type() && scene.dims <= 2);

    int ttype = templ.type(), tdepth = CV_MAT_DEPTH(ttype), tcn = CV_MAT_CN(ttype);
    int mtype = scene.type(), mdepth = CV_MAT_DEPTH(type), mcn = CV_MAT_CN(mtype);

    if (depth != CV_32F)
    {
        depth = CV_32F;
        type = CV_MAKETYPE(CV_32F, cn);
        scene.convertTo(scene, type, 1.0 / 255);
    }

    if (tdepth != CV_32F)
    {
        tdepth = CV_32F;
        ttype = CV_MAKETYPE(CV_32F, tcn);
        templ.convertTo(templ, ttype, 1.0 / 255);
    }

    if (mdepth != CV_32F)
    {
        mdepth = CV_32F;
        mtype = CV_MAKETYPE(CV_32F, mcn);
        //BUG: fix this part, convert to {0;1} independently of mask type
        cv::compare(mask, cv::Scalar::all(0), mask, cv::CMP_NE);
        mask.convertTo(mask, mtype, 1.0 / 255);
    }
}

void airvision::TemplateMatcher::cvinMatchTemplateRaw(cv::InputArray _img, cv::InputArray _templ, cv::OutputArray _result, cv::InputArray _mask, int method)
{
    CV_Assert(method == CV_TM_CCORR_NORMED || method == CV_TM_CCOEFF_NORMED);

    //prepare data structures
    int type, depth, cn;
    cv::Mat scene = _img.getMat(), templ = _templ.getMat(), mask = _mask.getMat();
    UnifyImageFormats(scene, templ, mask, type, depth, cn);

    cv::Size resultSize(scene.cols - templ.cols + 1, scene.rows - templ.rows + 1);
    _result.create(resultSize, CV_32F);
    cv::Mat result = _result.getMat();

    //start computation
    cv::MatExpr templMask;
    if (method == CV_TM_CCOEFF_NORMED) {
        auto templMaskTemp = templ.mul(mask);

        cv::Scalar avgTemplPixels(cv::sum(templMaskTemp) / (templMaskTemp.size().width * templMaskTemp.size().height));
        templMask = (templMaskTemp - avgTemplPixels).mul(mask);
    } else if (method == CV_TM_CCORR_NORMED) {
        templMask = templ.mul(mask);
    }

    auto templMaskSqr = templMask.mul(templMask);
    auto templMaskSqrSum = cv::sum(templMaskSqr);
    for (int sceneX = 0; sceneX < resultSize.width; sceneX++)
        for (int sceneY = 0; sceneY < resultSize.height; sceneY++) {
            cv::Mat sceneRoi;
            if (method == CV_TM_CCOEFF_NORMED) {
                cv::Mat sceneRoiTemp = scene(cv::Rect(sceneX, sceneY, templ.size().width, templ.size().height)).mul(mask);
                cv::Scalar avgPixels(cv::sum(sceneRoiTemp) / (sceneRoiTemp.size().width * sceneRoiTemp.size().height));
                sceneRoi = (sceneRoiTemp - avgPixels).mul(mask);
            } else if (method == CV_TM_CCORR_NORMED) {
                sceneRoi = scene(cv::Rect(sceneX, sceneY, templ.size().width, templ.size().height));
            }

            auto crossCorr = cv::sum(templMask.mul(sceneRoi));

            auto sceneMaskSqr = mask.mul(sceneRoi).mul(sceneRoi);
            auto sceneMaskSqrSum = cv::sum(sceneMaskSqr);

            cv::Scalar normalizer;
            cv::sqrt(templMaskSqrSum.mul(sceneMaskSqrSum), normalizer);

            auto normCrossCorr = crossCorr.div(normalizer);
            float mean = (zeroIfNan(normCrossCorr.val[0])
                + zeroIfNan(normCrossCorr.val[1])
                + zeroIfNan(normCrossCorr.val[2])
                + zeroIfNan(normCrossCorr.val[3])) / cn;

            result.at<float>(cv::Point(sceneX, sceneY)) = mean;
        }
}
/*
void airvision::TemplateMatcher::cvinMatchTemplateForSelectedPoints(cv::InputArray _scene, cv::InputArray _templ, cv::InputArray _mask, int method, std::vector<airvision::PointValue>& selectedPoints)
{
    cvinMatchTemplateForSelectedPoints(_scene, _templ, _mask, method, &selectedPoints[0], selectedPoints.size());
}


void airvision::TemplateMatcher::cvinMatchTemplateForSelectedPoints(cv::InputArray _scene, cv::InputArray _templ, cv::InputArray _mask, int method, airvision::PointValue* selectedPoints, int selectedPointsCount)
{
    CV_Assert(method == CV_TM_CCORR_NORMED || method == CV_TM_CCOEFF_NORMED);

    //prepare data structures
    int type, depth, cn;
    cv::Mat scene = _scene.getMat(), templ = _templ.getMat(), mask = _mask.getMat();
    UnifyImageFormats(scene, templ, mask, type, depth, cn);
    //start computation

    cv::MatExpr templMask;
    if (method == CV_TM_CCOEFF_NORMED)
    {
        auto templMaskTemp = templ.mul(mask);
        auto sum = cv::sum(templMaskTemp);
        cv::Scalar avgTemplPixels(sum / (templMaskTemp.size().width * templMaskTemp.size().height));
        templMask = (templMaskTemp - avgTemplPixels).mul(mask);
    }
    else if (method == CV_TM_CCORR_NORMED)
    {
        templMask = templ.mul(mask);
    }

    auto templMaskSqr = templMask.mul(templMask);

    auto templMaskSqrSum = cv::sum(templMaskSqr);

    for (int pointIndx = 0; pointIndx < selectedPointsCount; pointIndx++)
    {
        PointValue& selectedPoint = selectedPoints[pointIndx];

        cv::Mat sceneRoi;
        if (method == CV_TM_CCOEFF_NORMED)
        {
            cv::Mat sceneRoiTemp = scene(cv::Rect(selectedPoint.x, selectedPoint.y, templ.size().width, templ.size().height)).mul(mask);
            cv::Scalar avgPixels(cv::sum(sceneRoiTemp) / (sceneRoiTemp.size().width * sceneRoiTemp.size().height));
            sceneRoi = (sceneRoiTemp - avgPixels).mul(mask);
        }
        else if (method == CV_TM_CCORR_NORMED)
        {
            sceneRoi = scene(cv::Rect(selectedPoint.x, selectedPoint.y, templ.size().width, templ.size().height));
        }

        auto crossCorr = cv::sum(templMask.mul(sceneRoi));

        auto sceneMaskSqr = mask.mul(sceneRoi).mul(sceneRoi);
        auto sceneMaskSqrSum = cv::sum(sceneMaskSqr);

        cv::Scalar normalizer;
        cv::sqrt(templMaskSqrSum.mul(sceneMaskSqrSum), normalizer);

        auto normCrossCorr = crossCorr.div(normalizer);
        float mean = (zeroIfNan(normCrossCorr.val[0])
            + zeroIfNan(normCrossCorr.val[1])
            + zeroIfNan(normCrossCorr.val[2])
            + zeroIfNan(normCrossCorr.val[3])) / cn;

        selectedPoint.value = mean;
    }
}
*/

//--------------------------------------------------------------- EXPORTS ---------------------------------------------------------------
void cvinMatchTemplateCcorrNormedRaw(cv::_InputArray* image, cv::_InputArray* templ, cv::_OutputArray* result, cv::_InputArray* mask)
{
    airvision::TemplateMatcher::cvinMatchTemplateRaw(*image, *templ, *result, mask ? *mask : (cv::InputArray)cv::noArray(), CV_TM_CCORR_NORMED);
}

void cvinMatchTemplateCcoeffNormedRaw(cv::_InputArray* image, cv::_InputArray* templ, cv::_OutputArray* result, cv::_InputArray* mask)
{
    airvision::TemplateMatcher::cvinMatchTemplateRaw(*image, *templ, *result, mask ? *mask : (cv::InputArray)cv::noArray(), CV_TM_CCOEFF_NORMED);
}

void cvinMatchTemplateRaw(cv::_InputArray* image, cv::_InputArray* templ, cv::_OutputArray* result, cv::_InputArray* mask, int method)
{
    airvision::TemplateMatcher::cvinMatchTemplateRaw(*image, *templ, *result, mask ? *mask : (cv::InputArray)cv::noArray(), method);
}

/*
void cvinMatchTemplateForSelectedPoints(cv::_InputArray* image, cv::_InputArray* templ, cv::_InputArray* mask, int method, airvision::PointValue* selectedPoints, int selectedPointsCount)
{
    airvision::TemplateMatcher::cvinMatchTemplateForSelectedPoints(*image, *templ, *mask, method, selectedPoints, selectedPointsCount);
}
*/
//void airvision::TemplateMatcher::cvinMatchTemplateCcorrNormedRaw(cv::InputArray _img, cv::InputArray _templ, cv::OutputArray _result, cv::InputArray _mask)
//{
//	//prepare data structures
//	int type, depth, cn;
//	cv::Mat scene = _img.getMat(), templ = _templ.getMat(), mask = _mask.getMat();
//	UnifyImageFormats(scene, templ, mask, type, depth, cn);
//
//	cv::Size resultSize(scene.cols - templ.cols + 1, scene.rows - templ.rows + 1);
//	_result.create(resultSize, CV_32F);
//	cv::Mat result = _result.getMat();
//
//	//start computation
//	auto templMask = templ.mul(mask);
//	auto templMaskSqr = templMask.mul(templMask);
//
//	auto templMaskSqrSum = cv::sum(templMaskSqr);
//
//	for (int sceneX = 0; sceneX < resultSize.width; sceneX++)
//		for (int sceneY = 0; sceneY < resultSize.height; sceneY++)
//		{
//			cv::Mat sceneRoi = scene(cv::Rect(sceneX, sceneY, templ.size().width, templ.size().height));
//			auto crossCorr = cv::sum(templMask.mul(sceneRoi));
//
//			auto sceneMaskSqr = mask.mul(sceneRoi).mul(sceneRoi);
//			auto sceneMaskSqrSum = cv::sum(sceneMaskSqr);
//
//			cv::Scalar normalizer;
//			cv::sqrt(templMaskSqrSum.mul(sceneMaskSqrSum), normalizer);
//
//
//			auto normCrossCorr = crossCorr.div(normalizer);
//			float mean = (zeroIfNan(normCrossCorr.val[0])
//				+ zeroIfNan(normCrossCorr.val[1])
//				+ zeroIfNan(normCrossCorr.val[2])
//				+ zeroIfNan(normCrossCorr.val[3])) / cn;
//
//			result.at<float>(cv::Point(sceneX, sceneY)) = mean;
//		}
//}
//
//void airvision::TemplateMatcher::cvinMatchTemplateCcoeffNormedRaw(cv::InputArray _img, cv::InputArray _templ, cv::OutputArray _result, cv::InputArray _mask)
//{
//	//prepare data structures
//	int type, depth, cn;
//	cv::Mat scene = _img.getMat(), templ = _templ.getMat(), mask = _mask.getMat();
//	UnifyImageFormats(scene, templ, mask, type, depth, cn);
//
//	cv::Size resultSize(scene.cols - templ.cols + 1, scene.rows - templ.rows + 1);
//	_result.create(resultSize, CV_32F);
//	cv::Mat result = _result.getMat();
//
//	//start computation
//
//	cv::MatExpr avgTemplMask;
//	{
//		auto templMask = templ.mul(mask);
//
//		cv::Scalar avgTemplPixels(cv::sum(templMask) / (templMask.size().width * templMask.size().height));
//		avgTemplMask = (templMask - avgTemplPixels).mul(mask);
//	}
//
//	auto templMaskSqr = avgTemplMask.mul(avgTemplMask);
//
//	auto templMaskSqrSum = cv::sum(templMaskSqr);
//
//	for (int sceneX = 0; sceneX < resultSize.width; sceneX++)
//		for (int sceneY = 0; sceneY < resultSize.height; sceneY++)
//		{
//			cv::MatExpr avgSceneRoi;
//			{
//				cv::Mat sceneRoi = scene(cv::Rect(sceneX, sceneY, templ.size().width, templ.size().height)).mul(mask);
//				cv::Scalar avgPixels(cv::sum(sceneRoi) / (sceneRoi.size().width * sceneRoi.size().height));
//				avgSceneRoi = (sceneRoi - avgPixels).mul(mask);
//			}
//
//			auto crossCorr = cv::sum(avgTemplMask.mul(avgSceneRoi));
//
//			auto sceneMaskSqr = mask.mul(avgSceneRoi).mul(avgSceneRoi);
//			auto sceneMaskSqrSum = cv::sum(sceneMaskSqr);
//
//			cv::Scalar normalizer;
//			cv::sqrt(templMaskSqrSum.mul(sceneMaskSqrSum), normalizer);
//
//			auto normCrossCorr = crossCorr.div(normalizer);
//			float mean = (zeroIfNan(normCrossCorr.val[0])
//				+ zeroIfNan(normCrossCorr.val[1])
//				+ zeroIfNan(normCrossCorr.val[2])
//				+ zeroIfNan(normCrossCorr.val[3])) / cn;
//
//			result.at<float>(cv::Point(sceneX, sceneY)) = mean;
//		}
//}