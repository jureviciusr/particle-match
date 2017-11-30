#pragma once

#include <opencv2/core/mat.hpp>

namespace airvision {

	class TemplateMatcher {
	public:
		static void cvinMatchTemplateRaw(
                cv::InputArray _scene,
                cv::InputArray _templ,
                cv::OutputArray _result,
                cv::InputArray _mask, int method
        );

		/*static void cvinMatchTemplateForSelectedPoints(
                cv::InputArray _scene,
                cv::InputArray _templ,
                cv::InputArray _mask,
                int method,
                airvision::PointValue* selectedPoints,
                int selectedPointsCount
        );

		static void cvinMatchTemplateForSelectedPoints(
                cv::InputArray _scene,
                cv::InputArray _templ,
                cv::InputArray _mask,
                int method,
                std::vector<airvision::PointValue>& selectedPoints
        );*/

	};
}


void  cvinMatchTemplateCcorrNormedRaw(
        cv::_InputArray* image,
        cv::_InputArray* templ,
        cv::_OutputArray* result,
        cv::_InputArray* mask
);

void cvinMatchTemplateCcoeffNormedRaw(
        cv::_InputArray* image,
        cv::_InputArray* templ,
        cv::_OutputArray* result,
        cv::_InputArray* mask
);

void cvinMatchTemplateRaw(
        cv::_InputArray* image,
        cv::_InputArray* templ,
        cv::_OutputArray* result,
        cv::_InputArray* mask,
        int method
);

/*void cvinMatchTemplateForSelectedPoints(
        cv::_InputArray* image,
        cv::_InputArray* templ,
        cv::_InputArray* mask,
        int method,
        airvision::PointValue* selectedPoints,
        int selectedPointsCount
);*/