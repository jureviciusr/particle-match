//
// Created by rokas on 17.6.20.
//

#pragma once

#include <opencv/cv.h>
#include "../FAsT-Match/MatchConfig.h"

class Utilities {
public:
    static cv::Mat preprocessImage(const cv::Mat& image);
    static cv::Mat makeOdd(cv::Mat& image);
    static std::vector<cv::Point> calcCorners( cv::Size image_size, cv::Size templ_size, cv::Mat& affine );
    static cv::Point calculateLocationInMap(
            const cv::Size& image_size,
            const cv::Size& templ_size,
            const cv::Mat& affine,
            const cv::Point& templPoint
    );
    static float getThresholdPerDelta(float delta);
    static double normal_dist();
    static double gausian_noise(double u);
    static double uniform_dist();

    static std::vector<cv::Mat> configsToAffine(
            std::vector<fast_match::MatchConfig> &configs, std::vector<bool> &insiders,
            const cv::Size& imageSize, const cv::Size& templSize);

    static cv::Mat getMapRoiMask(const cv::Size &image_size, const cv::Size &templ_size, cv::Mat &affine);

    static cv::Mat extractWarpedMapPart(const cv::_InputArray &map, const cv::Size &templ_size, const cv::Mat &affine);

    static cv::Mat extractMapPart(const cv::Mat &map, const cv::Size &size, const cv::Point &position,
                                  double angle, float scale);

    static double calculateCorrelation(cv::Mat i1, cv::Mat i2);

    static float calculateCorrCoeff(cv::Mat scene, cv::Mat templ);

    static cv::Mat photometricNormalization(cv::Mat scene, cv::Mat templ);
};


