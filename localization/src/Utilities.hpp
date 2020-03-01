//
// Created by rokas on 17.6.20.
//

#pragma once

#include <opencv/cv.h>
#include "../FAsT-Match/MatchConfig.h"

class Utilities {
public:
    static cv::Mat preprocessImage(const cv::Mat &image);

    static cv::Mat makeOdd(cv::Mat &image);

    static std::vector<cv::Point> calcCorners(cv::Size image_size, cv::Size templ_size, cv::Mat &affine);

    static cv::Point calculateLocationInMap(
            const cv::Size &image_size,
            const cv::Size &templ_size,
            const cv::Mat &affine,
            const cv::Point &templPoint
    );

    static float getThresholdPerDelta(float delta);

    static double normal_dist();

    static double gausian_noise(double u);

    static double uniform_dist();

    static std::vector<cv::Mat> configsToAffine(
            std::vector<fast_match::MatchConfig> &configs, std::vector<bool> &insiders,
            const cv::Size &imageSize, const cv::Size &templSize);

    static cv::Mat getMapRoiMask(const cv::Size &image_size, const cv::Size &templ_size, cv::Mat &affine);

    static cv::Mat extractWarpedMapPart(cv::InputArray map, const cv::Size &templ_size, const cv::Mat &affine);

    static cv::Mat extractMapPart(const cv::Mat &map, const cv::Size &size, const cv::Point &position,
                                  double angle, float scale);

    static double calculateCorrelation(cv::Mat i1, cv::Mat i2);

    static float calculateCorrCoeff(cv::Mat scene, cv::Mat templ);

    static cv::Mat photometricNormalization(cv::Mat scene, cv::Mat templ);
#ifdef USE_CV_GPU
    static cv::cuda::GpuMat
    extractWarpedMapPart(cv::cuda::GpuMat map, const cv::Size &templ_size, const cv::Mat &affine);

    static cv::cuda::GpuMat
    extractMapPart(const cv::cuda::GpuMat &map, const cv::Size &size, const cv::Point &position, double angle,
                   float scale);

    static float calculateCorrCoeff(cv::cuda::GpuMat scene, cv::cuda::GpuMat templ);
#endif
    static cv::Mat eulerAnglesToRotationMatrix(const cv::Point3d &angles);

    static cv::Point3d
    intersectPlaneV3(const cv::Point3d &a, const cv::Point3d &b, const cv::Point3d &p_co, const cv::Point3d &p_no,
                     float epsilon = 1e-6);

    static std::string matType(int type);
};


