//
// Created by rokas on 2020-02-29.
//

#pragma once

#include "Utilities.hpp"
#include <opencv2/opencv.hpp>

class ImageSample {
public:
    std::vector<float> sample;
    double squared_sum = 0.0;
    double standart_deviation = 0.0;

    ImageSample(const cv::Mat& image, const std::vector<cv::Point>& samplePoints, float average);

    ImageSample(const cv::Mat& image, const std::vector<cv::Point>& samplePoints);

    ImageSample(
            const cv::Mat& image,
            const std::vector<cv::Point>& samplePoints,
            const cv::Mat& rotation,
            const cv::Point& offset,
            float average
    );

    ImageSample(
            const cv::Mat& image,
            const std::vector<cv::Point>& samplePoints,
            const cv::Mat& rotation,
            const cv::Point& offset
    );

    ImageSample() = default;

    double calcSimilarity(const ImageSample& other) const;
};

