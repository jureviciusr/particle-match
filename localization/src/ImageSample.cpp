//
// Created by rokas on 2020-02-29.
//

#include "ImageSample.hpp"

ImageSample::ImageSample(const cv::Mat& image, const std::vector<cv::Point>& samplePoints, float average) {
    for (const auto& p : samplePoints) {
        double val = ((float) image.at<uint8_t>(p)) - average;
        squared_sum += val * val;
        sample.push_back(val);
    }
    standart_deviation = std::sqrt(squared_sum);
};

ImageSample::ImageSample(
        const cv::Mat& image,
        const std::vector<cv::Point>& samplePoints,
        const cv::Mat& rotation,
        const cv::Point& offset,
        float average
) {
    double m11 = rotation.at<double>(0, 0),
            m12 = rotation.at<double>(0, 1),
            m13 = rotation.at<double>(0, 2),
            m21 = rotation.at<double>(1, 0),
            m22 = rotation.at<double>(1, 1),
            m23 = rotation.at<double>(1, 2);
    cv::Point centerOffset(320, 240);
    for (const auto& ps : samplePoints) {
        // Centerfix
        cv::Point p = (ps + offset) - centerOffset;

        // Transform points
        cv::Point pTran(
                m11 * p.x + m12 * p.y + m13,
                m21 * p.x + m22 * p.y + m23
        );
        double val = ((float) image.at<uint8_t>(pTran)) - average;
        squared_sum += val * val;
        sample.push_back(val);
    }
    standart_deviation = std::sqrt(squared_sum);
}

double ImageSample::calcSimilarity(const ImageSample& other) const {
    double top = 0.0;
    auto sampleLen = sample.size();
    for (int i = 0; i < sampleLen; i++) {
        top += sample[i] * other.sample[i];
    }
    double res = top / (standart_deviation * other.standart_deviation);
    if (res > 1.0 || res < -1.0) {
        std::cout << "Problem " << std::endl;
    }
    return res;
}

ImageSample::ImageSample(const cv::Mat &image, const std::vector<cv::Point> &samplePoints) {
    double sum_ = 0.0;
    for (const auto& p : samplePoints) {
        float val = ((float) image.at<uint8_t>(p));
        sum_ += val;
        sample.push_back(val);
    }
    auto average = (float) (sum_ / (double) samplePoints.size());
    for(float& val : sample) {
        val -= average;
        squared_sum += val * val;
    }
    standart_deviation = std::sqrt(squared_sum);
}

ImageSample::ImageSample(const cv::Mat &image, const std::vector<cv::Point> &samplePoints, const cv::Mat &rotation,
                         const cv::Point &offset) {
    double m11 = rotation.at<double>(0, 0),
            m12 = rotation.at<double>(0, 1),
            m13 = rotation.at<double>(0, 2),
            m21 = rotation.at<double>(1, 0),
            m22 = rotation.at<double>(1, 1),
            m23 = rotation.at<double>(1, 2);
    cv::Point centerOffset(320, 240);
    double sum_ = 0.0;
    for (const auto& ps : samplePoints) {
        // Centerfix
        cv::Point p = (ps + offset) - centerOffset;

        // Transform points
        cv::Point pTran(
                m11 * p.x + m12 * p.y + m13,
                m21 * p.x + m22 * p.y + m23
        );
        double val = ((float) image.at<uint8_t>(pTran));
        sum_ += val;
        sample.push_back(val);
    }
    auto average = (float) (sum_ / (double) samplePoints.size());
    for(float& val : sample) {
        val -= average;
        squared_sum += val * val;
    }
    standart_deviation = std::sqrt(squared_sum);
}

