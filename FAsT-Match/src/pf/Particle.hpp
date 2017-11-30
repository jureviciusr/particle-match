//
// Created by rokas on 17.6.20.
//

#pragma once


#include <FAsT-Match/MatchConfig.h>

class Particle {
public:
    int x, y;
    float getProbability() const;

    void setProbability(float probability);

    void setMinimalProbability(float probability);

    void setMaximalProbability(float probability);

    Particle(int x, int y);

    Particle(const Particle& a);

    //Particle();

    const std::vector<fast_match::MatchConfig> & getConfigs(int id);

    void propagate(const cv::Point2f& movement);

    void updateConfigs();

    static void setMapDimensions(const cv::Size& dims);

protected:

    std::vector<fast_match::MatchConfig> configs;
    static cv::Point2i mapCenter;
    float probability;
    float samplingFactor;
    float accumulatedProbability = 0.f;
    std::vector<float> oldProbabilities = {};
    uint32_t iteration = 0;
    float weight;
public:
    float getWeight() const;

    void setWeight(float weight);

    float getSamplingFactor() const;

    void setSamplingFactor(float samplingFactor);

protected:
    std::vector<cv::Mat> getAffines(const cv::Size& imageSize, const cv::Size& templSize);
    cv::Mat bestTransform;
public:
    const cv::Mat &getBestTransform() const;

private:
    static std::vector<float> r_initial;
    static std::vector<float> s_initial;
    static float r_step;
    static double direction;

public:
    static void setDirection(double direction);
    double evaluate(cv::Mat& image, cv::Mat& templ, cv::Mat& xs, cv::Mat& ys);

    bool operator<(const Particle& str) const;
    bool operator>(const Particle& str) const;
    std::string serialize(int binSize);

};


