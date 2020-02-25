//
// Created by rokas on 17.5.8.
//

#pragma once


#include <FAsT-Match/FAsTMatch.h>
#include <mutex>

class FastMatcherThread {
protected:
    fast_match::FAsTMatch matcher;
    double directionPrecision = M_PI_4;
    std::mutex mtex;
    std::unique_lock<std::mutex> lock;
    cv::Point2f processingResult;
    bool resultAvailable = false;
    bool debug = true;
    double scaleDownFactor = .5;

public:
    void setDirectionPrecision(double directionPrecision);
    FastMatcherThread();
    cv::Point2f match(cv::Mat image, cv::Mat templ, double direction);
    void matchAsync(cv::Mat image, cv::Mat templ, double direction);
    bool isRunning();
    bool getResultIfAvailable(cv::Point2f& result);
};


