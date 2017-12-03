//
// Created by rokas on 17.11.30.
//

#pragma once

#include <string>
#include <opencv2/core/mat.hpp>
#include "Quaternion.hpp"
#include "Vector3d.hpp"

class MetadataEntry {
public:
    std::string imageFileName;
    std::string imageFullPath;
    double latitude;
    double longitude;
    double altitude;
    Quaternion imuOrientation;

    Vector3d groundTruthPose;
    Quaternion groundTruthOrientation;

    cv::Point2i mapLocation;

    Vector3d svoPose;

    cv::Mat getImage();
    cv::Mat getImageSharpened(bool smooth = false);
};


