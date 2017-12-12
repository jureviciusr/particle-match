//
// Created by rokas on 17.11.30.
//

#pragma once

#include <string>
#include <opencv2/core/mat.hpp>
#include "Quaternion.hpp"
#include "Vector3d.hpp"
#include "Map.hpp"

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

    cv::Mat map;
    cv::Mat imageBuffer;

    cv::Mat getImage() const;
    cv::Mat getImageSharpened(bool smooth = false) const;

    cv::Mat getImageColored() const;

    std::shared_ptr<Map> mapper;
};


