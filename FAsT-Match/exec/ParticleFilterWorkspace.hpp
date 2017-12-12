//
// Created by rokas on 17.12.4.
//

#pragma once

#include <vector>
#include <src/ParticleFastMatch.hpp>
#include <fastmatch-dataset/MetadataEntry.hpp>
#include <GeographicLib/LocalCartesian.hpp>

class ParticleFilterWorkspace {
protected:
    std::shared_ptr<ParticleFastMatch> pfm;
    cv::Point initialPosition;
    double direction;
    cv::Mat map;
    std::vector<cv::Point> corners;
    cv::Mat bestTransform;

    std::shared_ptr<GeographicLib::LocalCartesian> svoCoordinates;

    static void visualizeGT(const cv::Point &loc, double yaw, cv::Mat &image, int radius, int thickness,
                            const cv::Scalar &color = CV_RGB(255, 255, 0));

    void updateScale(float hfov, float altitude, uint32_t imageWidth);

    cv::Point getMovementFromSvo(const MetadataEntry& metadata);

public:
    void initialize(const MetadataEntry& metadata);

    void update(const MetadataEntry& metadata);

    void preview(const MetadataEntry& metadata) const;
};


