//
// Created by rokas on 17.12.3.
//

#pragma once

#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>
#include <GeographicLib/GeoCoords.hpp>

class Map {
protected:
    // Georegion is described by 2 points
    // [0] - topLeft geo-coordinates
    // [1] - bottomRight geo-coordinates
    std::vector<GeographicLib::GeoCoords> geoRegion = std::vector<GeographicLib::GeoCoords>(2);
    cv::Mat image;
    cv::Size dimensions;
    double scale = 0.;
    bool valid = false;
public:
    bool isValid() const;

    Map(const std::string &mapFile, const std::string &mapDescription);

};

typedef std::shared_ptr<Map> MapPtr;


