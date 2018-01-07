//
// Created by rokas on 18.1.2.
//

#pragma once

#include "Map.hpp"


class GeotiffMap : public Map {
protected:
    double adfGeoTransform[6];
    int zoneNumber = 0;
    bool northp = true;

public:
    void open(const std::string& filename);

    cv::Point2i toPixels(double latitude, double longitude) const override;

    void toCoords(const cv::Point2i &loc, double &latitude, double &longitude) override;

    GeographicLib::GeoCoords pixelCoordinates(const cv::Point2i &loc);


};

typedef std::shared_ptr<GeotiffMap> GeoMapPtr;


