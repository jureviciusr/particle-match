//
// Created by rokas on 18.1.2.
//
#include <gdal/gdal.h>
#include <gdal/gdal_priv.h>
#include <gdal/ogr_spatialref.h>
#include <iomanip>
#include <iostream>
#include <regex>
#include <opencv2/core.hpp>
#include <opencv/cv.hpp>

#include "fastmatch-dataset/GeotiffMap.hpp"

void GeotiffMap::open(const std::string &filename) {
    GDALDataset *poDataset;
    GDALAllRegister();
    poDataset = (GDALDataset *) GDALOpen( filename.c_str(), GA_ReadOnly );
    if(poDataset != nullptr) {
        const char* projectionString = poDataset->GetProjectionRef();
        auto srs = OGRSpatialReference(projectionString);
        auto zone = std::string(srs.GetAttrValue("projcs"));
        static const std::regex zoneRegex(R"(.*UTM\szone\s(\d+)(\w))");
        std::match_results<std::string::const_iterator> res;
        if(std::regex_match(zone, res, zoneRegex)) {
            zoneNumber = std::atoi(std::string(res[1]).c_str());
            northp = (res[2] == "N");
            if(poDataset->GetGeoTransform(adfGeoTransform) == CE_None) {
                image = cv::imread(filename);
                geoRegion[0] = pixelCoordinates(cv::Point(0, 0));
                geoRegion[1] = pixelCoordinates(cv::Point(image.cols, image.rows));
                valid = true;
            }
        }
    }
}

cv::Point2i GeotiffMap::toPixels(double latitude, double longitude) const {
    return Map::toPixels(latitude, longitude);
}

void GeotiffMap::toCoords(const cv::Point2i &loc, double &latitude, double &longitude) {
    auto coords = pixelCoordinates(loc);
    latitude = coords.Latitude();
    longitude = coords.Longitude();
}

GeographicLib::GeoCoords GeotiffMap::pixelCoordinates(const cv::Point2i &loc) {
    return {
            zoneNumber,
            northp,
            adfGeoTransform[0] + loc.x * adfGeoTransform[1] + loc.y * adfGeoTransform[2],
            adfGeoTransform[3] + loc.x * adfGeoTransform[4] + loc.y * adfGeoTransform[5]
    };
}
