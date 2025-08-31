//
// Created by rokas on 17.12.3.
//

#include <opencv2/imgcodecs.hpp>
#include <fstream>
#include <regex>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "fastmatch-dataset/Map.hpp"

Map::Map(const std::string &mapFile, const std::string &mapDescription) {
    image = cv::imread(mapFile);
    dimensions = image.size();
    static const std::regex e(R"(\s*(\w+)\:\s*(\-?\d+(\.\d+)?)\s*)");
    std::match_results<std::string::const_iterator> results;
    std::ifstream description(mapDescription);
    std::string str;
    double coords[4];
    while(std::getline(description, str)) {
        if (std::regex_match(str, results, e)) {
            if(results[1] == "North_Bounding_Coordinate") {
                std::string result = results[2];
                coords[0] = std::atof(result.c_str());
            }
            if(results[1] == "South_Bounding_Coordinate") {
                std::string result = results[2];
                coords[1] = std::atof(result.c_str());
            }
            if(results[1] == "West_Bounding_Coordinate") {
                std::string result = results[2];
                coords[2] = std::atof(result.c_str());
            }
            if(results[1] == "East_Bounding_Coordinate") {
                std::string result = results[2];
                coords[3] = std::atof(result.c_str());
            }
        }
    }
    if(coords[0] != 0. && coords[1] != 0. && coords[2] != 0. && coords[3] != 0.) {
        geoRegion[0].Reset(coords[0], coords[2]);
        geoRegion[1].Reset(coords[1], coords[3]);
        valid = true;
    } else {
        valid = false;
    }
}

bool Map::isValid() const {
    return valid;
}

const cv::Mat &Map::getImage() const {
    return image;
}

cv::Point2i Map::toPixels(double latitude, double longitude) const {
    return cv::Point2i(
            static_cast<int>(
                    ((geoRegion[0].Longitude() - longitude) / (geoRegion[0].Longitude() - geoRegion[1].Longitude()))
                    * (double) image.cols
            ),
            static_cast<int>(
                    ((geoRegion[0].Latitude() - latitude) / (geoRegion[0].Latitude() - geoRegion[1].Latitude()))
                    * (double) image.rows
            )
    );
}

void Map::toCoords(const cv::Point2i &loc, double &latitude, double &longitude) {
    longitude = geoRegion[0].Longitude() -
            ((geoRegion[0].Longitude() - geoRegion[1].Longitude()) * ((double) loc.x / (double) dimensions.width));
    latitude = geoRegion[0].Latitude() -
            ((geoRegion[0].Latitude() - geoRegion[1].Latitude()) * ((double) loc.y / (double) dimensions.height));
}

cv::Mat Map::subregion(const cv::Size &size, double latitude, double longitude, double heading, double scale) {
    float imangle = std::atan(((float) size.width) / ((float) size.height));
    auto roiReserver = (int) (std::ceil(((float) size.width / scale) / std::sin(imangle)) * 1.1);
    int centerLoc = roiReserver / 2;
    // Cut part of map
    cv::Mat output(size, image.type());
    cv::Point position = toPixels(latitude, longitude);
    cv::Mat region = image(cv::Rect(position.x - centerLoc, position.y - centerLoc, roiReserver, roiReserver));
    cv::Rect rotationRoi(
            static_cast<int>(((float) centerLoc * scale) - (size.width / 2)),
            static_cast<int>(((float) centerLoc * scale) - (size.height / 2)),
            size.width,
            size.height
    );
    cv::Size reservedSize(
            static_cast<int>(roiReserver * scale),
            static_cast<int>(roiReserver * scale)
    );
    cv::Mat view(reservedSize, region.type());
    auto center = cv::Point2f(
            (float) region.cols / 2.f,
            (float) region.rows / 2.f
    );
    cv::Mat rot_mat = cv::getRotationMatrix2D(
            center,
            heading,
            scale
    );
    rot_mat.at<double>(0, 2) += std::abs(1 - (std::cos((imangle * 180.0) / M_PI))) * center.x;
    rot_mat.at<double>(1, 2) += std::abs(1 - (std::cos((imangle * 180.0) / M_PI))) * center.y;
    cv::warpAffine(region, view, rot_mat, cv::Size(reservedSize));
    return view(rotationRoi);
}

bool Map::isWithinMap(double latitude, double longitude) {
    double minlat = std::min(geoRegion[0].Latitude(), geoRegion[1].Latitude());
    double maxlat = std::max(geoRegion[0].Latitude(), geoRegion[1].Latitude());
    double minlon = std::min(geoRegion[0].Longitude(), geoRegion[1].Longitude());
    double maxlon = std::max(geoRegion[0].Longitude(), geoRegion[1].Longitude());
    return latitude >= minlat && latitude <= maxlat && longitude >= minlon && longitude <= maxlon;
}

Map::Map() = default;
