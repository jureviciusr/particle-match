//
// Created by rokas on 17.12.3.
//

#include <opencv2/imgcodecs.hpp>
#include <fstream>
#include <regex>
#include "fastmatch-dataset/Map.hpp"

Map::Map(const std::string &mapFile, const std::string &mapDescription) {
    image = cv::imread(mapFile);
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
