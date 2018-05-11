//
// Created by rokas on 18.5.1.
//

#pragma once

#include <boost/filesystem.hpp>

#include <boost/property_tree/ptree.hpp>
#include <opencv2/core/types.hpp>

namespace fs = boost::filesystem;
namespace pt = boost::property_tree;

class PascalVocWriter {
private:
    std::string outfilename;

public:
    pt::ptree tree;

    PascalVocWriter(const fs::path& imagePath, const cv::Size& imsize);

    void addPolygon(const std::vector<cv::Point>& poly, const std::string& name);

    void write() const;

};

