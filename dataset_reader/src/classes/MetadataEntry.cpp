//
// Created by rokas on 17.11.30.
//

#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include "fastmatch-dataset/MetadataEntry.hpp"

cv::Mat MetadataEntry::getImage() const {
    return cv::imread(imageFullPath, cv::IMREAD_GRAYSCALE);
}

cv::Mat MetadataEntry::getImageColored() const {
    return imageBuffer.clone();
}

cv::Mat MetadataEntry::getImageSharpened(bool smooth) const {
    cv::Mat im = getImage();
    cv::equalizeHist(im, im);
    if(smooth) {
        cv::medianBlur(im, im, 5);
    }
    return im;
}
