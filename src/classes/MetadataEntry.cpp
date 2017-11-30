//
// Created by rokas on 17.11.30.
//

#include <opencv2/imgcodecs.hpp>
#include <opencv/cv.hpp>
#include "MetadataEntry.hpp"

cv::Mat MetadataEntry::getImage() {
    return cv::imread(imageFullPath, cv::IMREAD_GRAYSCALE);
}

cv::Mat MetadataEntry::getImageSharpened(bool smooth) {
    cv::Mat im = getImage();
    cv::equalizeHist(im, im);
    if(smooth) {
        cv::medianBlur(im, im, 5);
    }
    return im;
}
