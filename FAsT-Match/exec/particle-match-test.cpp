//
//  main.cpp
//  FAsT-Match
//
//  Created by Saburo Okita on 22/05/14.
//  Copyright (c) 2014 Saburo Okita. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv/highgui.h>
#include <chrono>

#include<boost/tokenizer.hpp>
#include <src/Utilities.hpp>

#include "../src/ParticleFastMatch.hpp"

using namespace std;
using namespace std::chrono;
using namespace cv;

std::vector<double> readCsvParams(std::string file) {
    std::vector<double> vec;
    std::ifstream infile(file);
    std::string line;
    std::getline(infile, line);
    boost::tokenizer<boost::escaped_list_separator<char> > tk(line,
                                                              boost::escaped_list_separator<char>('\\', ',', '\"'));
    for (boost::tokenizer<boost::escaped_list_separator<char> >::iterator i(tk.begin()); i != tk.end(); ++i) {
        vec.push_back(std::atof((*i).c_str()));
    }
    return vec;
}

double calcangle(cv::Point A, cv::Point B) {
    double val = (B.y - A.y) / (B.x - A.x); // calculate slope between the two points
    return val - pow(val, 3) / 3 + pow(val, 5) / 5; // find arc tan of the slope using taylor series approximation
}

void visualizeGT(const cv::Point &loc, double yaw, cv::Mat &image, int radius, int thickness,
                 const cv::Scalar &color = CV_RGB(255, 255, 0)) {
    cv::circle(image, loc, radius, color, thickness);
    cv::line(
            image,
            loc,
            cv::Point(
                    static_cast<int>(loc.x + (4 * radius * sin(yaw))),
                    static_cast<int>(loc.y - (4 * radius * cos(yaw)))
            ),
            color,
            thickness
    );
}


int main(int argc, const char *argv[]) {
    std::string templatesDir = "templates-extended";
    Mat image = imread("templates/map.png");
    std::vector<double> initParams = readCsvParams(templatesDir + "/templ_00000.txt");
    cv::Point2i loc((int) initParams[0], (int) initParams[1]);
    cv::Point2d odoLoc(initParams[0], initParams[1]);
    Particle::setDirection(initParams[2]);
    ParticleFastMatch fast_match(
            loc,    // startLocation
            image.size(), // mapSize
            350, // radius
            .1f, // epsilon
            500, // particleCount
            .99, // quantile_
            .5, // kld_error_
            10, // bin_size_
            true, // use_gaussian
            .9, // _min_scale
            1.1 // _max_scale
    );
    fast_match.setImage(image.clone());
    /*cv::namedWindow("template");
    cv::namedWindow("best_view");*/
    std::cout << "\"Iteration\",\"Particle count\",\"Distance\",\"Odometry distance\",\"LocX\",\"LocY\",\"PreX\",\"PreY\",\"OdoX\",\"OdoY\"\n";
    cv::Mat bestTransform;
    for (int i = 0; i < 280; i++) {
        std::cout << i << ",";
        std::stringstream ss;
        ss << std::setw(5) << std::setfill('0') << i;
        std::string s = ss.str();

        Mat templ = imread(templatesDir + "/templ_" + s + ".png");
        std::vector<double> params = readCsvParams(templatesDir + "/templ_" + s + ".txt");
        cv::Point2d movement(-params[3], -params[4]);
        cv::Point2i curloc((int) params[0], (int) params[1]);
        odoLoc += movement;
        Particle::setDirection(params[2]);
        Mat map = image.clone();
        fast_match.setTemplate(templ);
        cv::Mat bestView;
        std::vector<cv::Point> corners;
        if(true) {
            corners = fast_match.filterParticles(movement, bestTransform);
        } else {
            corners = fast_match.filterParticlesAffine(movement, bestTransform);
            bestView = Utilities::extractWarpedMapPart(map, templ.size(), bestTransform);
        }
        double  sumX = 0.,
                sumY = 0.;
        for(const auto& corner : corners) {
            sumX += corner.x;
            sumY += corner.y;
        }

        cv::Point2i bestParticleLocation((int) sumX / 4, (int) sumY / 4);
        cv::Point2i prediction = fast_match.getPredictedLocation();

        double distance = sqrt(pow(curloc.x - prediction.x, 2) + pow(curloc.y - prediction.y, 2));
        double odometryDistance = sqrt(pow(curloc.x - (int) odoLoc.x, 2) + pow(curloc.y - (int) odoLoc.y, 2));

        std::cout << fast_match.particleCount() << ",";
        double pixelmeters = 0.247489;
        std::cout << distance * pixelmeters << "," << odometryDistance * pixelmeters << ",";
        std::cout << curloc.x << "," << curloc.y << "," << prediction.x << "," << prediction.y << ","
                  << odoLoc.x << "," << odoLoc.y << "\n";

        /*fast_match.visualizeParticles(map);
        line( map, corners[0], corners[1], Scalar(0, 0, 255), 4);
        line( map, corners[1], corners[2], Scalar(0, 0, 255), 4);
        line( map, corners[2], corners[3], Scalar(0, 0, 255), 4);
        line( map, corners[3], corners[0], Scalar(0, 0, 255), 4);
        cv::Point2i arrowhead((corners[0].x + corners[1].x) / 2, (corners[0].y + corners[1].y) / 2);
        cv::arrowedLine(map, bestParticleLocation, arrowhead, CV_RGB(255,0,0), 20);
        visualizeGT(curloc, params[2], map, 20, 20, CV_RGB(255, 255, 0));
        visualizeGT(prediction, params[2], map, 20, 20, CV_RGB(255, 255, 255));
        cv::imshow("best_view", bestView);
        cv::imshow("Preview", map);
        cv::imshow("template", templ);
        cv::waitKey(100);*/
    }
    return 0;
}
