//
//  FAsTMatch.h
//  FAsT-Match
//
//  Created by Saburo Okita on 23/05/14.
//  Copyright (c) 2014 Saburo Okita. All rights reserved.
//

#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <iterator>
#include <boost/shared_ptr.hpp>

#include "../FAsT-Match/MatchNet.h"
#include "../FAsT-Match/MatchConfig.h"
#include "ConfigExpanderBase.hpp"
#include "ConfigVisualizer.hpp"

using namespace std;
using namespace cv;

namespace fast_match {
    class FAsTMatch{
    public:
        FAsTMatch();

        virtual void init( float epsilon = 0.15f, float delta = 0.25f, bool photometric_invariance = false,
                   float min_scale = 0.5f, float max_scale = 2.0f );

        virtual void apply(Mat &image, Mat &templ, double &best_distance,
                              float min_rotation = (float) -M_PI, float max_rotation = (float) M_PI);
        virtual void calculate();
        virtual vector<Point> getBestCorners();


        bool calculateLevel();
        int no_of_points = 0;
        int level = 0;
        cv::Mat original_image;
        Mat imageGray, templGray;
        float imageGrayAvg = 0.0;

    protected:
#ifdef USE_CV_GPU
        cv::cuda::GpuMat imageGrayGpu, templGrayGpu;
#endif
        Mat image, templ;
        float templAvg = 0.0;
        float imageAvg = 0.0;
        float templGrayAvg = 0.0;

    public:
        virtual void setImage(const Mat &image);

        virtual void setTemplate(const Mat &templ);

        static vector<double> evaluateConfigs( Mat& image, Mat& templ, vector<Mat>& affine_matrices,
                                        Mat& xs, Mat& ys, bool photometric_invariance );

    protected:

        RNG rng;
        float epsilon;
        float delta;
        bool photometricInvariance;
        float minScale;
        float maxScale;
        Size halfTempl;
        Size halfImage;

        std::shared_ptr<ConfigExpanderBase> configExpander;
        ConfigVisualizer visualizer;
        bool visualize = true;



        /*vector<MatchConfig> createListOfConfigs( MatchNet& net, Size templ_size, Size image_size );*/
        vector<Mat> configsToAffine( vector<MatchConfig>& configs, vector<bool>& insiders );

        vector<MatchConfig> getGoodConfigsByDistance( vector<MatchConfig>& configs, float best_dist, float new_delta,
                                                      vector<double>& distances, float& thresh, bool& too_high_percentage );

        /*vector<MatchConfig> randomExpandConfigs( vector<MatchConfig>& configs, MatchNet& net,
                                                 int level, int no_of_points, float delta_factor );*/


        float delta_fact = 1.511f;
        float new_delta;


        MatchConfig best_config;
        Mat best_trans;
        vector<double> best_distances;
        vector<double> distances;
        vector<bool> insiders;
        vector<MatchConfig> configs;

    };
}

template<typename type>
ostream &operator <<( ostream& os, const std::pair<type, type> & vec ) {
    os << "[";
    os << vec.first << " " << vec.second;
    os << "]";
    return os;
}

template<typename type>
ostream &operator <<( ostream& os, const vector<type> & vec ) {
    os << "[";
    std::copy( vec.begin(), vec.end(), ostream_iterator<type>(os, ", ") );
    os << "]";
    return os;
}
