//
//  FAsTMatch.cpp
//  FAsT-Match
//
//  Created by Saburo Okita on 23/05/14.
//  Copyright (c) 2014 Saburo Okita. All rights reserved.
//

#include "FastMatch.hpp"
#include "GridConfigExpander.hpp"
#include "Utilities.hpp"
#include <iomanip>
#include <random>
#include <tbb/tbb.h>


#define WITHIN(val, top_left, bottom_right) (\
            val.x > top_left.x && val.y > top_left.y && \
            val.x < bottom_right.x && val.y < bottom_right.y )

namespace fast_match {
    FAsTMatch::FAsTMatch() {
        init();
    }

    void FAsTMatch::init(float epsilon, float delta, bool photometric_invariance, float min_scale, float max_scale) {
        this->epsilon = epsilon;
        this->delta = delta;
        this->photometricInvariance = photometric_invariance;
        this->minScale = min_scale;
        this->maxScale = max_scale;
        GridConfigExpander* expander = new GridConfigExpander();
        configExpander.reset(dynamic_cast<ConfigExpanderBase*>(expander));
    }

    void FAsTMatch::apply(Mat &original_image, Mat &original_template, double &best_distance, float min_rotation,
                                         float max_rotation) {
        /* Preprocess the image and template first */
        image = Utilities::preprocessImage(original_image);
        templ = Utilities::preprocessImage(original_template);
        FAsTMatch::original_image = original_image;

        int r1x = (int) (0.5 * (templ.cols - 1)),
                r1y = (int) (0.5 * (templ.rows - 1)),
                r2x = (int) (0.5 * (image.cols - 1)),
                r2y = (int) (0.5 * (image.rows - 1));

        float min_trans_x = -(r2x - r1x * minScale),
                max_trans_x = -min_trans_x,
                min_trans_y = -(r2y - r1y * minScale),
                max_trans_y = -min_trans_y;

        /* Create the matching grid / net */
        MatchNet net(templ.cols, templ.rows, delta, min_trans_x, max_trans_x, min_trans_y, max_trans_y,
                     min_rotation, max_rotation, minScale, maxScale);
        configExpander->setNet(net);


        /* Smooth our images */
        GaussianBlur(templ, templ, Size(0, 0), 2.0, 2.0);
        GaussianBlur(image, image, Size(0, 0), 2.0, 2.0);

        no_of_points = (int) round(10 / (epsilon * epsilon));


        level = 0;

        distances = vector<double>(20, 0.0);
        best_distances = vector<double>(20, 0.0);
        new_delta = delta;
    }


    /**
     * From given list of configurations, convert them into affine matrices.
     * But filter out all the rectangles that are out of the given boundaries.
     **/
    vector<Mat> FAsTMatch::configsToAffine(vector<MatchConfig> &configs, vector<bool> &insiders) {
        int no_of_configs = static_cast<int>(configs.size());
        vector<Mat> affines((unsigned long) no_of_configs);

        /* The boundary, between -10 to image size + 10 */
        Point2d top_left(-10., -10.);
        Point2d bottom_right(image.cols + 10, image.rows + 10);


        /* These are for the calculations of affine transformed corners */
        int r1x = 0.5 * (templ.cols - 1),
                r1y = 0.5 * (templ.rows - 1),
                r2x = 0.5 * (image.cols - 1),
                r2y = 0.5 * (image.rows - 1);

        Mat corners = (Mat_<float>(3, 4) << 1 - (r1x + 1), templ.cols - (r1x + 1), templ.cols - (r1x + 1), 1 - (r1x + 1),
                1 - (r1y + 1), 1 - (r1y + 1), templ.rows - (r1y + 1), templ.rows - (r1y + 1),
                1.0, 1.0, 1.0, 1.0);

        Mat transl = (Mat_<float>(4, 2) << r2x + 1, r2y + 1,
                r2x + 1, r2y + 1,
                r2x + 1, r2y + 1,
                r2x + 1, r2y + 1);

        insiders.assign((unsigned long) no_of_configs, false);

        /* Convert each configuration to corresponding affine transformation matrix */
        tbb::parallel_for(0, no_of_configs, 1, [&](int i) {
            Mat affine = configs[i].getAffineMatrix();

            /* Check if our affine transformed rectangle still fits within our boundary */
            Mat affine_corners = (affine * corners).t();
            affine_corners = affine_corners + transl;

            if (WITHIN(affine_corners.at<Point2f>(0, 0), top_left, bottom_right) &&
                WITHIN(affine_corners.at<Point2f>(1, 0), top_left, bottom_right) &&
                WITHIN(affine_corners.at<Point2f>(2, 0), top_left, bottom_right) &&
                WITHIN(affine_corners.at<Point2f>(3, 0), top_left, bottom_right)) {

                affines[i] = affine;
                insiders[i] = true;
            }
        });

        /* Filter out empty affine matrices (which initially don't fit within the preset boundary) */
        /* It's done this way, so that I could parallelize the loop */
        vector<Mat> result;
        for (int i = 0; i < no_of_configs; i++) {
            if (insiders[i])
                result.push_back(affines[i]);
        }

        return result;
    }


    /**
     * Evaluate the score of the given configurations
     */
    vector<double> FAsTMatch::evaluateConfigs(Mat &image, Mat &templ, vector<Mat> &affine_matrices,
                                              Mat &xs, Mat &ys, bool photometric_invariance) {

        int r1x = (int) (0.5 * (templ.cols - 1)),
                r1y = (int) (0.5 * (templ.rows - 1)),
                r2x = (int) (0.5 * (image.cols - 1)),
                r2y = (int) (0.5 * (image.rows - 1));

        int no_of_configs = static_cast<int>(affine_matrices.size());
        int no_of_points = xs.cols;

        /* Use a padded image, to avoid boundary checking */
        Mat padded(image.rows * 3, image.cols, image.type(), Scalar(0.0));
        image.copyTo(Mat(padded, Rect(0, image.rows, image.cols, image.rows)));

        /* Create a lookup array for our template values based on the given random x and y points */
        int *xs_ptr = xs.ptr<int>(0),
                *ys_ptr = ys.ptr<int>(0);

        vector<float> vals_i1(no_of_points);
        for (int i = 0; i < no_of_points; i++)
            vals_i1[i] = templ.at<float>(ys_ptr[i] - 1, xs_ptr[i] - 1);


        /* Recenter our indices */
        Mat xs_centered = xs.clone() - (r1x + 1),
                ys_centered = ys.clone() - (r1y + 1);

        int *xs_ptr_cent = xs_centered.ptr<int>(0),
                *ys_ptr_cent = ys_centered.ptr<int>(0);

        vector<double> distances(no_of_configs, 0.0);

        /* Calculate the score for each configurations on each of our randomly sampled points */
        tbb::parallel_for(0, no_of_configs, 1, [&](int i) {

            float a11 = affine_matrices[i].at<float>(0, 0),
                    a12 = affine_matrices[i].at<float>(0, 1),
                    a13 = affine_matrices[i].at<float>(0, 2),
                    a21 = affine_matrices[i].at<float>(1, 0),
                    a22 = affine_matrices[i].at<float>(1, 1),
                    a23 = affine_matrices[i].at<float>(1, 2);

            double tmp_1 = (r2x + 1) + a13 + 0.5;
            double tmp_2 = (r2y + 1) + a23 + 0.5 + 1 * image.rows;
            double score = 0.0;

            if (!photometric_invariance) {
                for (int j = 0; j < no_of_points; j++) {
                    int target_x = int(a11 * xs_ptr_cent[j] + a12 * ys_ptr_cent[j] + tmp_1),
                            target_y = int(a21 * xs_ptr_cent[j] + a22 * ys_ptr_cent[j] + tmp_2);

                    //score += abs(vals_i1[j] - padded.at<float>(target_y - 1, target_x - 1) );
                    if (target_x - 1 >= 0 && target_x - 1 < padded.size().width)
                        score += abs(vals_i1[j] - padded.at<float>(target_y - 1, target_x - 1));

                }
            } else {
                vector<double> xs_target(no_of_points),
                        ys_target(no_of_points);

                double sum_x = 0.0,
                        sum_y = 0.0,
                        sum_x_squared = 0.0,
                        sum_y_squared = 0.0;

                for (int j = 0; j < no_of_points; j++) {
                    int target_x = int(a11 * xs_ptr_cent[j] + a12 * ys_ptr_cent[j] + tmp_1),
                            target_y = int(a21 * xs_ptr_cent[j] + a22 * ys_ptr_cent[j] + tmp_2);

                    float xi = vals_i1[j],
                            yi = padded.at<float>(target_y - 1, target_x - 1);

                    xs_target[j] = xi;
                    ys_target[j] = yi;

                    sum_x += xi;
                    sum_y += yi;

                    sum_x_squared += (xi * xi);
                    sum_y_squared += (yi * yi);
                }

                double epsilon = 1e-7;
                double mean_x = sum_x / no_of_points,
                        mean_y = sum_y / no_of_points,
                        sigma_x = sqrt((sum_x_squared - (sum_x * sum_x) / no_of_points) / no_of_points) + epsilon,
                        sigma_y = sqrt((sum_y_squared - (sum_y * sum_y) / no_of_points) / no_of_points) + epsilon;

                double sigma_div = sigma_x / sigma_y;
                double temp = -mean_x + sigma_div * mean_y;


                for (int j = 0; j < no_of_points; j++)
                    score += fabs(xs_target[j] - sigma_div * ys_target[j] + temp);
            }

            distances[i] = score / no_of_points;
            //configs[i].setProbability((float) distances[i]);
        });


        return distances;
    }


    /**
     * Given the previously calcuated distances for each configurations,
     * filter out all distances that fall within a certain threshold
     */
    vector<MatchConfig>
    FAsTMatch::getGoodConfigsByDistance(vector<MatchConfig> &configs, float best_dist, float new_delta,
                                        vector<double> &distances, float &thresh, bool &too_high_percentage) {
        thresh = best_dist + Utilities::getThresholdPerDelta(new_delta);

        /* Only those configs that have distances below the given threshold are */
        /* categorized as good configurations */
        vector<MatchConfig> good_configs;
        for (int i = 0; i < distances.size(); i++) {
            if (distances[i] <= thresh)
                good_configs.push_back(configs[i]);
        }

        int no_of_configs = static_cast<int>(good_configs.size());

        /* Well if there's still too many configurations */
        /* keep shrinking the threshold */
        while (no_of_configs > 27000) {
            thresh *= 0.99;
            good_configs.clear();

            for (int i = 0; i < distances.size(); i++) {
                if (distances[i] <= thresh)
                    good_configs.push_back(configs[i]);
            }

            no_of_configs = static_cast<int>(good_configs.size());
        }

        assert(no_of_configs > 0);

        float percentage = 1.0 * no_of_configs / configs.size();

        /* If it's above 97.8% it's too high percentage */
        too_high_percentage = percentage > 0.022;

        return good_configs;
    }



    bool FAsTMatch::calculateLevel() {
        level++;

        /* Randomly sample points */
        Mat xs(1, no_of_points, CV_32SC1),
                ys(1, no_of_points, CV_32SC1);

        rng.fill(xs, RNG::UNIFORM, 1, templ.cols);
        rng.fill(ys, RNG::UNIFORM, 1, templ.rows);


        /* First create configurations based on our net */
        //vector<MatchConfig> configs = createListOfConfigs(net, templ.size(), image.size());
        configs = configExpander->createListOfConfigs(templ.size(), image.size());

        int configs_count = static_cast<int>(configs.size());

        /* Convert the configurations into affine matrices */
        vector<Mat> affines = configsToAffine(configs, insiders);

        /* Filter out configurations that fall outside of the boundaries */
        /* the internal logic of configsToAffine has more information */
        vector<MatchConfig> temp_configs;
        for (int i = 0; i < insiders.size(); i++)
            if (insiders[i] == true)
                temp_configs.push_back(configs[i]);
        configs = temp_configs;

        /* For the configs, calculate the scores / distances */
        distances = evaluateConfigs(image, templ, affines, xs, ys, photometricInvariance);
        if(visualize) {
            visualizer.visualiseConfigs(original_image.clone(), configs);
        }

        /* Find the minimum distance */
        auto min_itr = min_element(distances.begin(), distances.end());
        int min_index = static_cast<int>(min_itr - distances.begin());
        double best_distance = distances[min_index];
        best_distances[level] = best_distance;

        auto max_itr = max_element(distances.begin(), distances.end());
        int max_index = static_cast<int>(max_itr - distances.begin());
        double worst_distance = distances[max_index];

        best_config = configs[min_index];
        best_trans = best_config.getAffineMatrix();


        /* Conditions to exit the loop */
        if ((best_distance < 0.005) || ((level > 2) && (best_distance < 0.015)) || level >= 20)
            return false;

        if (level > 3) {
            float mean_value =
                    (float) (std::accumulate(best_distances.begin() + level - 3, best_distances.begin() + level - 1, 0) *
                             1.0 / distances.size());

            if (best_distance > mean_value * 0.97)
                return false;
        }


        float thresh;
        bool too_high_percentage;

        /* Get the good configurations that falls between certain thresholds */
        vector<MatchConfig> good_configs = getGoodConfigsByDistance(configs, (float) best_distance, new_delta, distances,
                                                                    thresh, too_high_percentage);

        if ((too_high_percentage && (best_distance > 0.05) && ((level == 1) && (configs_count < 7.5e6))) ||
            ((best_distance > 0.1) && ((level == 1) && (configs_count < 5e6)))) {

            static float factor = 0.9;
            new_delta = new_delta * factor;
            level = 0;
            configExpander->setNet(configExpander->getNet() * factor);
            configs = configExpander->createListOfConfigs(templ.size(), image.size());
        } else {
            new_delta = new_delta / delta_fact;

            vector<MatchConfig> expanded_configs =
                    configExpander->randomExpandConfigs(good_configs, level, 80, delta_fact);

            configs.clear();
            configs.insert(configs.end(), good_configs.begin(), good_configs.end());
            configs.insert(configs.end(), expanded_configs.begin(), expanded_configs.end());
        }

        return true;
    }

    void FAsTMatch::calculate() {
        while (true) {
            if(!calculateLevel()) {
                break;
            }
        }
    }

    vector<Point> FAsTMatch::getBestCorners() {
        /* Return the rectangle corners based on the best affine transformation */
        return Utilities::calcCorners(image.size(), templ.size(), best_trans);
    }

    void FAsTMatch::setImage(const Mat &image) {
        FAsTMatch::original_image = image;
        if(image.type() == CV_8UC3) {
            cv::cvtColor(image, imageGray, CV_BGR2GRAY);
            FAsTMatch::image = Utilities::preprocessImage(imageGray);
        } else {
            FAsTMatch::image = Utilities::preprocessImage(image);
        }
        imageAvg = cv::sum(FAsTMatch::image).val[0] / ((float) image.cols * (float) image.rows);
        imageGrayAvg = cv::sum(FAsTMatch::imageGray).val[0] / ((float) image.cols * (float) image.rows);
        GaussianBlur( imageGray, imageGray, Size( 9, 9 ), 0, 0 );
#ifdef USE_CV_GPU
        imageGrayGpu.upload(imageGray);
#endif
    }

    void FAsTMatch::setTemplate(const Mat &templ_) {
        if(templ_.type() == CV_8UC3) {
            cv::cvtColor(templ_, templGray, CV_BGR2GRAY);
            FAsTMatch::templ = Utilities::preprocessImage(templGray);
        } else {
            FAsTMatch::templ = Utilities::preprocessImage(templ);
        }
        templAvg = cv::sum(FAsTMatch::templ).val[0] / ((float) templ.cols * (float) templ.rows);
        templGrayAvg = cv::sum(FAsTMatch::templGray).val[0] / ((float) templ.cols * (float) templ.cols);
        GaussianBlur( templGray, templGray, Size( 9, 9 ), 0, 0 );
#ifdef USE_CV_GPU
        templGrayGpu.upload(templGray);
#endif
    }
}