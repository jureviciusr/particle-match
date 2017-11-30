//
// Created by rokas on 17.6.20.
//

#pragma once

#include "FastMatch.hpp"
#include "pf/Particles.hpp"
#include "AffineTransformation.hpp"

class ParticleFastMatch : public fast_match::FAsTMatch {
public:
    ParticleFastMatch(
            cv::Point2i startLocation,
            const cv::Size mapSize,
            double radius,
            float epsilon,
            int particleCount,
            float quantile_ = 0.7,
            float kld_error_ = 0.8,
            int bin_size_ = 5,
            bool use_gaussian = false,
            float _min_scale = .9f,
            float _max_scale = 1.1f);
    void visualizeParticles(cv::Mat image);

    vector<Point> filterParticles(const cv::Point2f& movement, cv::Mat& bestTransform);

    cv::Mat evaluateParticle(Particle& particle, int id, double &bestProbability);

    void propagateParticles(const cv::Point2f& movement);

    void setDirection(const double& _d);

    virtual vector<double> evaluateConfigs( Mat& templ, vector<AffineTransformation>& affine_matrices,
                                           Mat& xs, Mat& ys, bool photometric_invariance );


    void setImage(const Mat &image) override;

protected:
    vector<AffineTransformation> configsToAffine(vector<fast_match::MatchConfig> &configs, vector<bool> &insiders);

    Particles particles;
    float kld_error = 0.5f;
    int binSize = 5;

    cv::Mat xs, ys, paddedCurrentImage;

    int minParticles = 50;
    cv::Mat map;

    float zvalue;
    std::vector<float> ztable;


private:
    using fast_match::FAsTMatch::init;

public:

    vector<Point> evaluateParticlesv2();

    void setTemplate(const Mat &templ) override;

    void buildZTable();

    void initTemplatePixels();

    void initPaddedImage();

    uint32_t particleCount() const;

    cv::Point2i getPredictedLocation() const;

};


