//
// Created by rokas on 17.6.20.
//

#pragma once

#include "FastMatch.hpp"
#include "pf/Particles.hpp"
#include "AffineTransformation.hpp"

class ParticleFastMatch : public fast_match::FAsTMatch {
public:
    enum MatchMode {
        PearsonCorrelation, BriskMatch, ORBMatch
    };

    enum ConversionMode {
        HPRELU, GLF, Softmax
    };

    ConversionMode conversionMode = HPRELU;

    cv::Mat templDescriptors;
    cv::cuda::GpuMat templGpuDescriptors;

    MatchMode matching = PearsonCorrelation;

    cv::Ptr<cv::Feature2D> detector;

    //cv::Ptr<cv::cuda::Feature2DAsync> detectorGPU;

    cv::Ptr<cv::cuda::DescriptorMatcher> matcher;

    ParticleFastMatch(
            const cv::Point2i& startLocation,
            const cv::Size& mapSize,
            double radius,
            float epsilon,
            int particleCount,
            float quantile_ = 0.7,
            float kld_error_ = 0.8,
            int bin_size_ = 5,
            bool use_gaussian = false);
    void visualizeParticles(cv::Mat image);

    vector<Point> filterParticlesAffine(const cv::Point2f &movement, cv::Mat &bestTransform);

    vector<Point> filterParticles(const cv::Point2f &movement, cv::Mat &bestTransform);

    cv::Mat evaluateParticle(Particle& particle, int id, double &bestProbability);

    void propagateParticles(const cv::Point2f& movement);

    void setDirection(const double& _d);

    virtual vector<double> evaluateConfigs( Mat& templ, vector<AffineTransformation>& affine_matrices,
                                           Mat& xs, Mat& ys, bool photometric_invariance );


    void setImage(const Mat &image) override;

    float calculateSimilarity(cv::Mat im) const;

protected:
    vector<AffineTransformation> configsToAffine(vector<fast_match::MatchConfig> &configs, vector<bool> &insiders);

    Particles particles;
public:
    const Particles &getParticles() const;

protected:
    float kld_error = 0.5f;
    int binSize = 5;

    cv::Mat xs, ys, paddedCurrentImage;

    int minParticles = 50;

    float zvalue;
    std::vector<float> ztable;

    float lowBound = 0.00f;
public:
    float getLowBound() const;

    void setLowBound(float lowBound);

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

    void setScale(float min, float max, uint32_t searchSteps = 5);

    cv::Mat getBestParticleView(cv::Mat map);

    void calculateSimilarity(cv::cuda::GpuMat im, Particle& particle) const;

    float convertProbability(float in) const;

    std::string conversionModeString() const;
};


