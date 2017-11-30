//
// Created by rokas on 17.6.20.
//

#include <zconf.h>
#include <src/Utilities.hpp>
#include <src/FastMatch.hpp>
#include "Particle.hpp"

float Particle::r_step = 0.05f;

double Particle::direction = .0;

// -5deg ~ + 5deg
std::vector<float> Particle::r_initial = {
        -(3 * Particle::r_step),
        -(2.5f * Particle::r_step),
        -(2 * Particle::r_step),
        -(1.5f * Particle::r_step),
        -(Particle::r_step),
        -(0.5f * Particle::r_step),
        0.f,
        0.5f * Particle::r_step,
        Particle::r_step,
        1.5f * Particle::r_step,
        2 * Particle::r_step,
        2.5f * Particle::r_step,
        3 * Particle::r_step
};

std::vector<float> Particle::s_initial = {.9f, .95f, 1.f, 1.05f, 1.1f};

cv::Point2i Particle::mapCenter;

float Particle::getProbability() const {
    return probability;
}

void Particle::setProbability(float probability) {
    accumulatedProbability += probability;
    oldProbabilities.push_back(probability);
    if(oldProbabilities.size() >= 5) {
        accumulatedProbability -= oldProbabilities[0];
        oldProbabilities.erase(oldProbabilities.begin());
    } else {
        iteration++;
    }
    Particle::probability = accumulatedProbability / iteration;
    //Particle::probability = probability;
}

//Particle::Particle() : x(0), y(0), probability(1.0) {}

Particle::Particle(int x, int y) : x(x), y(y), probability(1.0) {
    updateConfigs();
}

const vector<fast_match::MatchConfig> & Particle::getConfigs(int id) {
    for (auto& config : configs) {
        config.setId(id);
    }
    return configs;
}

void Particle::propagate(const cv::Point2f &movement) {
    double alpha = 2.0;
    x += movement.x + ((Utilities::gausian_noise(1)) * (movement.x * alpha));
    y += movement.y + ((Utilities::gausian_noise(1)) * (movement.y * alpha));
    updateConfigs();
}

void Particle::setMapDimensions(const cv::Size& dims) {
    mapCenter.y = dims.height / 2;
    mapCenter.x = dims.width / 2;
}

void Particle::setDirection(double direction) {
    Particle::direction = direction;
}

void Particle::updateConfigs() {
    configs.clear();
    auto scale_steps = (int) s_initial.size();
    auto rotation_steps = (int) r_initial.size();


    static std::vector<float> r2_rotations = {
            -(3 * Particle::r_step),
            0,
            3 * Particle::r_step,
    };
    unsigned long nr2_steps = r2_rotations.size();

    auto rotations = r_initial;
    for(float& rotation : rotations) {
        rotation += Particle::direction;
    }

    for(int sx = 0; sx < scale_steps; sx++) {
        for (int sy = 0; sy < scale_steps; sy++) {
            for (int r1 = 0; r1 < rotation_steps; r1++) {
                for (int r2 = 0; r2 < nr2_steps; r2++) {
                    configs.emplace_back(
                            x - mapCenter.x, y - mapCenter.y, r2_rotations[r2], s_initial[sx], s_initial[sy], rotations[r1]
                    );
                }
            }
        }
    }
}

double Particle::evaluate(cv::Mat &image, cv::Mat &templ, cv::Mat &xs, cv::Mat &ys) {
    std::vector<cv::Mat> affines = getAffines(image.size(), templ.size());
    /* For the configs, calculate the scores / distances */
    std::vector<double> distances = fast_match::FAsTMatch::evaluateConfigs(image, templ, affines, xs, ys, true);
    /* Find the minimum distance */
    auto min_itr = min_element(distances.begin(), distances.end());
    int min_index = static_cast<int>(min_itr - distances.begin());
    double best_distance = distances[min_index];

    //best_distances[level] = best_distance;

    //auto max_itr = max_element(distances.begin(), distances.end());
    //int max_index = static_cast<int>(max_itr - distances.begin());
    //double worst_distance = distances[max_index];

    bestTransform = configs[min_index].getAffineMatrix();

    //return Utilities::calcCorners(image.size(), templ.size(), best_trans);
    setProbability((float) best_distance);
    return best_distance;
}

std::vector<cv::Mat> Particle::getAffines(const cv::Size& imageSize, const cv::Size& templSize) {
    std::vector<bool> insiders;
    std::vector<cv::Mat> affines = Utilities::configsToAffine(configs, insiders, imageSize, templSize);

    /* Filter out configurations that fall outside of the boundaries */
    /* the internal logic of configsToAffine has more information */
    std::vector<fast_match::MatchConfig> temp_configs;
    for (int i = 0; i < insiders.size(); i++)
        if (insiders[i] == true)
            temp_configs.push_back(configs[i]);
    configs = temp_configs;

    return affines;
}

const Mat &Particle::getBestTransform() const {
    return bestTransform;
}

void Particle::setMinimalProbability(float probability) {
    if(Particle::probability > probability) {
        Particle::probability = probability;
    }
}

void Particle::setMaximalProbability(float probability) {
    if(Particle::probability < probability) {
        Particle::probability = probability;
    }
}

float Particle::getSamplingFactor() const {
    return samplingFactor;
}

void Particle::setSamplingFactor(float samplingFactor) {
    Particle::samplingFactor = samplingFactor;
}

bool Particle::operator<(const Particle &str) const {
    return samplingFactor < str.samplingFactor;
}

bool Particle::operator>(const Particle &str) const {
    return samplingFactor > str.samplingFactor;
}

float Particle::getWeight() const {
    return weight;
}

void Particle::setWeight(float weight) {
    Particle::weight = weight;
}

std::string Particle::serialize(int binSize) {
    return std::to_string(x - (x % binSize)) + "x" + std::to_string(y - (y % binSize));
}

Particle::Particle(const Particle &a) = default;

