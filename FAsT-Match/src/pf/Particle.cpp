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

    cv::Point2f m = movement;
    float alpha = 4.0f;
    float min_movement = 10.0f;
    float min_noise_level = 5.0f;
    // This is the case when the odometry is lost
    if(movement.x == 0.f && movement.y == 0.f) {
        m.x = static_cast<float>(Utilities::gausian_noise(1) * min_movement * alpha);
        m.y = static_cast<float>(Utilities::gausian_noise(1) * min_movement * alpha);
    } else {
        // Do not trust in noise level measurements
        if(movement.x < min_noise_level) {
            m.x += Utilities::gausian_noise(1) * min_noise_level * alpha;
        } else {
            m.x += Utilities::gausian_noise(1) * m.x * alpha;
        }
        if(movement.y < min_noise_level) {
            m.y += Utilities::gausian_noise(1) * min_noise_level * alpha;
        } else {
            m.y += Utilities::gausian_noise(1) * m.y * alpha;
        }
    }
    x += m.x;
    y += m.y;
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
    auto scale_steps = (int) s_initial->size();
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

    for(uint64_t sx = 0; sx < scale_steps; sx++) {
        for (uint64_t sy = 0; sy < scale_steps; sy++) {
            for (int r1 = 0; r1 < rotation_steps; r1++) {
                for (int r2 = 0; r2 < nr2_steps; r2++) {
                    configs.emplace_back(
                            x - mapCenter.x,
                            y - mapCenter.y,
                            r2_rotations[r2],
                            s_initial->at(sx),
                            s_initial->at(sy),
                            rotations[r1]
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
    bestTransform = configs[min_index].getAffineMatrix();
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

void Particle::setS_initial(const shared_ptr<vector<float>> &s_initial) {
    Particle::s_initial = s_initial;
}

cv::Point2i Particle::getLocationInMapCoords() const {
    return {x - mapCenter.x, y - mapCenter.y};
}

double Particle::getDirection() {
    return direction;
}

double Particle::getDirectionDegrees() const {
    return direction * 57.2958;
}

cv::Point2i Particle::toPoint() const {
    return cv::Point2i(x, y);
}

cv::Mat Particle::staticTransformation() const {
    cv::Mat T = cv::getRotationMatrix2D(cv::Point(320, 240), getDirectionDegrees() , getScale());
    //T.at<float>(0, 2) += (float) x;
   // T.at<float>(1, 2) += (float) y;
    return T;
}

float Particle::getScale() const {
    return (*s_initial)[2];
}

Particle::Particle(const Particle &a) = default;

