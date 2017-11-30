//
// Created by rokas on 17.6.20.
//

#include <src/Utilities.hpp>
#include "Particles.hpp"

void Particles::init(cv::Point2i startLocation, const cv::Size mapSize,  double radius, int particleCount, bool use_gaussian) {
    double r, a;
    int size = 0;
    Particle::setMapDimensions(mapSize);
    while (size < particleCount) {
        if(use_gaussian) {
            a = ((Utilities::gausian_noise(1.0f) - 0.5) * 2) * 2 * M_PI;
            double u = Utilities::gausian_noise(radius) + Utilities::gausian_noise(radius);
            r = u > radius ? (2 * radius) - u : u;
        } else {
            a = ((Utilities::uniform_dist() - 0.5) * 2) * 2 * M_PI;
            double r1 = Utilities::uniform_dist() * radius,
                    r2 = Utilities::uniform_dist() * radius;
            double u = r1 + r2;
            r = u > radius ? (2 * radius) - u : u;
        }
        auto x = (int) (startLocation.x + (r * cos(a)));
        auto y = (int) (startLocation.y + (r * sin(a)));
        // Skip duplicate particles
        if(!isLocationOccupied(x, y)) {
            addParticle(x, y);
            back().setProbability(.5f);
            size++;
        }
    }
    normalize();
}

float Particles::getMinScale() const {
    return minScale;
}

void Particles::setMinScale(float minScale) {
    Particles::minScale = minScale;
}

float Particles::getMaxScale() const {
    return maxScale;
}

void Particles::setMaxScale(float maxScale) {
    Particles::maxScale = maxScale;
}

void Particles::addParticle(int x, int y) {
    emplace_back(x, y);
}

void Particles::addParticle(Particle p) {
    emplace_back(p);
}

std::vector<fast_match::MatchConfig> Particles::getConfigs() {
    std::vector<fast_match::MatchConfig> configs;
    int i = 0;
    for (auto &it : *this) {
        auto& curConfigs = it.getConfigs(i++);
        configs.insert(configs.end(), curConfigs.begin(), curConfigs.end());
    }
    return configs;
}

bool Particles::isLocationOccupied(int x, int y) {
    for (const_iterator it = begin() ; it != end(); ++it) {
        if((*it).x == x && (*it).y == y) {
            return true;
        }
    }
    return false;
}

void Particles::propagate(const cv::Point2f &movement, float alpha) {
    for (auto &it : *this) {
        it.propagate(cv::Point2f(
                movement.x * alpha * rng.uniform(-1.f, 1.f),
                movement.y * alpha * rng.uniform(-1.f, 1.f)
        ));
    }
}

void Particles::printProbabilities() {
    int i = 1;
    for (const_iterator it = begin(); it != end(); ++it) {
        std::cout << "Particle no " << i++ << " probability is: " << (*it).getProbability() << "\n";
    }
}

void Particles::assignProbabilities(const std::vector<fast_match::MatchConfig> &configs,
                                    const std::vector<double> probabilities) {

}

std::vector<cv::Point> Particles::evaluate(cv::Mat image, cv::Mat templ, int no_of_points) {
    /* Randomly sample points */
    cv::Mat xs(1, no_of_points, CV_32SC1),
            ys(1, no_of_points, CV_32SC1);
    rng.fill(xs, cv::RNG::UNIFORM, 1, templ.cols);
    rng.fill(ys, cv::RNG::UNIFORM, 1, templ.rows);

    double lowestDistance = +INFINITY;
    cv::Mat bestTrasform;
    for (auto it = begin() ; it != end(); ++it) {
        double distance = (*it).evaluate(image, templ, xs, ys);
        if(distance < lowestDistance) {
            lowestDistance = distance;
            bestTrasform = (*it).getBestTransform();
        }
    }
    return Utilities::calcCorners(image.size(), templ.size(), bestTrasform);
}

Particle Particles::sample() {
    double sampleThreshold = Utilities::uniform_dist();
    if(sampleThreshold < .5f) {
        for (const auto &it : *this) {
            if (it.getSamplingFactor() > sampleThreshold) {
                return Particle(it);
            }
        }
    } else {
        for (auto i = rbegin(); i != rend(); ++i ) {
            if (i->getSamplingFactor() < sampleThreshold) {
                return Particle(*i);
            }
        }
    }
}

void Particles::sortAscending() {
    std::sort(begin(), end(), std::greater<>());
}


void Particles::normalize() {
    float normalizationFactor = 0.f,
            total = 0.f;
    for (const auto &it : *this) {
        normalizationFactor += it.getProbability();
    }
    for (auto &it : *this) {
        float weight = it.getProbability() / normalizationFactor;
        it.setWeight(weight);
        total += weight;
        it.setSamplingFactor(1 - total);
    }
}

cv::Point2i Particles::getWeightedSum() const {
    double s_x = .0, s_y = .0;
    for(const auto& it : *this) {
        s_x += it.x * it.getWeight();
        s_y += it.y * it.getWeight();
    }
    return cv::Point2i((int) s_x, (int) s_y);
}




