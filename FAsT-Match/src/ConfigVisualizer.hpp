//
// Created by rokas on 17.6.20.
//

#pragma once


#include <FAsT-Match/MatchConfig.h>
#include <src/pf/Particles.hpp>

class ConfigVisualizer {
public:
    ConfigVisualizer();
    void visualiseConfigs(cv::Mat image, const std::vector<fast_match::MatchConfig>& configs);
    void visualiseParticles(cv::Mat image, const Particles& particles);

};


