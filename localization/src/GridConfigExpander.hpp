//
// Created by rokas on 17.6.19.
//

#pragma once

#include "ConfigExpanderBase.hpp"

#include "../FAsT-Match/MatchNet.h"


class GridConfigExpander : public ConfigExpanderBase {
protected:
public:
    GridConfigExpander();

protected:
    cv::RNG rng;
    virtual std::vector<fast_match::MatchConfig> createListOfConfigs(cv::Size templ_size, cv::Size image_size) override;

    virtual std::vector<fast_match::MatchConfig>
    randomExpandConfigs(std::vector<fast_match::MatchConfig> &configs, int level, int no_of_points,
                        float delta_factor) override;

};


