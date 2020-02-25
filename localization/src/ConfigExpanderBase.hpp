//
// Created by rokas on 17.6.19.
//

#pragma once

#include <FAsT-Match/MatchNet.h>
#include "../FAsT-Match/MatchConfig.h"


class ConfigExpanderBase {
protected:
    std::unique_ptr<fast_match::MatchNet> net;

public:
    ConfigExpanderBase();
    fast_match::MatchNet getNet() const;
    void setNet(fast_match::MatchNet net);
    virtual std::vector<fast_match::MatchConfig> createListOfConfigs(cv::Size templ_size, cv::Size image_size) = 0;
    virtual std::vector<fast_match::MatchConfig> randomExpandConfigs(std::vector<fast_match::MatchConfig> &configs,
                                                                     int level, int no_of_points,
                                                                     float delta_factor) = 0;

};


