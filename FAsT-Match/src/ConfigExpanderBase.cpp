//
// Created by rokas on 17.6.19.
//


#include "ConfigExpanderBase.hpp"

fast_match::MatchNet ConfigExpanderBase::getNet() const {
    return *net;
}

void ConfigExpanderBase::setNet(fast_match::MatchNet net) {
    *(ConfigExpanderBase::net) = net;
}

ConfigExpanderBase::ConfigExpanderBase() {
    net = std::make_unique<fast_match::MatchNet>();
}
