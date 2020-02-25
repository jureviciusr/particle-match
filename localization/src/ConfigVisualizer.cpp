//
// Created by rokas on 17.6.20.
//

#include "ConfigVisualizer.hpp"

void ConfigVisualizer::visualiseConfigs(cv::Mat image, const std::vector<fast_match::MatchConfig>& configs) {
    std::vector<cv::Point2i> drawnPoints;
    for(const auto& config : configs) {
        cv::Point2i curPoint(cv::Point(
                (int) config.getTranslateX() + (image.cols / 2),
                (int) config.getTranslateY() + (image.rows / 2)
        ));
        if(std::find(drawnPoints.begin(), drawnPoints.end(), curPoint) == drawnPoints.end()) {
            cv::drawMarker(
                    image,
                    curPoint,
                    cv::Scalar(0, 255, 0),
                    cv::MARKER_TILTED_CROSS,
                    10,
                    3
            );
            drawnPoints.push_back(curPoint);
        }
    }
    cv::imshow("Preview", image);
    cv::waitKey(1000);
}

ConfigVisualizer::ConfigVisualizer() = default;

void ConfigVisualizer::visualiseParticles(cv::Mat image, const Particles& particles) {
    //cv::Mat preview = image.clone();
    for(auto iter = particles.rbegin(); iter != particles.rend(); iter++) {
        cv::Point2i curPoint(
                (*iter).x,
                (*iter).y
        );
        cv::Scalar color(
                (*iter).getProbability() * 255,
                0,
                255 - ((*iter).getProbability() * 255)
        );
        cv::drawMarker(
                image,
                curPoint,
                color,
                cv::MARKER_TILTED_CROSS,
                50,
                5
        );
    }

}
