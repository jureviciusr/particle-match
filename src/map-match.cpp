#include <boost/program_options.hpp>
#include <iostream>
#include <chrono>

#include <airvision/map/BaseMap.h>
#include <FAsT-Match/FAsTMatch.h>

using namespace std::chrono;

namespace po = boost::program_options;

int main(int ac, char *av[]) {
    po::options_description desc("Allowed options");
    desc.add_options()
            ("help", "produce help message");

    po::variables_map vm;
    po::store(po::parse_command_line(ac, (const char *const *) av, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }


    airvision::BaseMap mapper(L"/home/rokas/Workspace/assets/maps/lithuania/Vilnius2/Satellite",
                              airvision::Coordinate(54.581691903,25.172194660),
                              cv::Size(4, 4),
                              18);

    cv::Mat templ = cv::imread("/home/rokas/Workspace/AirVision-Global/Pipelines/snapshots/48bc9618-4aac-4cf0-8d64-557e8604b893.png");

    cv::Mat map = mapper.getSatelliteImage();
    cv::Mat graymap;
    cv::cvtColor(map, graymap, CV_BGR2GRAY);
    fast_match::FAsTMatch fast_match;
    fast_match.init( 0.1f, 0.9f, true, 0.8f, 1.2f );

    double scaleFactor = 4;
    double scaleDownFactor = 1 / scaleFactor;
    cv::resize(graymap, graymap, cv::Size(0, 0), scaleDownFactor, scaleDownFactor, CV_INTER_NN);
    cv::resize(templ, templ, cv::Size(0, 0), scaleDownFactor, scaleDownFactor, CV_INTER_NN);

    double distance;
    std::cout << "Matching map size [" << graymap.cols << "," << graymap.rows << "] with template size ["
              << templ.cols << "," << templ.rows << "]\n";
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    std::vector<cv::Point2f> corners = fast_match.apply(graymap, templ, distance, 0, M_PI_2);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    std::cout << "* Distance : " << distance << " Time took: "
              << (double) duration_cast<milliseconds>( t2 - t1 ).count() / 1000. << "\n";

    line(map, corners[0] * scaleFactor, corners[1] * scaleFactor, Scalar(0, 0, 255), 2);
    line(map, corners[1] * scaleFactor, corners[2] * scaleFactor, Scalar(0, 0, 255), 2);
    line(map, corners[2] * scaleFactor, corners[3] * scaleFactor, Scalar(0, 0, 255), 2);
    line(map, corners[3] * scaleFactor, corners[0] * scaleFactor, Scalar(0, 0, 255), 2);

    //cv::resize(map, map, cv::Size(0, 0), 0.33, 0.33);
    //cv::imwrite("map.png", map);
    cv::namedWindow("Map", CV_WINDOW_AUTOSIZE);
    cv::imshow("Map", map);
    cv::waitKey(0);

    return 0;
}
