
#include <boost/program_options.hpp>
#include <iostream>
#include <iomanip>

#include "fastmatch-dataset/Map.hpp"

namespace po = boost::program_options;

int main(int ac, char *av[]) {
    po::options_description desc("Allowed options");
    desc.add_options()
            ("image,i", po::value<std::string>(), "Map image file")
            ("description,d", po::value<std::string>(), "Map description file")
            ("help,h", "produce help message");

    po::variables_map vm;
    po::store(po::parse_command_line(ac, (const char *const *) av, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }

    if(vm.count("image") && vm.count("description")) {
        Map map(vm["image"].as<std::string>(), vm["description"].as<std::string>());
        double lat = 38.103414712, lon = -86.858718808;
        double lat1 = .0, lon1 = .0;
        cv::Point loc = map.toPixels(lat, lon);
        map.toCoords(loc, lat1, lon1);
        if(lat1 - lat < 1e-5 && lon1 - lon < 1e-5) {
            std::cout << "DISTANCE TEST PASSED\n";
        } else {
            std::cout << "DISTANCE TEST FAILED\n";
        }
    }
    return 0;
}
