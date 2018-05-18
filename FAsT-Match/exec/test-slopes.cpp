
#include <boost/program_options.hpp>
#include <iostream>

namespace po = boost::program_options;

float fn(float ccoef, float lowBound) {
    float prob;
    if (lowBound < 0.0f) {
        lowBound = std::abs(lowBound);
        if (ccoef > (lowBound + 0.00001f)) {
            float slope = 1 + lowBound;
            prob = (slope * ccoef) - (slope * lowBound) + (lowBound * lowBound);
        } else {
            prob = 0.f;
        }
    } else {
        if (ccoef > 0.f) {
            prob = lowBound + (ccoef * (1.f - lowBound));
        } else {
            prob = lowBound - (std::abs(ccoef) * lowBound);
        }
    }
    return prob;
}

int main(int ac, char *av[]) {
    po::options_description desc("Allowed options");
    desc.add_options()
            ("help,h", "produce help message");

    po::variables_map vm;
    // Replace the ugly boost error message with help message
    try {
        po::store(po::parse_command_line(ac, (const char *const *) av, desc), vm);
        // Fill missing values from config file
        po::notify(vm);
    } catch (po::error &e) {
        std::cerr << e.what() << "\n" << desc << "\n";
        return 1;
    }

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }
    float d = -0.4f;
    for(int j = 0; j < 7; j++) {
        std::cout << "============ RUNNING d = " << d << "\n";
        float in = -1.f;
        for(int i = 0; i < 21; i++) {
            float val = fn(in, d);
            std::cout << std::fixed << in << " => " << val << "\n";
            in += 0.1f;
        }
        d += .1f;
    }

    return 0;
}
