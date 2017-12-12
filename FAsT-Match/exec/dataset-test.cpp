
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <iostream>

#include <fastmatch-dataset/MetadataEntryReader.hpp>
#include <opencv/cv.hpp>
#include <src/ParticleFastMatch.hpp>
#include "ParticleFilterWorkspace.hpp"

namespace fs = boost::filesystem;
namespace po = boost::program_options;

int main(int ac, char *av[]) {
    po::options_description desc("Allowed options");
    desc.add_options()
            ("map-image,m", po::value<std::string>(), "Path to map image")
            ("map-description,D", po::value<std::string>(), "Path to map description text file")
            ("dataset,d", po::value<std::string>(), "Path to dataset directory")
            ("help", "produce help message");

    po::variables_map vm;
    po::store(po::parse_command_line(ac, (const char *const *) av, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }

    // If dataset was not passed as an argument, throw error and show help
    if(!vm.count("dataset")) {
        std::cerr << "Please set dataset\n";
        std::cout << desc << "\n";
        return 1;
    }

    bool pfInitialized = false;
    std::shared_ptr<ParticleFastMatch> pfm;

    // Declare window to preview the image
    cv::namedWindow("Preview", CV_WINDOW_NORMAL);

    // Declare reader
    MetadataEntryReader reader;
    ParticleFilterWorkspace pf;
    if(vm.count("map-image") && vm.count("map-description")) {

        auto mapImage = vm["map-image"].as<std::string>();
        auto mapDescription = vm["map-description"].as<std::string>();
        if(fs::exists(mapImage) && fs::exists(mapDescription)) {
            reader.setMap(mapImage, mapDescription);
        } else {
            std::cerr << "Map configuration files were not found\n";
        }
        if(!reader.getMap()->isValid()) {
            std::cerr << "Map configuration is corrupted!\n";
        }
    }
    reader.setSkipRate(10);
    // Declare path and sanity check
    fs::path datasetPath(vm["dataset"].as<std::string>());
    if(fs::exists(datasetPath / "metadata.csv")) {
        if(reader.openDirectory(datasetPath.string())) {

            // Parse line by line into the structure
            MetadataEntry entry;
            while (reader.readNextEntry(entry)) {
                if(!pfInitialized) {
                    pf.initialize(entry);
                    pfInitialized = true;
                } else {
                    pf.update(entry);
                }
                cv::Mat image = entry.getImageColored();
                pf.preview(entry);

                // Preview the image
                cv::imshow("Preview", image);

                // Show at ~50 FPS
                int key = cv::waitKey(20);

                // Break the cycle on ESC key
                if(key == 27) {
                    break;
                }
            }
        } else {
            std::cerr << "Failed to open metadata file in the dataset\n";
        }
    } else {
        std::cerr << "Dataset directory does not contain metadata.csv file!\n";
        std::cout << desc << "\n";
        return 2;
    }

    return 0;
}
