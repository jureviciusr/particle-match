//
// Created by  on 2020-02-29.
//

#include <cmath>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <iostream>
#include <fastmatch-dataset/MetadataEntryReader.hpp>
#include <opencv2/imgproc.hpp>
#include <src/Particle.hpp>
#include <src/ImageSample.hpp>
#include <chrono>
#ifdef USE_TBB
#include <tbb/tbb.h>
#endif

namespace po = boost::program_options;
namespace fs = boost::filesystem;
using namespace std::chrono;

Particle locationAsParticle(const MetadataEntry &entry) {
    Particle particle(entry.mapLocation.x, entry.mapLocation.y);
    float currentScale = std::tan(1.0f / 2.0f) * entry.altitude / 320.0f;
    float min = 0.9f * currentScale;
    float max = 1.1f * currentScale;
    int steps = 5;
    std::shared_ptr<std::vector<float>> s_initial = std::make_shared<std::vector<float>>();;
    float delta = (std::abs(min - max)) / (float) (steps - 1);
    for (uint32_t i = 0; i < steps; i++) {
        s_initial->push_back(min + (i * delta));
    }
    particle.setS_initial(s_initial);
    particle.setDirection(entry.imuOrientation.toRPY().getZ());
    return particle;
}

float calculateSimilarity(const cv::Mat& im1, const cv::Mat& im2, const std::vector<cv::Point>& samplePoints) {
    steady_clock::time_point begin = steady_clock::now();
    ImageSample cam_sample(im1, samplePoints);
    ImageSample map_sample(im2, samplePoints);
    float val = cam_sample.calcSimilarity(map_sample);
    steady_clock::time_point end = steady_clock::now();
    std::cout << "Time: " << duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
    return val;

}

int main(int ac, char *av[]) {
    po::options_description desc("Allowed options");
    desc.add_options()
            ("map-image,m", po::value<std::string>(), "Path to map image")
            ("dataset,d", po::value<std::string>(), "Path to dataset directory")
            ("help,h", "produce help message");

    po::variables_map vm;
    // Replace the ugly boost error message with help message
    try {
        po::store(po::parse_command_line(ac, (const char *const *) av, desc), vm);
        po::notify(vm);
    } catch (po::error &e) {
        std::cerr << e.what() << "\n" << desc << "\n";
        return 1;
    }

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 0;
    }

    MetadataEntryReader reader;
    std::string mapName;
    if (vm.count("map-image")) {
        auto mapImage = vm["map-image"].as<std::string>();
        mapName = fs::path(mapImage).stem().string();
        if (fs::exists(mapImage)) {
            reader.setMap(mapImage);
        } else {
            std::cerr << "Map configuration files were not found\n";
        }
        if (!reader.getMap()->isValid()) {
            std::cerr << "Map configuration is corrupted!\n";
        }
    }
    fs::path datasetPath(vm["dataset"].as<std::string>());

    std::vector<cv::Point> samplePoints;


    cv::namedWindow("cam", cv::WINDOW_NORMAL);
    cv::namedWindow("map", cv::WINDOW_NORMAL);
    if (fs::exists(datasetPath / "metadata.csv") && reader.openDirectory(datasetPath.string())) {
        MetadataEntry entry;
        int iteration = 0;
        cv::Mat map;
        cv::cvtColor(reader.getMap()->getImage(), map, cv::COLOR_BGR2GRAY);
        reader.readNextEntry(entry);
        cv::Mat camview = entry.getImage();
        Particle particle = locationAsParticle(entry);
        cv::Mat mapview = particle.getMapImage(map, camview.size());

        float fullSimilarity, blurredSimilarity, halfSimilarity, randomSimilarity;
        // Test 1, full image similarity
        for (int y_ = 0; y_ < camview.rows; y_++) {
            for (int x_ = 0; x_ < camview.cols; x_++) {
                samplePoints.emplace_back(x_, y_);
            }
        }
        fullSimilarity = calculateSimilarity(mapview, camview, samplePoints);
        std::cout << "Full similarity: " << fullSimilarity << std::endl;

        cv::GaussianBlur(camview, camview, cv::Size(9, 9), 0.0, 0.0);
        cv::GaussianBlur(map, map, cv::Size(9, 9), 0.0, 0.0);
        mapview = particle.getMapImage(map, camview.size());
        blurredSimilarity = calculateSimilarity(mapview, camview, samplePoints);
        std::cout << "Blurred similarity: " << blurredSimilarity << std::endl;

        samplePoints.clear();
        for (int y_ = 0; y_ < camview.rows; y_ += 2) {
            for (int x_ = 0; x_ < camview.cols; x_ += 2) {
                samplePoints.emplace_back(x_, y_);
            }
        }
        halfSimilarity = calculateSimilarity(mapview, camview, samplePoints);
        std::cout << "Half similarity: " << halfSimilarity << std::endl;

        samplePoints.clear();
        for (int y_ = 0; y_ < (float) (camview.rows * camview.cols) * 0.05f; y_ ++) {
            samplePoints.emplace_back(
                    (int) (Utilities::uniform_dist() * camview.cols),
                    (int) (Utilities::uniform_dist() * camview.rows)
                    );
        }
        randomSimilarity = calculateSimilarity(mapview, camview, samplePoints);
        std::cout << "Random similarity: " << randomSimilarity << std::endl;
        // Sorting these coordinates can reduce cache misses
        std::sort(samplePoints.begin(), samplePoints.end(),[] (const cv::Point& a, const cv::Point& b) {
            return a.y == b.y ? a.x < b.x : a.y < b.y;
        });
        randomSimilarity = calculateSimilarity(mapview, camview, samplePoints);
        std::cout << "Random sorted similarity: " << randomSimilarity << std::endl;

        // This is just a test to evaluate large number of images.
        std::cout << "Evaluating " << samplePoints.size() << " particles..." << std::endl;
        int ox = particle.x, oy = particle.y;
        std::vector<float> falses;
        ImageSample cam_sample(camview, samplePoints);
        float mapAverage = cv::sum(map).val[0] / (float) (map.cols * map.rows);
        steady_clock::time_point begin = steady_clock::now();
#ifdef USE_TBB
        tbb::parallel_for_each(samplePoints.begin(), samplePoints.end(), [&] (const cv::Point2i& offset) {
            particle.x = ox + offset.x;
            particle.y = ox + offset.y;
            ImageSample mymapsample(map, samplePoints, particle.mapTransformation(), particle.toPoint(), mapAverage);
            falses.push_back(cam_sample.calcSimilarity(mymapsample));
        });
#else
        for (const auto& offset : samplePoints) {
            particle.x = ox + offset.x;
            particle.y = ox + offset.y;
            ImageSample mymapsample(map, samplePoints, particle.mapTransformation(), particle.toPoint(), mapAverage);
            falses.push_back(cam_sample.calcSimilarity(mymapsample));
        }
#endif
        steady_clock::time_point end = steady_clock::now();
        double dur = duration_cast<std::chrono::microseconds>(end - begin).count();
        std::cout << "Time took: " << dur / 1000000.0 << "[s]" << std::endl;
        std::cout << "Time per evaluation: " << dur / (double) samplePoints.size() << "[µs]" << std::endl;

        float maxFalse = *std::max_element(falses.begin(), falses.end());
        std::cout << "Max false similarity: " << maxFalse << std::endl;


        cv::imshow("cam", camview);
        cv::waitKey(10);
        cv::imshow("map", mapview);
        cv::waitKey(0);
    } else {
        std::cerr << "Dataset directory does not contain metadata.csv file!\n";
        std::cout << desc << "\n";
        return 2;
    }


    return 0;
}
