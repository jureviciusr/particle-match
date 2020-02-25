
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <iostream>
#include <opencv/cv.hpp>
#include <curl/curl.h>
#include <iomanip>

#include "fastmatch-dataset/GeotiffMap.hpp"
#include "xml-parser.h"
#include "PascalVocWriter.hpp"

namespace po = boost::program_options;
namespace fs = boost::filesystem;
using boost::format;

static const std::string osm_api_url = "https://api.openstreetmap.org/api/0.6/map?bbox=";

size_t curl_write_fn(void *ptr, size_t size, size_t nmemb, void* data) {
    auto* str = static_cast<std::string*>(data);
    str->append((char*) ptr, size * nmemb);
    return size * nmemb;
}

std::unique_ptr<Xml> performOSMApiRequest(const GeographicLib::GeoCoords& from, const GeographicLib::GeoCoords& to) {
    auto curl = curl_easy_init();
    if (curl) {
        std::stringstream urlBuilder;
        urlBuilder << std::fixed << std::setprecision(9) << osm_api_url << from.Longitude() << "," << to.Latitude()
                   << "," << to.Longitude() << "," << from.Latitude();
        curl_easy_setopt(curl, CURLOPT_URL, urlBuilder.str().c_str());
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 1L);
        curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 50L);
        curl_easy_setopt(curl, CURLOPT_TCP_KEEPALIVE, 1L);
        curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);

        std::string response_string;
        std::string header_string;
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);
        curl_easy_setopt(curl, CURLOPT_HEADERDATA, &header_string);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_fn);

        char* url;
        long response_code;
        double elapsed;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
        curl_easy_getinfo(curl, CURLINFO_TOTAL_TIME, &elapsed);
        curl_easy_getinfo(curl, CURLINFO_EFFECTIVE_URL, &url);


        curl_easy_perform(curl);

        curl_easy_cleanup(curl);

        std::cout << response_string << "\n";
        if(!response_string.empty()) {
            std::unique_ptr<Xml> parsed = std::make_unique<Xml>();
            parsed->parseXML(response_string);
            return std::move(parsed);
        }
    }
    return nullptr;
}


int main(int ac, char *av[]) {
    po::options_description desc("Allowed options");
    desc.add_options()
            ("image-map,i", po::value<std::string>(), "Geotiff map to be for image only")
            ("geotiff-map,m", po::value<std::string>()->required(), "Geotiff map to be used")
            ("tiles-size,s", po::value<int>()->default_value(480), "Cutting tile dimension")
            ("output-directory,o", po::value<std::string>()->default_value("output"), "Tile output directory")
            ("no-xml,n", "Do not generate XML")
            ("filename-prefix,f", po::value<std::string>()->default_value("img"), "A prefix to use while generating files")
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
    std::string prefix = vm["filename-prefix"].as<std::string>();
    int tileSize = vm["tiles-size"].as<int>();
    fs::path outputDir(vm["output-directory"].as<std::string>());
    if(!fs::exists(outputDir)) {
        fs::create_directories(outputDir);
    }
    bool noXML = vm.count("no-xml") > 0;
    GeotiffMap map;
    map.open(vm["geotiff-map"].as<std::string>());
    cv::Mat mapImage;
    if(vm.count("image-map") > 0) {
        mapImage = cv::imread(vm["image-map"].as<std::string>());
    } else {
        mapImage = map.getImage();
    }

    cv::Mat streetMask = cv::Mat(mapImage.size(), CV_8UC1, cv::Scalar(0));

    auto streets = performOSMApiRequest(
            map.pixelCoordinates(cv::Point(0, 0)),
            map.pixelCoordinates(cv::Point(mapImage.cols, mapImage.rows))
    );

    cv::namedWindow("preview", cv::WINDOW_NORMAL);

    if(streets) {
        std::vector<std::vector<cv::Point>> cvStreets;
        std::cout << "Street count = " <<  streets->ways.size() << "\n";
        for(const auto& way : streets->ways) {
            std::vector<cv::Point> street;
            for(const auto& node : way.nodes) {
                if (!way.type.empty()) {
                    ffPair coords = streets->nodes[node];
                    cv::Point2i pxCoords = map.toPixels(coords.second, coords.first);
                    street.emplace_back(pxCoords);
                }
            }
            cvStreets.push_back(street);
        }
        cv::polylines(streetMask, cvStreets, false, cv::Scalar(255), 12);
        int tilesHoriz = (mapImage.cols / tileSize);
        int tilesVert = (mapImage.rows / tileSize);
        int index = 0;
        for(int vIndex = 0; vIndex < tilesVert; vIndex++) {
            for(int hIndex = 0; hIndex < tilesHoriz; hIndex++) {
                cv::Point from(hIndex * tileSize, vIndex * tileSize),
                          to((hIndex + 1) * tileSize, (vIndex + 1) * tileSize);
                cv::Mat tile = mapImage(cv::Rect(from, to));
                cv::Mat tileMask = streetMask(cv::Rect(from, to));
                auto fromCoords = map.pixelCoordinates(from);
                auto toCoords = map.pixelCoordinates(to);
                std::vector<std::vector<cv::Point>> contours;
                std::vector<cv::Vec4i> hierarchy;
                fs::path outImage = fs::path(outputDir) / (format("%s_%05d.jpg") % prefix % index).str();
                if(!noXML) {
                    PascalVocWriter xmlWriter(outImage, tile.size());
                    cv::findContours(tileMask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_TC89_KCOS);
                    for(const auto& contour : contours) {
                        xmlWriter.addPolygon(contour, "street");
                    }
                    xmlWriter.write();
                }
                cv::imwrite(outImage.string(), tile);
                index++;
            }
        }
        return 0;
    } else {
        return 1;
    }
}
