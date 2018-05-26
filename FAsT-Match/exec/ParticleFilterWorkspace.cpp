//
// Created by rokas on 17.12.4.
//

#include <src/Utilities.hpp>
#include <boost/filesystem/path.hpp>
#include "ParticleFilterWorkspace.hpp"

namespace fs = boost::filesystem;

void ParticleFilterWorkspace::initialize(const MetadataEntry &metadata) {
    direction = metadata.imuOrientation.toRPY().getZ();
    Particle::setDirection(direction);
    svoCurPosition = metadata.mapLocation;
    svoCoordinates = std::make_shared<GeographicLib::LocalCartesian>(
            metadata.latitude,
            metadata.longitude,
            metadata.altitude
    );
    pfm = std::make_shared<ParticleFastMatch>(
            svoCurPosition, // startLocation
            metadata.map.size(), // mapSize
            500, // radius
            .1f, // epsilon
            200, // particleCount
            .99, // quantile_
            .5, // kld_error_
            5, // bin_size_
            true // use_gaussian
    );
    cv::Mat templ = metadata.getImageColored();
    pfm->setTemplate(templ);
    pfm->setImage(metadata.map.clone());
    map = metadata.map;
    updateScale(1.0, static_cast<float>(metadata.altitude), 640);
    if(displayImage) {
        cv::namedWindow("Map", cv::WINDOW_NORMAL);
    }
    startLocation = pfm->getPredictedLocation();
    //cv::namedWindow("BestTransform", CV_WINDOW_NORMAL);
}

void ParticleFilterWorkspace::update(const MetadataEntry &metadata) {
    cv::Point movement = getMovementFromSvo(metadata);
    updateScale(1.0, static_cast<float>(metadata.altitude), 640);
    direction = metadata.imuOrientation.toRPY().getZ();
    Particle::setDirection(direction);
    cv::Mat templ = metadata.getImageColored();
    pfm->setTemplate(templ);
    if(!affineMatching) {
        pfm->filterParticles(movement, bestTransform);
        bestView = pfm->getBestParticleView(metadata.map);
        //cv::Mat bestView = Utilities::extractWarpedMapPart(metadata.map, templ.size(), bestTransform);
    } else {
        corners = pfm->filterParticlesAffine(movement, bestTransform);
        //cv::Mat bestView = Utilities::extractWarpedMapPart(metadata.map, templ.size(), bestTransform);
    }
   // cv::Point2i prediction = pfm->getPredictedLocation();
}

bool ParticleFilterWorkspace::preview(const MetadataEntry &metadata, cv::Mat planeView, std::stringstream& stringOutput)
const {
    cv::Point2i prediction = pfm->getPredictedLocation();
    cv::Point2i relativeLocation = prediction - startLocation;
    stringOutput << pfm->particleCount() << ",";
    stringOutput << relativeLocation.x << "," << relativeLocation.y << ",";
    cv::Mat image = map.clone();
    pfm->visualizeParticles(image);
    /*std::cout << std::setprecision(9) << "SVO COORDS: " << lat << ", " << lon << "\n";
    std::cout << std::setprecision(9) << "GT  COORDS: " << metadata.latitude << ", " << metadata.longitude << "\n";*/
    if(!corners.empty()) {
        line(image, corners[0], corners[1], Scalar(0, 0, 255), 4);
        line(image, corners[1], corners[2], Scalar(0, 0, 255), 4);
        line(image, corners[2], corners[3], Scalar(0, 0, 255), 4);
        line(image, corners[3], corners[0], Scalar(0, 0, 255), 4);
        cv::Point2i arrowhead((corners[0].x + corners[1].x) / 2, (corners[0].y + corners[1].y) / 2);
        cv::Point2i center((corners[0].x + corners[2].x) / 2, (corners[0].y + corners[2].y) / 2);
        cv::arrowedLine(image, center, arrowhead, CV_RGB(255,0,0), 20);
    }
    visualizeGT(metadata.mapLocation, direction, image, 50, 3, CV_RGB(255, 255, 0));
    visualizeGT(prediction, direction, image, 50, 3, CV_RGB(255, 255, 255));
    cv::Mat mapDisplay = image(cv::Rect(
            prediction.x - 1000,
            prediction.y - 1000,
            3000,
            2000
    ));
    cv::Rect planeViewROI = cv::Rect(
            (mapDisplay.cols - 1) - planeView.cols,
            0,
            planeView.cols,
            planeView.rows
    );
    planeView.copyTo(mapDisplay(planeViewROI));
    cv::rectangle(mapDisplay, planeViewROI, Scalar(0,0,255));

    int fontFace = FONT_HERSHEY_COMPLEX_SMALL;
    double fontScale = 2;
    int thickness = 3;
    int textOffset = 50;

    cv::putText(mapDisplay, "Simulated view", cv::Point(planeViewROI.x + 10, planeViewROI.y + textOffset),
                fontFace, fontScale, Scalar::all(255), thickness, 8);


    if(!bestTransform.empty()) {
        std::cout << bestTransform << "\n";
        cv::Mat best = Utilities::extractWarpedMapPart(metadata.map, metadata.getImage().size(), bestTransform);
        auto bestParticleROI = cv::Rect(
                (mapDisplay.cols - 1) - best.cols,
                planeView.rows,
                best.cols,
                best.rows
        );
        best.copyTo(mapDisplay(bestParticleROI));
        cv::rectangle(mapDisplay, bestParticleROI, Scalar(0,0,255));
        cv::putText(mapDisplay, "Best particle view",
                    cv::Point(bestParticleROI.x + 10, bestParticleROI.y + textOffset),
                    fontFace, fontScale, Scalar::all(255), thickness, 8);
    }
    if(!bestView.empty()) {
        auto bestParticleROI = cv::Rect(
                (mapDisplay.cols - 1) - bestView.cols,
                planeView.rows,
                bestView.cols,
                bestView.rows
        );
        bestView.copyTo(mapDisplay(bestParticleROI));
        cv::rectangle(mapDisplay, bestParticleROI, Scalar(0,0,255));
        cv::putText(mapDisplay, "Best particle view",
                    cv::Point(bestParticleROI.x + 10, bestParticleROI.y + textOffset),
                    fontFace, fontScale, Scalar::all(255), thickness, 8);
    }
    double distance = sqrt(pow(metadata.mapLocation.x - prediction.x, 2) + pow(metadata.mapLocation.y - prediction.y, 2));
    // svoCurPosition contains latest location of the SVO
    double svoDistance = sqrt(pow(metadata.mapLocation.x - svoCurPosition.x, 2) +
                                      pow(metadata.mapLocation.y - svoCurPosition.y, 2));
    stringOutput << std::fixed << std::setprecision(2) << distance << "," << svoDistance;
    cv::putText(mapDisplay, "Location error = " + std::to_string(distance) + " m",
                cv::Point(10, textOffset), fontFace, fontScale, Scalar::all(255), thickness, 8);
    if(writeImageToDisk) {
        static int counter = 0;
        char integers[6];
        std::snprintf(integers, 6, "%05d", counter++);
        std::string filename = "preview_" + std::string(integers) + ".jpg";
        fs::path p(outputDirectory);
        cv::imwrite((p / filename).string(), mapDisplay);
    }
    if(displayImage) {
        cv::imshow("Map", mapDisplay);
        int key = cv::waitKey(10);
        // Break the cycle on ESC key
        return key != 27;
    } else {
        return true;
    }
}

void ParticleFilterWorkspace::visualizeGT(const cv::Point &loc, double yaw, cv::Mat &image, int radius, int thickness,
                                        const Scalar &color) {
    cv::circle(image, loc, radius, color, thickness);
    cv::line(
            image,
            loc,
            cv::Point(
                    static_cast<int>(loc.x + (4 * radius * sin(yaw))),
                    static_cast<int>(loc.y - (4 * radius * cos(yaw)))
            ),
            color,
            thickness
    );
}

void ParticleFilterWorkspace::updateScale(float hfov, float altitude, uint32_t imageWidth) {
    currentScale = (tan(hfov / 2.0f) * altitude) / ((float) imageWidth / 2.0f);
    pfm->setScale(currentScale * .9f, currentScale * 1.1f);
}

cv::Point ParticleFilterWorkspace::getMovementFromSvo(const MetadataEntry &metadata) {
    double lat, lon, h;
    // I had to negate both X and Y to achieve good combination
    svoCoordinates->Reverse(
            metadata.svoPose.getX(),
            metadata.svoPose.getY(),
            metadata.svoPose.getZ(),
            lat,
            lon,
            h
    );
    cv::Point curLoc = metadata.mapper->toPixels(lat, lon);
    cv::Point movement = curLoc - svoCurPosition;

    // Don't use direction from SVO, it may be misleading, just use the distance from odometry
    // and direction from compass which is way more reliable.
    float distance = static_cast<float>(std::sqrt(std::pow(movement.x, 2.f) + std::pow(movement.y, 2.f)));
    movement = cv::Point2f(
            static_cast<float>(std::sin(direction) * distance),
            static_cast<float>(-std::cos(direction) * distance)
    );

    svoCurPosition = curLoc;
    return movement;
}

void ParticleFilterWorkspace::setWriteImageToDisk(bool writeImageToDisk) {
    ParticleFilterWorkspace::writeImageToDisk = writeImageToDisk;
}

void ParticleFilterWorkspace::setOutputDirectory(const string &outputDirectory) {
    ParticleFilterWorkspace::outputDirectory = outputDirectory;
}

bool ParticleFilterWorkspace::isAffineMatching() const {
    return affineMatching;
}

void ParticleFilterWorkspace::setAffineMatching(bool affineMatching) {
    ParticleFilterWorkspace::affineMatching = affineMatching;
}

bool ParticleFilterWorkspace::isDisplayImage() const {
    return displayImage;
}

void ParticleFilterWorkspace::setDisplayImage(bool displayImage) {
    ParticleFilterWorkspace::displayImage = displayImage;
}

void ParticleFilterWorkspace::setCorrelationLowBound(float bound) {
    pfm->setLowBound(bound);
}

void ParticleFilterWorkspace::setConversionMethod(ParticleFastMatch::ConversionMode method) {
    pfm->conversionMode = method;
}

void ParticleFilterWorkspace::describe() const {
    std::cout << "Using conversion mode: " << (pfm->conversionMode == ParticleFastMatch::HPRELU ? "HPRELU\n" : "GLF\n");
    std::cout << "Conversion bound: " << pfm->getLowBound() << "\n";
}
