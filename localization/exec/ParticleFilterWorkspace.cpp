//
// Created by rokas on 17.12.4.
//

#include <src/Utilities.hpp>
#include <boost/filesystem/path.hpp>
#include <Eigen/Eigen>
#include <opencv/cxeigen.hpp>
#include "ParticleFilterWorkspace.hpp"

namespace fs = boost::filesystem;

void ParticleFilterWorkspace::initialize(const MetadataEntry &metadata) {
    std::cout << "Initializing...";
    std::cout.flush();
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
    pfm->setImage(metadata.map);
    map = metadata.map;
    updateScale(1.0, static_cast<float>(metadata.altitude), templ.cols);
    if(displayImage) {
        cv::namedWindow("Map", cv::WINDOW_NORMAL);
        cv::waitKey(10);
    }
    startLocation = pfm->getPredictedLocation();
    std::cout << " done!" << std::endl;
}

void ParticleFilterWorkspace::update(const MetadataEntry &metadata) {
    cv::Point movement = getMovementFromSvo(metadata);
    updateScale(1.0, static_cast<float>(metadata.altitude), 640);
    direction = metadata.imuOrientation.toRPY().getZ();
    Particle::setDirection(direction);
    cv::Mat templ = metadata.getImageColored();
    pfm->setTemplate(templ);
    if(!affineMatching) {
        corners = pfm->filterParticles(movement, bestTransform);
        bestView = pfm->getBestParticleView(metadata.map);
    } else {
#ifdef USE_CV_GPU
        corners = pfm->filterParticlesAffine(movement, bestTransform);
#else
        std::cerr << "Affine particle matching is available with GPU support only at this moment" << std::endl;
        exit(1);
#endif
    }
}

bool ParticleFilterWorkspace::preview(const MetadataEntry &metadata, cv::Mat planeView, std::stringstream& stringOutput)
const {
    cv::Point2i prediction = pfm->getPredictedLocation();
    cv::Point2i relativeLocation = prediction - startLocation;
    stringOutput << pfm->particleCount() << ",";
    stringOutput << relativeLocation.x << "," << relativeLocation.y << ",";
    cv::Point2i offset = cv::Point2i(
            -(prediction.x - 1000),
            -(prediction.y - 1000)
    );
    cv::Mat mapDisplay = map(cv::Rect(
            prediction.x - 1000,
            prediction.y - 1000,
            3000,
            2000
    )).clone();
    pfm->visualizeParticles(mapDisplay, offset);
    if(!corners.empty()) {
        std::vector<cv::Point> newCorners = {
                corners[0] + offset,
                corners[1] + offset,
                corners[2] + offset,
                corners[3] + offset
        };
        line(mapDisplay, newCorners[0], newCorners[1], Scalar(0, 0, 255), 4);
        line(mapDisplay, newCorners[1], newCorners[2], Scalar(0, 0, 255), 4);
        line(mapDisplay, newCorners[2], newCorners[3], Scalar(0, 0, 255), 4);
        line(mapDisplay, newCorners[3], newCorners[0], Scalar(0, 0, 255), 4);
        cv::Point2i arrowhead((newCorners[0].x + newCorners[1].x) / 2, (newCorners[0].y + newCorners[1].y) / 2);
        cv::Point2i center((newCorners[0].x + newCorners[2].x) / 2, (newCorners[0].y + newCorners[2].y) / 2);
        cv::arrowedLine(mapDisplay, center, arrowhead, CV_RGB(255,0,0), 20);
    }
    visualizeGT(metadata.mapLocation + offset, direction, mapDisplay, 50, 3, CV_RGB(255, 255, 0));
    visualizeGT(prediction + offset, direction, mapDisplay, 50, 3, CV_RGB(255, 255, 255));
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
        if (bestView.channels() == 1) {
            cv::cvtColor(bestView, mapDisplay(bestParticleROI), cv::COLOR_GRAY2BGR);
        } else if (bestView.channels() == 3) {
            bestView.copyTo(mapDisplay(bestParticleROI));
        }
        cv::rectangle(mapDisplay, bestParticleROI, Scalar(0,0,255));
        cv::putText(mapDisplay, "Best particle " + std::to_string(pfm->getParticles().back().getCorrelation()),
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
        cv::Mat preview;
        cv::resize(mapDisplay, preview, cv::Size(1200, 800));
        cv::imshow("Map", preview);
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

cv::Point ParticleFilterWorkspace::getMovementFromSvo2(const MetadataEntry &metadata) {
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
    cv::Mat cameraRot = Utilities::eulerAnglesToRotationMatrix(cv::Point3d(-M_PI_2, 0, 0));
    cv::Mat zeroLookVector = (cv::Mat_<double>(3, 1) << 0.0, 1.0, 0.0);
    Eigen::Quaterniond q(
            metadata.imuOrientation.getW(),
            metadata.imuOrientation.getX(),
            metadata.imuOrientation.getY(),
            metadata.imuOrientation.getZ()
            );
    q.normalize();
    cv::Mat quatTransform(3, 3, CV_64FC1);//
    cv::eigen2cv(q.toRotationMatrix(), quatTransform);
    cv::Mat cameraLookVec = (quatTransform * cameraRot) * zeroLookVector;
    // Z is used from barometer data
    cv::Point3d planePos = cv::Point3d(curLoc.x, curLoc.y, metadata.altitude);
    cv::Point3d lookVector = cv::Point3d(
            planePos.x + cameraLookVec.at<double>(0, 0),
            planePos.y + cameraLookVec.at<double>(1, 0),
            planePos.z + cameraLookVec.at<double>(2, 0)
    );
    cv::Point3d isection = Utilities::intersectPlaneV3(planePos, lookVector, cv::Point3d(0, 0, 0), cv::Point3d(0, 0, 1));
    curLoc = cv::Point(static_cast<int>(isection.x), static_cast<int>(isection.y));
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
    std::cout << "Using conversion mode: " << pfm->conversionModeString() << "\n";
    std::cout << "Conversion bound: " << pfm->getLowBound() << "\n";
}

const Particles &ParticleFilterWorkspace::getParticles() const {
    return pfm->getParticles();
}
