//
// Created by rokas on 17.12.4.
//

#include <src/Utilities.hpp>
#include "ParticleFilterWorkspace.hpp"

void ParticleFilterWorkspace::initialize(const MetadataEntry &metadata) {
    direction = metadata.imuOrientation.toRPY().getZ();
    Particle::setDirection(direction);
    initialPosition = metadata.mapLocation;
    svoCoordinates = std::make_shared<GeographicLib::LocalCartesian>(
            metadata.latitude,
            metadata.longitude,
            metadata.altitude
    );
    pfm = std::make_shared<ParticleFastMatch>(
            initialPosition, // startLocation
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
    cv::namedWindow("Map", CV_WINDOW_NORMAL);
    //cv::namedWindow("BestTransform", CV_WINDOW_NORMAL);
}

void ParticleFilterWorkspace::update(const MetadataEntry &metadata) {
    cv::Point movement = getMovementFromSvo(metadata);
    updateScale(1.0, static_cast<float>(metadata.altitude), 640);
    direction = metadata.imuOrientation.toRPY().getZ();
    Particle::setDirection(direction);
    cv::Mat templ = metadata.getImageColored();
    pfm->setTemplate(templ);
    corners = pfm->filterParticles(movement, bestTransform);
    cv::Mat bestView = Utilities::extractMapPart(metadata.map, templ.size(), bestTransform);
    cv::Point2i prediction = pfm->getPredictedLocation();
}

bool ParticleFilterWorkspace::preview(const MetadataEntry &metadata, cv::Mat planeView) const {
    cv::Point2i prediction = pfm->getPredictedLocation();
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
        cv::Mat best = Utilities::extractMapPart(metadata.map, metadata.getImage().size(), bestTransform);
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

    double distance = sqrt(pow(metadata.mapLocation.x - prediction.x, 2) + pow(metadata.mapLocation.y - prediction.y, 2));
    std::cout << "Location error = " << distance << "\n";
    cv::putText(mapDisplay, "Location error = " + std::to_string(distance * currentScale) + " m",
                cv::Point(10, textOffset), fontFace, fontScale, Scalar::all(255), thickness, 8);

    cv::imshow("Map", mapDisplay);
    static int counter = 0;
    cv::imwrite("images/test-1/preview_" + std::to_string(counter++) + ".jpg", mapDisplay);
    int key = cv::waitKey(10);
    // Break the cycle on ESC key
    return key != 27;
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
    // I had to reverse both X and Y to achieve good combination
    svoCoordinates->Reverse(
            -metadata.svoPose.getX(),
            -metadata.svoPose.getY(),
            metadata.svoPose.getZ(),
            lat,
            lon,
            h
    );
    cv::Point curLoc = metadata.mapper->toPixels(lat, lon);
    cv::Point movement = curLoc - initialPosition;
    initialPosition = curLoc;
    return movement;
}
