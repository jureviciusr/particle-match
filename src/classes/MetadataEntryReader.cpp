//
// Created by rokas on 17.11.30.
//

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include "MetadataEntryReader.hpp"

std::vector<std::string> MetadataEntryReader::parseString(const std::string &line) {
    std::vector<std::string> parts;
    boost::split(parts,line,boost::is_any_of(","));
    return parts;
}

std::map<std::string, std::string>
MetadataEntryReader::parseLine(const std::vector<std::string> &header, const std::string &line) {
    std::vector<std::string> values = parseString(line);
    std::map<std::string, std::string> valueMap;
    int i = 0;
    for(const auto& value : values) {
        valueMap.insert(std::pair<std::string, std::string>(
                header[i], value
        ));
        i++;
    }
    return valueMap;
}

bool MetadataEntryReader::openDirectory(const std::string &datasetDir) {
    datasetPath = datasetDir;
    in = std::ifstream(datasetDir + "/metadata.csv");
    if(in.is_open()) {
        std::string line;
        std::getline(in, line);
        header = parseString(line);
        return true;
    } else {
        return false;
    }
}

bool MetadataEntryReader::readNextEntry(MetadataEntry &metadataEntry) {
    // Reset the metadata
    metadataEntry = MetadataEntry();
    std::string line;
    if(std::getline(in, line)) {
        std::map<std::string, std::string> values = parseLine(header, line);
        fillMetadata(metadataEntry, values);
        return true;
    } else {
        return false;
    }
}

void MetadataEntryReader::fillMetadata(MetadataEntry &entry, std::map<std::string, std::string> &values) {
    entry.imageFileName = values["Filename"];
    entry.imageFileName.erase(
            remove( entry.imageFileName.begin(), entry.imageFileName.end(), '\"' ),
            entry.imageFileName.end()
    );
    entry.imageFullPath = datasetPath + "/" + entry.imageFileName;
    entry.latitude = std::atof(values["Latitude"].c_str());
    entry.longitude = std::atof(values["Longitude"].c_str());
    entry.altitude = std::atof(values["RelativeAltitude"].c_str());
    entry.imuOrientation = Quaternion(
            std::atof(values["ImuX"].c_str()),
            std::atof(values["ImuY"].c_str()),
            std::atof(values["ImuZ"].c_str()),
            std::atof(values["ImuW"].c_str())
    );
    entry.groundTruthPose = Vector3d(
            std::atof(values["PoseX"].c_str()),
            std::atof(values["PoseY"].c_str()),
            std::atof(values["PoseZ"].c_str())
    );
    entry.groundTruthOrientation = Quaternion(
            std::atof(values["OrientationX"].c_str()),
            std::atof(values["OrientationY"].c_str()),
            std::atof(values["OrientationZ"].c_str()),
            std::atof(values["OrientationW"].c_str())
    );
    entry.mapLocation = cv::Point(
            std::atoi(values["MapX"].c_str()),
            std::atoi(values["MapY"].c_str())
    );
    entry.svoPose = Vector3d(
            std::atof(values["SvoX"].c_str()),
            std::atof(values["SvoY"].c_str()),
            std::atof(values["SvoZ"].c_str())

    );
};
