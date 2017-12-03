//
// Created by rokas on 17.11.30.
//

#pragma once

#include <boost/token_functions.hpp>
#include <map>
#include <boost/tokenizer.hpp>
#include <fstream>
#include "MetadataEntry.hpp"
#include "Map.hpp"


class MetadataEntryReader {
private:
    static std::vector<std::string> parseString(const std::string& line);

    static std::map<std::string, std::string> parseLine(
            const std::vector<std::string>& header,
            const std::string& line
    );

    std::ifstream in;

    std::vector<std::string> header;

    std::string datasetPath;

    void fillMetadata(MetadataEntry& entry, std::map<std::string, std::string>& values);

    MapPtr map = nullptr;

public:

    bool openDirectory(const std::string &datasetDir);

    bool readNextEntry(MetadataEntry& metadataEntry);

    void setMap(const std::string& mapFile, const std::string& mapDescription);

    const MapPtr &getMap() const;

};


