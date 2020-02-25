#! /bin/bash

# The data directory
mkdir -p dataset

# Download maps if necessary
MAPS="dataset/urban_maps.zip"
if [ ! -z "$MAPS" ]; then
  wget https://zenodo.org/record/1211730/files/urban_maps.zip?download=1 -O $MAPS
  unzip $MAPS -d dataset/
fi

# Download Flight data if necessary
# Flight data, one of: UL-200, UL-300, UR-200, UR-300, UC-200, UC-300
FLIGHT=UL-200
FLIGHT_FILE="dataset/$FLIGHT.zip"
if [ ! -z "$FLIGHT_FILE" ]; then
  wget https://zenodo.org/record/1211730/files/$FLIGHT.zip?download=1 -O $FLIGHT_FILE
  unzip $FLIGHT_FILE -d dataset/
fi
