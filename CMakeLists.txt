cmake_minimum_required(VERSION 3.0)

project(particle-match)

include_directories(localization)
include_directories(dataset_reader/include)

set(CMAKE_CXX_STANDARD 14)
add_definitions(-fext-numeric-literals)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")

FIND_PACKAGE( Boost 1.40 COMPONENTS system program_options filesystem thread REQUIRED )
find_package( OpenCV REQUIRED )
find_package( GeographicLib REQUIRED )
find_package( Eigen3 REQUIRED )
find_package(GDAL REQUIRED)
find_package(CURL REQUIRED)
include_directories(${CURL_INCLUDE_DIR})

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${GeographicLib_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})


set(SOURCES
        localization/src/FastMatcherThread.cpp
        localization/src/FastMatch.cpp
        localization/src/ParticleFastMatch.cpp
        localization/src/Utilities.cpp
        localization/src/pf/Particle.cpp
        localization/src/pf/Particles.cpp
        localization/src/AffineTransformation.cpp
        localization/src/ConfigVisualizer.cpp
        localization/src/ParticleFilterConfigExpander.cpp
        localization/src/ConfigExpanderBase.cpp
        localization/src/GridConfigExpander.cpp
        localization/FAsT-Match/MatchConfig.cpp
        localization/FAsT-Match/MatchNet.cpp)

include_directories(include)
add_library(datasetreader SHARED
        dataset_reader/include/fastmatch-dataset/MetadataEntry.hpp
        dataset_reader/include/fastmatch-dataset/MetadataEntryReader.hpp
        dataset_reader/include/fastmatch-dataset/Quaternion.hpp
        dataset_reader/include/fastmatch-dataset/Vector3d.hpp
        dataset_reader/include/fastmatch-dataset/Map.hpp
        dataset_reader/include/fastmatch-dataset/GeotiffMap.hpp
        dataset_reader/src/classes/MetadataEntryReader.cpp
        dataset_reader/src/classes/MetadataEntry.cpp
        dataset_reader/src/classes/Quaternion.cpp
        dataset_reader/src/classes/Vector3d.cpp
        dataset_reader/src/classes/Map.cpp
        dataset_reader/src/classes/GeotiffMap.cpp
        )
target_link_libraries(datasetreader ${OpenCV_LIBS} ${Boost_LIBRARIES} ${GeographicLib_LIBRARY} ${GDAL_LIBRARY})

add_library(fastmatch SHARED ${SOURCES})
target_link_libraries(fastmatch ${OpenCV_LIBS} ${Boost_LIBRARIES} tbb)

#add_executable(map-masker FAsT-Match/exec/map-masker.cpp FAsT-Match/exec/xml-parser.h
#                            FAsT-Match/exec/PascalVocWriter.cpp)
#target_link_libraries(map-masker datasetreader ${CURL_LIBRARIES}
#        ${OpenCV_LIBS} ${Boost_LIBRARIES} ${GeographicLib_LIBRARY} ${GDAL_LIBRARY})

add_executable(dataset-match localization/exec/ParticleFilterWorkspace.cpp localization/exec/dataset-test.cpp)
target_link_libraries(dataset-match fastmatch ${Boost_LIBRARIES} datasetreader)

#add_executable(test-slopes FAsT-Match/exec/test-slopes.cpp)
#target_link_libraries(test-slopes ${Boost_LIBRARIES})

SET(fastmatch_LIBRARIES AirvisionSensorComm fastmatch ${OpenCV_LIBS} tbb)
SET(fastmatch_LIBRARY_DIR ${PROJECT_BINARY_DIR} )
SET(fastmatch_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/localization")
CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/fastmatchConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/fastmatchConfig.cmake @ONLY IMMEDIATE )
export( PACKAGE fastmatch )

