cmake_minimum_required(VERSION 3.0)

include_directories(FAsT-Match)

set(CMAKE_CXX_STANDARD 14)
add_definitions(-fext-numeric-literals)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")

FIND_PACKAGE( Boost 1.40 COMPONENTS system program_options filesystem thread REQUIRED )
find_package( OpenCV REQUIRED )
#find_package( map_library REQUIRED )
find_package( GeographicLib REQUIRED )
find_package( Eigen3 REQUIRED )
find_package(GDAL REQUIRED)
find_package(CURL REQUIRED)
include_directories(${CURL_INCLUDE_DIR})

#add_subdirectory(map-library)

include_directories(${OpenCV_INCLUDE_DIRS})
#include_directories(map-library/include)
include_directories(${GeographicLib_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})

#link_directories(${map_library_LIBRARY_DIR})
#message(STATUS ${map_library_LIBRARY_DIR})


set(SOURCES
        FAsT-Match/src/FastMatcherThread.cpp
        FAsT-Match/src/FastMatch.cpp
        FAsT-Match/src/ParticleFastMatch.cpp
        FAsT-Match/src/Utilities.cpp
        FAsT-Match/src/pf/Particle.cpp
        FAsT-Match/src/pf/Particles.cpp
        FAsT-Match/src/AffineTransformation.cpp
        FAsT-Match/src/ConfigVisualizer.cpp
        FAsT-Match/src/ParticleFilterConfigExpander.cpp
        FAsT-Match/src/ConfigExpanderBase.cpp
        FAsT-Match/src/GridConfigExpander.cpp
        FAsT-Match/FAsT-Match/MatchConfig.cpp
        FAsT-Match/FAsT-Match/MatchNet.cpp)

include_directories(include)
add_library(datasetreader SHARED
        include/fastmatch-dataset/MetadataEntry.hpp
        include/fastmatch-dataset/MetadataEntryReader.hpp
        include/fastmatch-dataset/Quaternion.hpp
        include/fastmatch-dataset/Vector3d.hpp
        include/fastmatch-dataset/Map.hpp
        include/fastmatch-dataset/GeotiffMap.hpp
        src/classes/MetadataEntryReader.cpp
        src/classes/MetadataEntry.cpp
        src/classes/Quaternion.cpp
        src/classes/Vector3d.cpp
        src/classes/Map.cpp
        src/classes/GeotiffMap.cpp
        )
target_link_libraries(datasetreader ${OpenCV_LIBS} ${Boost_LIBRARIES} ${GeographicLib_LIBRARY} ${GDAL_LIBRARY})

add_library(fastmatch SHARED ${SOURCES})
target_link_libraries(fastmatch ${OpenCV_LIBS} ${Boost_LIBRARIES} tbb)

#add_executable(fastmatch-exec FAsT-Match/FAsT-Match/main.cpp)
#target_link_libraries(fastmatch-exec fastmatch )

#add_executable(fastmatch-test FAsT-Match/exec/match-test.cpp)
#target_link_libraries(fastmatch-test fastmatch )

#add_executable(particle-fastmatch-test FAsT-Match/exec/particle-match-test.cpp)
#target_link_libraries(particle-fastmatch-test fastmatch ${Boost_LIBRARIES})

add_executable(map-masker FAsT-Match/exec/map-masker.cpp FAsT-Match/exec/xml-parser.h
                            FAsT-Match/exec/PascalVocWriter.cpp)
target_link_libraries(map-masker datasetreader ${CURL_LIBRARIES}
        ${OpenCV_LIBS} ${Boost_LIBRARIES} ${GeographicLib_LIBRARY} ${GDAL_LIBRARY})

add_executable(dataset-match FAsT-Match/exec/ParticleFilterWorkspace.cpp FAsT-Match/exec/dataset-test.cpp)
target_link_libraries(dataset-match fastmatch ${Boost_LIBRARIES} datasetreader)

add_executable(test-slopes FAsT-Match/exec/test-slopes.cpp)
target_link_libraries(test-slopes ${Boost_LIBRARIES})

#add_executable(fast-map-match src/map-match.cpp)
#target_link_libraries(fast-map-match fastmatch ${Boost_LIBRARIES} map_library)

#add_executable(map-test FAsT-Match/exec/map-tests.cpp)
#target_link_libraries(map-test fastmatch ${Boost_LIBRARIES} datasetreader)

#add_executable(run-dataset src/run-dataset.cpp)
#target_link_libraries(run-dataset fastmatch datasetreader ${Boost_LIBRARIES})

SET(fastmatch_LIBRARIES AirvisionSensorComm fastmatch ${OpenCV_LIBS} tbb)
SET(fastmatch_LIBRARY_DIR ${PROJECT_BINARY_DIR} )
SET(fastmatch_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/FAsT-Match")
CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/fastmatchConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/fastmatchConfig.cmake @ONLY IMMEDIATE )
export( PACKAGE fastmatch )

