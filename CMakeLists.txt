cmake_minimum_required(VERSION 3.0)

include_directories(FAsT-Match)

FIND_PACKAGE( Boost 1.40 COMPONENTS system program_options filesystem thread REQUIRED )
find_package( OpenCV REQUIRED )
find_package( map_library REQUIRED )

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${map_library_INCLUDE_DIR})

link_directories(${map_library_LIBRARY_DIR})
message(STATUS ${map_library_LIBRARY_DIR})

add_definitions(-std=c++14)

set(SOURCES
        FAsT-Match/src/FastMatcherThread.cpp
        FAsT-Match/src/FastMatch.cpp
        FAsT-Match/src/ConfigVisualizer.cpp
        FAsT-Match/src/ParticleFilterConfigExpander.cpp
        FAsT-Match/src/ConfigExpanderBase.cpp
        FAsT-Match/src/GridConfigExpander.cpp
        FAsT-Match/FAsT-Match/MatchConfig.cpp
        FAsT-Match/FAsT-Match/MatchNet.cpp)

add_library(fastmatch SHARED ${SOURCES})
target_link_libraries(fastmatch ${OpenCV_LIBS} ${Boost_LIBRARIES} tbb)

add_executable(fastmatch-exec FAsT-Match/FAsT-Match/main.cpp)
target_link_libraries(fastmatch-exec fastmatch )

add_executable(fastmatch-test FAsT-Match/exec/match-test.cpp)
target_link_libraries(fastmatch-test fastmatch )

add_executable(fast-map-match src/map-match.cpp)
target_link_libraries(fast-map-match fastmatch ${Boost_LIBRARIES} ${map_library_LIBRARIES})

SET(fastmatch_LIBRARIES AirvisionSensorComm fastmatch ${OpenCV_LIBS} tbb)
SET(fastmatch_LIBRARY_DIR ${PROJECT_BINARY_DIR} )
SET(fastmatch_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/FAsT-Match")
CONFIGURE_FILE( ${CMAKE_CURRENT_SOURCE_DIR}/fastmatchConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/fastmatchConfig.cmake @ONLY IMMEDIATE )
export( PACKAGE fastmatch )

