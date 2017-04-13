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
        FAsT-Match/FAsT-Match/FAsTMatch.cpp
        FAsT-Match/FAsT-Match/MatchConfig.cpp
        FAsT-Match/FAsT-Match/MatchNet.cpp)

add_library(fastmatch SHARED ${SOURCES})
target_link_libraries(fastmatch ${OpenCV_LIBS} tbb)

add_executable(fastmatch-exec FAsT-Match/FAsT-Match/main.cpp)
target_link_libraries(fastmatch-exec fastmatch )

add_executable(fast-map-match src/map-match.cpp)
target_link_libraries(fast-map-match fastmatch ${Boost_LIBRARIES} ${map_library_LIBRARIES})