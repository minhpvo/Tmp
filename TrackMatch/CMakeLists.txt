add_library(TrackMatch FeatureEst.cpp MatchingTracking.cpp)

target_include_directories(TrackMatch PRIVATE ${OpenCV_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})
target_link_libraries(TrackMatch PRIVATE ${OpenCV_LIBS} ${CERES_LIBRARIES})

set_target_properties(TrackMatch PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set_target_properties(TrackMatch PROPERTIES OUTPUT_NAME TrackMatch${BUILD_POSTFIX})

