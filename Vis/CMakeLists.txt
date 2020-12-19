add_library(Vis Visualization.cpp)


target_include_directories(Vis PRIVATE ${OpenCV_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})
target_link_libraries(Vis PRIVATE nlohmann_json::nlohmann_json ${OpenCV_LIBS} ${CERES_LIBRARIES})

set_target_properties(Vis PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set_target_properties(Vis PROPERTIES OUTPUT_NAME Vis${BUILD_POSTFIX})
