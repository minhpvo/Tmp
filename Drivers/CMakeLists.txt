add_library(Drivers Drivers.cpp)

target_include_directories(Drivers PRIVATE ${OpenCV_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})
target_link_libraries(Drivers PRIVATE nlohmann_json::nlohmann_json ${OpenCV_LIBS} ${CERES_LIBRARIES})

set_target_properties(Drivers PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set_target_properties(Drivers PROPERTIES OUTPUT_NAME Drivers${BUILD_POSTFIX})

