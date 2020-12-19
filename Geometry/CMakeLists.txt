add_library(Geometry Geometry1.cpp Geometry2.cpp)

target_include_directories(Geometry PRIVATE ${OpenCV_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})
target_link_libraries(Geometry PRIVATE ${OpenCV_LIBS} ${CERES_LIBRARIES})


set_target_properties(Geometry PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set_target_properties(Geometry PROPERTIES OUTPUT_NAME Geometry${BUILD_POSTFIX})
