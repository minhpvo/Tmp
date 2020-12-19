add_library(Audio AudioPro.cpp)

target_include_directories(Audio PRIVATE ${OpenCV_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})
target_link_libraries(Audio PRIVATE ${OpenCV_LIBS} ${CERES_LIBRARIES})


set_target_properties(Audio PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set_target_properties(Audio PROPERTIES OUTPUT_NAME Audio${BUILD_POSTFIX})


