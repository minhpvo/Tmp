add_library(ImgPro ImagePro.cpp)


target_include_directories(ImgPro PRIVATE ${OpenCV_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})
target_link_libraries(ImgPro PRIVATE ${OpenCV_LIBS} ${CERES_LIBRARIES})

set_target_properties(ImgPro PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set_target_properties(ImgPro PROPERTIES OUTPUT_NAME ImgPro${BUILD_POSTFIX})
