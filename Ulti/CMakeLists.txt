add_library(Ulti DataIO.cpp GeneralUlti.cpp MathUlti.cpp MiscAlgo.cpp)


target_include_directories(Ulti PRIVATE ${OpenCV_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})
target_link_libraries(Ulti PRIVATE ${OpenCV_LIBS} ${CERES_LIBRARIES})

set_target_properties(Ulti PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set_target_properties(Ulti PROPERTIES OUTPUT_NAME Ulti${BUILD_POSTFIX})
