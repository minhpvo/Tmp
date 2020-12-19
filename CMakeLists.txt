cmake_minimum_required(VERSION 2.8)
project( EnRecon )

if (CMAKE_BUILD_TYPE STREQUAL Debug)
    set(BUILD_POSTFIX "_d")
else(CMAKE_BUILD_TYPE STREQUAL Debug)
    set(BUILD_POSTFIX "")
endif(CMAKE_BUILD_TYPE STREQUAL Debug)

option(OPENMP_ENABLED "Whether to enable OpenMP" ON)
option(LTO_ENABLED "Whether to enable link-time optimization" ON)

if(OPENMP_ENABLED)
    find_package(OpenMP QUIET)
endif()

find_package( OpenCV REQUIRED )

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
include_directories(${Boost_LIBRARY_DIR})

find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIRS})
link_libraries(${GLEW_LIBRARIES})
add_definitions(${GLEW_DEFINITIONS})

find_package(GLUT REQUIRED)
include_directories(${GLUT_INCLUDE_DIRS})
link_directories(${GLUT_LIBRARY_DIRS})
add_definitions(${GLUT_DEFINITIONS})

find_package(OpenGL REQUIRED)
include_directories(${OpenGL_INCLUDE_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})
add_definitions(${OpenGL_DEFINITIONS})


find_package(nlohmann_json 3.2.0 REQUIRED)


include_directories(ThirdParty/graph)
file(GLOB headers
    ThirdParty/graph/andres/*.hxx
    ThirdParty/graph/andres/ilp/*.hxx
    ThirdParty/graph/andres/graph/*.hxx
    ThirdParty/graph/andres/graph/multicut/*.hxx
    ThirdParty/graph/andres/graph/multicut-lifted/*.hxx
    ThirdParty/graph/andres/graph/twocut-lifted/*.hxx
    ThirdParty/graph/andres/graph/hdf5/*.hxx
    ThirdParty/graph/andres/graph/detail/*.hxx
)

IF(WIN32)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /bigobj /openmp") 
	
	INCLUDE_DIRECTORIES(ThirdParty/vlfeat)
    link_directories(ThirdParty/vlfeat/bin/win64)
	
	add_definitions(-DUSESIFTGPU)
	add_definitions(-DNOMINMAX)
	add_definitions(-DGLOG_NO_ABBREVIATED_SEVERITIES)	
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)	
	
	include_directories(Audio)
	include_directories(Geometry)
	include_directories(ImgPro)
	include_directories(TrackMatch)
	include_directories(Ulti)
	include_directories(Drivers)
	include_directories(Vis)

	add_subdirectory(Audio)
	add_subdirectory(Geometry)
	add_subdirectory(ImgPro)
	add_subdirectory(TrackMatch)
	add_subdirectory(Ulti)
	add_subdirectory(Drivers)
	add_subdirectory(Vis)
	
	add_executable( EnRecon_Exp main_exp.cpp DataStructure.cpp)
    add_executable( EnRecon_Calib main_calib.cpp DataStructure.cpp)
	add_executable( EnRecon_Human main_human.cpp DataStructure.cpp)	
	
	target_link_libraries( EnRecon_Exp Audio Geometry ImgPro TrackMatch Ulti Drivers Vis nlohmann_json::nlohmann_json ${OpenCV_LIBS}  ${CERES_LIBRARIES} ${Boost_LIBRARIES} ${OPENGL_LIBRARIES} ${GLEW_LIBRARY} ${GLUT_LIBRARY} vl.lib)

	target_link_libraries( EnRecon_Calib Audio Geometry ImgPro TrackMatch Ulti Drivers Vis nlohmann_json::nlohmann_json ${OpenCV_LIBS}  ${CERES_LIBRARIES} ${Boost_LIBRARIES} ${OPENGL_LIBRARIES} ${GLEW_LIBRARY} ${GLUT_LIBRARY} vl.lib)
	
	target_link_libraries( EnRecon_Human Audio Geometry ImgPro TrackMatch Ulti Drivers Vis nlohmann_json::nlohmann_json ${OpenCV_LIBS}  ${CERES_LIBRARIES} ${Boost_LIBRARIES} ${OPENGL_LIBRARIES} ${GLEW_LIBRARY} ${GLUT_LIBRARY} vl.lib)
		
	set_target_properties(Audio PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
	set_target_properties(Audio PROPERTIES OUTPUT_NAME Audio${BUILD_POSTFIX})
	set_target_properties(Geometry PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
	set_target_properties(Geometry PROPERTIES OUTPUT_NAME Geometry${BUILD_POSTFIX})
	set_target_properties(ImgPro PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
	set_target_properties(ImgPro PROPERTIES OUTPUT_NAME ImgPro${BUILD_POSTFIX})
	set_target_properties(TrackMatch PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
	set_target_properties(TrackMatch PROPERTIES OUTPUT_NAME TrackMatch${BUILD_POSTFIX})
	set_target_properties(Ulti PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
	set_target_properties(Ulti PROPERTIES OUTPUT_NAME Ulti${BUILD_POSTFIX})
	set_target_properties(Drivers PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
	set_target_properties(Drivers PROPERTIES OUTPUT_NAME Drivers${BUILD_POSTFIX})
	set_target_properties(Vis PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
	set_target_properties(Vis PROPERTIES OUTPUT_NAME Vis${BUILD_POSTFIX})
		
elseif(UNIX) 
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -fpermissive -fopenmp -std=c++11 -O4")
	
    INCLUDE_DIRECTORIES(~/Research/DevSoft/EnRecon/ThirdParty/vlfeat)
    link_directories(~/Research/DevSoft/EnRecon/ThirdParty/vlfeat/bin/glnxa64)
	
    add_definitions(-DUSESIFTGPU)
	
	add_executable( EnRecon_Exp 	main_exp.cpp
                            DataStructure.cpp
							Audio/AudioPro.cpp
							ImgPro/ImagePro.cpp
							Geometry/Geometry1.cpp
							Geometry/Geometry2.cpp
							Geometry/modelest.cpp
							TrackMatch/FeatureEst.cpp
							TrackMatch/MatchingTracking.cpp
							Ulti/MiscAlgo.cpp
							Ulti/DataIO.cpp
							Ulti/MathUlti.cpp
							Ulti/GeneralUlti.cpp
							Drivers/Drivers.cpp
							Vis/Visualization.cpp)
							
	add_executable( EnRecon_Calib 	main_calib.cpp
                            DataStructure.cpp
							Audio/AudioPro.cpp
							ImgPro/ImagePro.cpp
							Geometry/Geometry1.cpp
							Geometry/Geometry2.cpp
							Geometry/modelest.cpp
							TrackMatch/FeatureEst.cpp
							TrackMatch/MatchingTracking.cpp
							Ulti/MiscAlgo.cpp
							Ulti/DataIO.cpp
							Ulti/MathUlti.cpp
							Ulti/GeneralUlti.cpp
							Drivers/Drivers.cpp
							Vis/Visualization.cpp)

	add_executable( EnRecon_Human 	main_human.cpp
                            DataStructure.cpp
							Audio/AudioPro.cpp
							ImgPro/ImagePro.cpp
							Geometry/Geometry1.cpp
							Geometry/Geometry2.cpp
							Geometry/modelest.cpp
							TrackMatch/FeatureEst.cpp
							TrackMatch/MatchingTracking.cpp
							Ulti/MiscAlgo.cpp
							Ulti/DataIO.cpp
							Ulti/MathUlti.cpp
							Ulti/GeneralUlti.cpp
							Drivers/Drivers.cpp
							Vis/Visualization.cpp)		
							
	target_link_libraries( EnRecon_Exp ${OpenCV_LIBS}  ${CERES_LIBRARIES} ${Boost_LIBRARIES} ${OPENGL_LIBRARIES} ${GLEW_LIBRARY} ${GLUT_LIBRARY} libvl.so -ldl)
	target_link_libraries( EnRecon_Calib ${OpenCV_LIBS}  ${CERES_LIBRARIES} ${Boost_LIBRARIES} ${OPENGL_LIBRARIES} ${GLEW_LIBRARY} ${GLUT_LIBRARY} libvl.so -ldl)
	target_link_libraries( EnRecon_Human ${OpenCV_LIBS}  ${CERES_LIBRARIES} ${Boost_LIBRARIES} ${OPENGL_LIBRARIES} ${GLEW_LIBRARY} ${GLUT_LIBRARY} libvl.so -ldl)
ENDIF()
