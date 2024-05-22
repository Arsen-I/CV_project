cmake_minimum_required(VERSION 2.8)
project(CVProject)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(project main.cpp 
	headers/detection/detection.h headers/detection/detection.cpp headers/detection/highLevelFunc.h headers/detection/highLevelFunc.cpp  headers/detection/lowLevelFunc.h headers/detection/lowLevelFunc.cpp 
	headers/beforeClassification/beforeClassification.h headers/beforeClassification/beforeClassification.cpp 
	headers/afterClassification/afterClassification.h headers/afterClassification/afterClassification.cpp
	headers/compare/compare.h headers/compare/compare.cpp)

target_link_libraries(project ${OpenCV_LIBS})
