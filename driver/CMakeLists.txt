cmake_minimum_required(VERSION 2.8)
project(driver)

FIND_PACKAGE(CUDA)
include_directories(${CMAKE_SOURCE_DIR} ${INCLUDE_DIRECTORIES} /home/bad/projects/fann-code/src/include ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../gpuann/)
link_directories(/home/bad/projects/fann-code/build/src/)

link_libraries(floatfann gpuann)
CUDA_ADD_EXECUTABLE(driver
main.cpp
testing/testing.cpp
testing/train.cpp
testing/trainSpeed.cpp
testing/run.cpp
common/util.cpp
common/fannHelper.cpp
)
