cmake_minimum_required(VERSION 2.8)
project(memorytest)

FIND_PACKAGE(CUDA)
include_directories(${CMAKE_SOURCE_DIR} ${INCLUDE_DIRECTORIES} /home/bad/projects/fann-code/src/include)
link_directories(/home/bad/projects/fann-code/build/src/)

link_libraries(floatfann)
CUDA_ADD_EXECUTABLE(memoryTest cuda/neuronsRunKernel.cpp fannCuda.cu cuda/gpuDataCreator.cpp cuda/neuralNetworkTypeCheck.cpp main.cu test.cpp)


add_subdirectory(cuda)