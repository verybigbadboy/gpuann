cmake_minimum_required(VERSION 2.8)
project(memorytest)

FIND_PACKAGE(CUDA)
include_directories(${CMAKE_SOURCE_DIR} ${INCLUDE_DIRECTORIES} /home/bad/projects/fann-code/src/include ${CMAKE_CURRENT_SOURCE_DIR})
link_directories(/home/bad/projects/fann-code/build/src/)

link_libraries(floatfann)
CUDA_ADD_EXECUTABLE(memoryTest
gpuann.cpp
base/gpuannDataCreator.cpp
base/neuralNetworkTypeCheck.cpp
test.cpp
testing/testing.cpp
kernels/straightforward/run.cu
kernels/computeMSE/run.cu
kernels/updateWeights/updateWeights.cu
kernels/backpropagateMSE/backpropagateMSErun.cu
kernels/updateSlopesBatch/updateSlopesBatch.cu
kernels/updateWeightsBatch/updateWeightsBatch.cu
kernels/updateWeightsIrpropm/updateWeightsIrpropm.cu
kernels/updateWeightsQuickprop/updateWeigthsQuickprop.cu
kernels/updateWeightsSarprop/updateWeightsSarprop.cu
)
