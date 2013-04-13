cmake_minimum_required(VERSION 2.8)
project(memorytest)

FIND_PACKAGE(CUDA)
include_directories(${CMAKE_SOURCE_DIR} ${INCLUDE_DIRECTORIES} /home/bad/projects/fann-code/src/include ${CMAKE_CURRENT_SOURCE_DIR})
link_directories(/home/bad/projects/fann-code/build/src/)

link_libraries(floatfann)
CUDA_ADD_EXECUTABLE(memoryTest
gpuann.cpp
gpuannTrain.cpp
base/gpuannDataCreator.cpp
base/neuralNetworkTypeCheck.cpp
test.cpp
testing/testing.cpp
testing/train.cpp
testing/util.cpp
kernels/backpropagateMSE/backpropagateMSErun.cu
kernels/computeMSE/run.cu
kernels/mergeSlopes/mergeSlopes.cu
kernels/straightforward/straightforward.cpp
kernels/straightforward/straightforward.cu
kernels/updateSlopesBatch/updateSlopesBatch.cu
kernels/updateWeights/updateWeights.cu
kernels/updateWeightsBatch/updateWeightsBatch.cu
kernels/updateWeightsIrpropm/updateWeightsIrpropm.cu
kernels/updateWeightsQuickprop/updateWeigthsQuickprop.cu
kernels/updateWeightsSarprop/updateWeightsSarprop.cu
)
