cmake_minimum_required(VERSION 2.8)
project(gpuann)

FIND_PACKAGE(CUDA)
include_directories(${CMAKE_SOURCE_DIR} ${INCLUDE_DIRECTORIES} /home/bad/projects/fann-code/src/include ${CMAKE_CURRENT_SOURCE_DIR})

CUDA_ADD_LIBRARY(gpuann
gpuann.cpp
gpuannTrain.cpp
gpuannParallelTrain.cpp
base/gpuannDataCreator.cpp
base/neuralNetworkTypeCheck.cpp
kernels/backpropagateMSE/backpropagateMSErun.cu
kernels/computeMSE/run.cu
kernels/d2dMemcpy/d2dMemcpy.cu
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
