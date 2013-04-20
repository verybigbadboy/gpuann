
#include <kernels/updateSlopesBatch/updateSlopesBatch.h>
#include <fann.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <common/math.h>
#include <string>

__global__ void gpuann_fann_update_slopes_batch_multineuron_gpu_kernel(unsigned int prevNeuronsCount,
                                                                       unsigned int neuronsCount,
                                                                       fann_type   *trainErrors,
                                                                       fann_type   *neuronSlopes,
                                                                       fann_type   *prevValue,
                                                                       unsigned int totalNeuronsCount,
                                                                       unsigned int totalWeightsCount)
{
  unsigned int tid                  = threadIdx.x;
  unsigned int instance             = blockIdx.y;
  unsigned int threadCount          = blockDim.x;
  unsigned int neuronCountPerKernel = threadCount / prevNeuronsCount;
  unsigned int neuronIndex          = blockIdx.x * neuronCountPerKernel + tid / prevNeuronsCount;
  unsigned int actualPrevNeuron     = tid % prevNeuronsCount;
  unsigned int prevLayerNeuronIndex = actualPrevNeuron + instance * totalNeuronsCount;
  unsigned int slopesIndex          = actualPrevNeuron + prevNeuronsCount * neuronIndex + instance * totalWeightsCount;
  unsigned int neuronIndexInstanced = neuronIndex + instance * totalNeuronsCount;

  if(neuronIndex < (neuronsCount - 1)) //due to bias
  {
    fann_type error = trainErrors[neuronIndexInstanced];
    if(tid < prevNeuronsCount * neuronCountPerKernel)
      neuronSlopes[slopesIndex] += error * prevValue[prevLayerNeuronIndex];
  }
}

//neuronSlopes actually points to prev layer values
template <unsigned int blockSize>
__global__ void gpuann_fann_update_slopes_batch_gpu_kernel(unsigned int prevNeuronsCount,
                                                           unsigned int neuronsCount,
                                                           fann_type *trainErrors,
                                                           fann_type *neuronSlopes,
                                                           fann_type *prevValue,
                                                           unsigned int totalNeuronsCount,
                                                           unsigned int totalWeightsCount)
{
  unsigned int tid                  = threadIdx.x;
  unsigned int instance             = blockIdx.y;
  unsigned int neuronIndex          = tid + blockIdx.x * blockSize;

  if(neuronIndex < neuronsCount - 1) //TODO: bias
  {
    unsigned int neuronIndexInstanced = neuronIndex + instance * totalNeuronsCount;

    fann_type error = trainErrors[neuronIndexInstanced];

    unsigned int prevLayerNeuronIndex = 0;
    unsigned int prevLayerNeuronIndexInstanced;
    unsigned int slopesIndexInstanced;

    while(prevLayerNeuronIndex < prevNeuronsCount)
    {
      prevLayerNeuronIndexInstanced = prevLayerNeuronIndex + instance * totalNeuronsCount;
      slopesIndexInstanced          = prevLayerNeuronIndex + prevNeuronsCount * neuronIndex + instance * totalWeightsCount;

      neuronSlopes[slopesIndexInstanced] += error * prevValue[prevLayerNeuronIndexInstanced];
      prevLayerNeuronIndex += 1;
    }
  }
}

void gpuann_fann_update_slopes_batch_simple_implementation(unsigned int prevNeuronsCount,
                                                           unsigned int neuronsCount,
                                                           fann_type   *trainErrors,
                                                           fann_type   *neuronSlopes,
                                                           fann_type   *prevValue,
                                                           unsigned int totalNeuronsCount,
                                                           unsigned int totalWeightsCount,
                                                           unsigned int instanceCount)
{
#define gpuann_fann_update_slopes_batch_blockSizeCase(X)   case X: \
gpuann_fann_update_slopes_batch_gpu_kernel<X> <<<dimGrid, dimBlock>>> (prevNeuronsCount, neuronsCount, trainErrors, neuronSlopes, prevValue, totalNeuronsCount, totalWeightsCount); \
break;

  unsigned int threadCount = pow2roundup(prevNeuronsCount);
  if(threadCount < 32)
    threadCount = 32;

  if(threadCount > 256)
    threadCount = 256;

  dim3 dimBlock(threadCount, 1, 1);
  dim3 dimGrid(neuronsCount / threadCount + 1, instanceCount, 1); // TODO create bias if

  switch (threadCount)
  {
    gpuann_fann_update_slopes_batch_blockSizeCase(32);
    gpuann_fann_update_slopes_batch_blockSizeCase(64);
    gpuann_fann_update_slopes_batch_blockSizeCase(128);
    gpuann_fann_update_slopes_batch_blockSizeCase(256);
  }
}

void gpuann_fann_update_slopes_batch_implementation(gpuann &data, fann_layer *layerBegin, fann_layer *layerEnd)
{
  const fann *ann = data._fann;
  unsigned int instanceCount = data._instanceCount;
  fann_neuron *firstNeuron = ann->first_layer->first_neuron;
  fann_layer *layerIt;

  for(layerIt = layerBegin; layerIt <= layerEnd; ++layerIt)
  {
    unsigned int layerSize = layerIt->last_neuron - layerIt->first_neuron;
    unsigned int layerNeuronShift = layerIt->first_neuron - firstNeuron;
    unsigned int layerWeightShift = layerIt->first_neuron->first_con;

    fann_layer  *prevLayer = layerIt - 1;
    fann_neuron *prevLayerFirstNeuron = prevLayer->first_neuron;
    unsigned int prevLayerNeuronShift = prevLayerFirstNeuron - firstNeuron;
    unsigned int prevLayerSize = prevLayer->last_neuron - prevLayerFirstNeuron;

    if(prevLayerSize < 32)
    {
      unsigned int threadNeeded = pow2roundup((layerSize - 1) * prevLayerSize);
      if(threadNeeded > 256)
        threadNeeded = 256;
      unsigned int neuronsPerBlock = threadNeeded / prevLayerSize;
      unsigned int blocksNeeded = (layerSize - 1) / neuronsPerBlock + 1;
      dim3 dimBlock(threadNeeded, 1, 1);
      dim3 dimGrid(blocksNeeded, instanceCount, 1); // TODO create bias if

      gpuann_fann_update_slopes_batch_multineuron_gpu_kernel<<<dimGrid, dimBlock>>>(prevLayerSize,
        layerSize,
        &(data.d_trainErrorsArray[layerNeuronShift]),
        &(data.d_trainSlopes[layerWeightShift]),
        &(data.d_valuesArray[prevLayerNeuronShift]),
        data._neuronsCountPerInstance,
        data._weightsCountPerInstance
        );
    }
    else
    {
      gpuann_fann_update_slopes_batch_simple_implementation(prevLayerSize,
                                                            layerSize,
                                                            &(data.d_trainErrorsArray[layerNeuronShift]),
                                                            &(data.d_trainSlopes[layerWeightShift]),
                                                            &(data.d_valuesArray[prevLayerNeuronShift]),
                                                            data._neuronsCountPerInstance,
                                                            data._weightsCountPerInstance,
                                                            instanceCount
                                                           );
    }
  }
}
