
#include <kernels/updateWeights/updateWeights.h>
#include <fann.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <common/math.h>

template <unsigned int blockSize>
__global__ void gpuann_fann_update_weights_gpu_kernel(unsigned int neuronInputCount,
                                                      fann_type *prevNeuronValues,
                                                      fann_type *trainErrors,
                                                      fann_type *weights,
                                                      fann_type *prevWeightsDeltas,
                                                      float learningMomentum,
                                                      float learningRate,
                                                      unsigned int totalNeuronsCount,
                                                      unsigned int totalWeightsCount)
{
  unsigned int tid                  = threadIdx.x;
  unsigned int neuronIndex          = blockIdx.x;
  unsigned int instance             = blockIdx.y;
  unsigned int prevNeuronIndex      = tid;
  unsigned int neuronIndexInstanced = neuronIndex + instance * totalNeuronsCount;

  fann_type tmpError = 0;

  if(prevNeuronIndex < neuronInputCount)
    tmpError = trainErrors[neuronIndexInstanced] * learningRate;

  while(prevNeuronIndex < neuronInputCount)
  {
    unsigned int weightIndexInstanced     = prevNeuronIndex + neuronIndex * neuronInputCount + instance * totalWeightsCount;
    unsigned int prevNeuronIndexInstanced = prevNeuronIndex + instance * totalNeuronsCount;

    fann_type delta_w = tmpError * prevNeuronValues[prevNeuronIndexInstanced] + learningMomentum * prevWeightsDeltas[weightIndexInstanced];
    weights[weightIndexInstanced] += delta_w;
    prevWeightsDeltas[weightIndexInstanced] = delta_w;

    prevNeuronIndex += blockSize;
  }
}

void gpuann_fann_update_weights_simple_implementation(unsigned int neuronInputCount,
                                                      fann_type *prevNeuronValues,
                                                      fann_type *trainErrors,
                                                      fann_type *weights,
                                                      fann_type *prevWeightsDeltas,
                                                      float learningMomentum,
                                                      float learningRate,
                                                      unsigned int totalNeuronsCount,
                                                      unsigned int totalWeightsCount,
                                                      unsigned int layerSize,
                                                      unsigned int instanceCount
                                                     )
{
#define gpuann_fann_update_weights_gpu_kernelCase(X)   case X: \
gpuann_fann_update_weights_gpu_kernel<X> <<<dimGrid, dimBlock>>>(neuronInputCount, prevNeuronValues, trainErrors, weights, prevWeightsDeltas, learningMomentum, learningRate, totalNeuronsCount, totalWeightsCount); \
break;

  unsigned int threadCount = pow2roundup(neuronInputCount) / 2;
  if(threadCount < 32)
    threadCount = 32;
  if(threadCount > 256)
    threadCount = 256;

  dim3 dimBlock(threadCount, 1, 1); //TODO remove bias
  dim3 dimGrid(layerSize - 1, instanceCount, 1);

  switch (threadCount)
  {
    gpuann_fann_update_weights_gpu_kernelCase(32);
    gpuann_fann_update_weights_gpu_kernelCase(64);
    gpuann_fann_update_weights_gpu_kernelCase(128);
    gpuann_fann_update_weights_gpu_kernelCase(256);
  }
}

void gpuann_fann_update_weights_implementation(gpuann &data)
{
  const fann *ann = data._fann;

  unsigned int instanceCount = data._instanceCount;
  float learningRate = ann->learning_rate;
  float learningMomentum = ann->learning_momentum;

  struct fann_neuron *neuronsArray = ann->first_layer->first_neuron;
  fann_layer *firstLayer = ann->first_layer;
  const fann_layer *lastLayer = ann->last_layer;
  fann_layer *layerIt;

  for(layerIt = (firstLayer + 1); layerIt != lastLayer; layerIt++)
  {
    fann_neuron *neuronIt = layerIt->first_neuron;

    unsigned int layerSize = layerIt->last_neuron - layerIt->first_neuron;
    unsigned int layerNeuronInputCount = neuronIt->last_con - neuronIt->first_con;
    unsigned int prevLayerNeuronArrayShift = (layerIt - 1)->first_neuron - neuronsArray;
    unsigned int currentLayerNeuronArrayShift = layerIt->first_neuron - neuronsArray;
    unsigned int weightsArrayShift = neuronIt->first_con;

    gpuann_fann_update_weights_simple_implementation(layerNeuronInputCount,
                                                     &(data.d_valuesArray[prevLayerNeuronArrayShift]),
                                                     &(data.d_trainErrorsArray[currentLayerNeuronArrayShift]),
                                                     &(data.d_weightsArray[weightsArrayShift]),
                                                     &(data.d_prevWeightsDeltas[weightsArrayShift]),
                                                     learningMomentum,
                                                     learningRate,
                                                     data._neuronsCountPerInstance,
                                                     data._weightsCountPerInstance,
                                                     layerSize,
                                                     instanceCount
                                                    );
  }
}
