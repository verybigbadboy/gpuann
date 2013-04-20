
template <unsigned int blockSize, unsigned int prevActivationFunction>
__global__ void fann_backpropagate_MSE_parallel_gpu_kernel(const unsigned int prevNeuronsCount,
                                                           const unsigned int neuronsCount,
                                                           fann_type *weights,
                                                           fann_type *trainErrors,
                                                           fann_type *prevTrainErrors,
                                                           fann_type *prevValue,
                                                           fann_type *prevSum,
                                                           fann_type prevSteepness,
                                                           const unsigned int totalNeuronsCount,
                                                           const unsigned int totalWeightsCount)
{
  const unsigned int tid                      = threadIdx.x;
  const unsigned int instance                 = blockIdx.y;
  const unsigned int weightPerNeuronCount     = prevNeuronsCount;
  const unsigned int prevLayerNeuron          = tid + blockSize * blockIdx.x;
  const unsigned int prevLayerNeuronInstanced = prevLayerNeuron + instance * totalNeuronsCount;

  fann_type mySum = 0;
  unsigned int neuronIndex = 0;
  if(prevLayerNeuron < prevNeuronsCount)
  {
    while (neuronIndex < neuronsCount)
    {
      const unsigned int neuronIndexInstanced = neuronIndex + instance * totalNeuronsCount;
      const unsigned int weightBeginIndex     = neuronIndex * weightPerNeuronCount + instance * totalWeightsCount;

      mySum += trainErrors[neuronIndexInstanced] * weights[weightBeginIndex + prevLayerNeuron];
      neuronIndex++;
    }

    prevTrainErrors[prevLayerNeuronInstanced] = mySum * gpuann_fann_activation_derived<prevActivationFunction>(prevSteepness, prevValue[prevLayerNeuronInstanced], prevSum[prevLayerNeuronInstanced]);
  }
}

template <unsigned int blockSize>
void fann_backpropagate_MSE_parallel_activationFunction(unsigned int instanceCount, unsigned int prevActivationFunction, unsigned int prevNeuronsCount, unsigned int neuronsCount, fann_type *weights, fann_type *trainErrors, fann_type *prevTrainErrors, fann_type *prevValue, fann_type *prevSum, fann_type prevSteepness
, unsigned int totalNeuronsCount, unsigned int totalWeightsCount)
{
  dim3 dimBlock(blockSize, 1, 1);
  dim3 dimGrid(neuronsCount / blockSize + 1, instanceCount, 1);

#define fann_backpropagate_MSE_parallel_gpu_kernel_case(X)   case X: \
fann_backpropagate_MSE_parallel_gpu_kernel<blockSize, X> <<<dimGrid, dimBlock>>> (prevNeuronsCount, neuronsCount, weights, trainErrors, prevTrainErrors, prevValue, prevSum, prevSteepness, totalNeuronsCount, totalWeightsCount); \
break;

  switch(prevActivationFunction)
  {
    fann_backpropagate_MSE_parallel_gpu_kernel_case(0);
    fann_backpropagate_MSE_parallel_gpu_kernel_case(1);
    fann_backpropagate_MSE_parallel_gpu_kernel_case(2);
    fann_backpropagate_MSE_parallel_gpu_kernel_case(3);
    fann_backpropagate_MSE_parallel_gpu_kernel_case(4);
    fann_backpropagate_MSE_parallel_gpu_kernel_case(5);
    fann_backpropagate_MSE_parallel_gpu_kernel_case(6);
    fann_backpropagate_MSE_parallel_gpu_kernel_case(7);
    fann_backpropagate_MSE_parallel_gpu_kernel_case(8);
    fann_backpropagate_MSE_parallel_gpu_kernel_case(9);
    fann_backpropagate_MSE_parallel_gpu_kernel_case(10);
    fann_backpropagate_MSE_parallel_gpu_kernel_case(11);
    fann_backpropagate_MSE_parallel_gpu_kernel_case(12);
    fann_backpropagate_MSE_parallel_gpu_kernel_case(13);
    fann_backpropagate_MSE_parallel_gpu_kernel_case(14);
    fann_backpropagate_MSE_parallel_gpu_kernel_case(15);
  }
}

void fann_backpropagate_MSE_parallel_implementation(unsigned int instanceCount,
                                                    unsigned int prevActivationFunction,
                                                    unsigned int prevNeuronsCount,
                                                    unsigned int neuronsCount,
                                                    fann_type *weights,
                                                    fann_type *trainErrors,
                                                    fann_type *prevTrainErrors,
                                                    fann_type *prevValue,
                                                    fann_type *prevSum,
                                                    fann_type prevSteepness,
                                                    unsigned int totalNeuronsCount,
                                                    unsigned int totalWeightsCount)
{
  unsigned int threadsCount = pow2roundup(prevNeuronsCount);
  
  if(threadsCount < 32)
    threadsCount = 32;
  
  #define fann_backpropagate_MSE_parallel_activationFunction_case(X)   case X: \
  fann_backpropagate_MSE_parallel_activationFunction<X> (instanceCount, prevActivationFunction, prevNeuronsCount, neuronsCount, weights, trainErrors, prevTrainErrors, prevValue, prevSum, prevSteepness, totalNeuronsCount, totalWeightsCount); \
  break;
  
  switch (threadsCount)
  {
    fann_backpropagate_MSE_parallel_activationFunction_case(32);
    fann_backpropagate_MSE_parallel_activationFunction_case(64);
    fann_backpropagate_MSE_parallel_activationFunction_case(128);
  default:
    fann_backpropagate_MSE_parallel_activationFunction_case(256);
  }
}
