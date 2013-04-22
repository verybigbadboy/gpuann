


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
  unsigned int neuronIndex          = blockIdx.x;
  unsigned int prevLayerNeuronIndex = tid;
  unsigned int neuronIndexInstanced = neuronIndex + instance * totalNeuronsCount;

  fann_type error = trainErrors[neuronIndexInstanced];
  unsigned int prevLayerNeuronIndexInstanced;
  unsigned int slopesIndexInstanced;

  while(prevLayerNeuronIndex < prevNeuronsCount)
  {
    prevLayerNeuronIndexInstanced = prevLayerNeuronIndex + instance * totalNeuronsCount;
    slopesIndexInstanced          = prevLayerNeuronIndex + prevNeuronsCount * neuronIndex + instance * totalWeightsCount;

    neuronSlopes[slopesIndexInstanced] += error * prevValue[prevLayerNeuronIndexInstanced];
    prevLayerNeuronIndex += blockSize;
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
  #define gpuann_fann_update_slopes_batch_gpu_kernelCase(X)   case X: \
gpuann_fann_update_slopes_batch_gpu_kernel<X> <<<dimGrid, dimBlock>>> (prevNeuronsCount, neuronsCount, trainErrors, neuronSlopes, prevValue, totalNeuronsCount, totalWeightsCount); \
break;

  unsigned int threadCount = pow2roundup(prevNeuronsCount) / 2;
  if(threadCount < 32)
    threadCount = 32;

  if(threadCount > 256)
    threadCount = 256;

  dim3 dimBlock(threadCount, 1, 1);
  dim3 dimGrid(neuronsCount - 1, instanceCount, 1); // TODO create bias if

  switch (threadCount)
  {
    gpuann_fann_update_slopes_batch_gpu_kernelCase(32);
    gpuann_fann_update_slopes_batch_gpu_kernelCase(64);
    gpuann_fann_update_slopes_batch_gpu_kernelCase(128);
    gpuann_fann_update_slopes_batch_gpu_kernelCase(256);
  }
}
