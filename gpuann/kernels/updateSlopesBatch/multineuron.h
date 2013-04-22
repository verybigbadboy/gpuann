
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
  
  if(neuronIndex < neuronsCount)
  {
    fann_type error = trainErrors[neuronIndexInstanced];
    if(tid < prevNeuronsCount * neuronCountPerKernel)
      neuronSlopes[slopesIndex] += error * prevValue[prevLayerNeuronIndex];
  }
}

void gpuann_fann_update_slopes_batch_multineuron_implementation(unsigned int prevNeuronsCount,
                                                                unsigned int neuronsCount,
                                                                fann_type   *trainErrors,
                                                                fann_type   *neuronSlopes,
                                                                fann_type   *prevValue,
                                                                unsigned int totalNeuronsCount,
                                                                unsigned int totalWeightsCount,
                                                                unsigned int instanceCount)
{
  unsigned int threadNeeded = pow2roundup((neuronsCount - 1) * prevNeuronsCount);
  if(threadNeeded > 256)
    threadNeeded = 256;
  unsigned int neuronsPerBlock = threadNeeded / prevNeuronsCount;
  unsigned int blocksNeeded = (neuronsCount - 1) / neuronsPerBlock + 1;
  dim3 dimBlock(threadNeeded, 1, 1);
  dim3 dimGrid(blocksNeeded, instanceCount, 1); // TODO create bias if
  
  gpuann_fann_update_slopes_batch_multineuron_gpu_kernel<<<dimGrid, dimBlock>>>(prevNeuronsCount,
                                                                                neuronsCount - 1,  //due to bias
                                                                                trainErrors,
                                                                                neuronSlopes,
                                                                                prevValue,
                                                                                totalNeuronsCount,
                                                                                totalWeightsCount);
}