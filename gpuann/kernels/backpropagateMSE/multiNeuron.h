

template <unsigned int prevActivationFunction>
__global__ void fann_backpropagate_MSE_multineuron_gpu_kernel(unsigned int prevNeuronsCount,
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
  const unsigned int threadCount        = 256;
  unsigned int tid                      = threadIdx.x;
  unsigned int instance                 = blockIdx.y;
  unsigned int weightPerNeuronCount     = prevNeuronsCount;
  unsigned int neuronIndexInKernel      = tid / neuronsCount;
  unsigned int neuronsPerKernel         = threadCount / neuronsCount;
  unsigned int neuronIndex              = tid % neuronsCount;
  unsigned int neuronIndexInstanced     = neuronIndex + instance * totalNeuronsCount;
  unsigned int prevLayerNeuron          = blockIdx.x * neuronsPerKernel + neuronIndexInKernel;
  unsigned int prevLayerNeuronInstanced = prevLayerNeuron + instance * totalNeuronsCount;
  unsigned int weightBeginIndex         = neuronIndex * weightPerNeuronCount + instance * totalWeightsCount;
  
  __shared__ fann_type sum[threadCount];
  
  fann_type l_summ = 0;
  
  if(tid < neuronsPerKernel * neuronsCount && prevLayerNeuron < prevNeuronsCount)
  {
    l_summ = trainErrors[neuronIndexInstanced] * weights[weightBeginIndex + prevLayerNeuron];
    
    sum[tid] = l_summ;
  }
  
  __syncthreads();
  
  if (tid < neuronsPerKernel * neuronsCount && prevLayerNeuron < prevNeuronsCount)
  {
    volatile fann_type *smem = sum;
    
    if(neuronsCount > 16)
      if(neuronIndex < 16)
        if(neuronIndex + 16 < neuronsCount)
          smem[tid] = l_summ = l_summ + smem[tid + 16];
        
        if(neuronsCount > 8)
          if(neuronIndex < 8)
            if(neuronIndex + 8 < neuronsCount)
              smem[tid] = l_summ = l_summ + smem[tid + 8];
            
            if(neuronsCount > 4)
              if(neuronIndex < 4)
                if(neuronIndex + 4 < neuronsCount)
                  smem[tid] = l_summ = l_summ + smem[tid + 4];
                
                if(neuronsCount > 2)
                  if(neuronIndex < 2)
                    if(neuronIndex + 2 < neuronsCount)
                      smem[tid] = l_summ = l_summ + smem[tid + 2];
                    
                    if(neuronsCount > 1)
                      if(neuronIndex < 1)
                        if(neuronIndex + 1 < neuronsCount)
                          smem[tid] = l_summ = l_summ + smem[tid + 1];
  }
  __syncthreads();
  
  if (neuronIndex == 0 && prevLayerNeuron < prevNeuronsCount && neuronIndexInKernel < neuronsPerKernel)
  {
    prevTrainErrors[prevLayerNeuronInstanced] = sum[tid] * gpuann_fann_activation_derived<prevActivationFunction>(prevSteepness, prevValue[prevLayerNeuronInstanced], prevSum[prevLayerNeuronInstanced]);
  }
}

void fann_backpropagate_MSE_multineuron_implementation(unsigned int instanceCount,
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
  unsigned int threadNeeded = pow2roundup(neuronsCount * prevNeuronsCount);
  if(threadNeeded > 256)
    threadNeeded = 256;
  unsigned int prevNeuronsPerBlock = threadNeeded / neuronsCount;
  unsigned int blocksNeeded = prevNeuronsCount / prevNeuronsPerBlock + 1;
  dim3 dimBlock(threadNeeded, 1, 1);
  dim3 dimGrid(blocksNeeded, instanceCount, 1);

  #define fann_backpropagate_MSE_multineuron_gpu_kernel_case(X)   case X: \
  fann_backpropagate_MSE_multineuron_gpu_kernel<X> <<<dimGrid, dimBlock>>> (prevNeuronsCount, neuronsCount, weights, trainErrors, prevTrainErrors, prevValue, prevSum, prevSteepness, totalNeuronsCount, totalWeightsCount); \
  break;

  switch(prevActivationFunction)
  {
    fann_backpropagate_MSE_multineuron_gpu_kernel_case(0);
    fann_backpropagate_MSE_multineuron_gpu_kernel_case(1);
    fann_backpropagate_MSE_multineuron_gpu_kernel_case(2);
    fann_backpropagate_MSE_multineuron_gpu_kernel_case(3);
    fann_backpropagate_MSE_multineuron_gpu_kernel_case(4);
    fann_backpropagate_MSE_multineuron_gpu_kernel_case(5);
    fann_backpropagate_MSE_multineuron_gpu_kernel_case(6);
    fann_backpropagate_MSE_multineuron_gpu_kernel_case(7);
    fann_backpropagate_MSE_multineuron_gpu_kernel_case(8);
    fann_backpropagate_MSE_multineuron_gpu_kernel_case(9);
    fann_backpropagate_MSE_multineuron_gpu_kernel_case(10);
    fann_backpropagate_MSE_multineuron_gpu_kernel_case(11);
    fann_backpropagate_MSE_multineuron_gpu_kernel_case(12);
    fann_backpropagate_MSE_multineuron_gpu_kernel_case(13);
    fann_backpropagate_MSE_multineuron_gpu_kernel_case(14);
    fann_backpropagate_MSE_multineuron_gpu_kernel_case(15);
  }
}