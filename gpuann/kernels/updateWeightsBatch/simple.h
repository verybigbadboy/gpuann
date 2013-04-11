#include <kernels/updateWeightsBatch/updateWeightsBatch.h>
#include <fann.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <common/math.h>


__global__ void gpuann_fann_update_weights_batch_gpu_kernel(unsigned int weightsToUpdateCount, fann_type *slopes, fann_type *weights, const float epsilon
, unsigned int totalWeightsCount)
{
  unsigned int tid = threadIdx.x;
  unsigned int blockIndex = blockIdx.x;
  unsigned int blockSize = blockDim.x;
  unsigned int instance = blockIdx.y;

  unsigned int weightIndex = tid + blockIndex * blockSize;
  unsigned int weightIndexInstanced = weightIndex + totalWeightsCount * instance;

  if(weightIndex < weightsToUpdateCount)
  {
    weights[weightIndexInstanced] += slopes[weightIndexInstanced] * epsilon;
    slopes[weightIndexInstanced] = 0;
  }
}

void gpuann_fann_update_weights_batch_implementation(gpuann &data, unsigned int num_data, unsigned int first_weight, unsigned int past_end)
{
  const fann *ann = data._fann;

  unsigned int instanceCount = data._instanceCount;
  unsigned int weightsToUpdateCount = past_end - first_weight;
  const float epsilon = ann->learning_rate / num_data;

  unsigned int threadCount = 256;
  dim3 dimBlock(threadCount, 1, 1);
  dim3 dimGrid(weightsToUpdateCount / threadCount + 1, instanceCount, 1);

  gpuann_fann_update_weights_batch_gpu_kernel<<<dimGrid, dimBlock>>> (weightsToUpdateCount,
                                                                      &(data.d_trainSlopes[first_weight]),
                                                                      &(data.d_weightsArray[first_weight]),
                                                                      epsilon,
                                                                      data._weightsCountPerInstance
                                                                     );
}

/*
void fann_update_weights_batch(struct fann *ann, unsigned int num_data, unsigned int first_weight, unsigned int past_end)
{
  fann_type *train_slopes = ann->train_slopes;
  fann_type *weights = ann->weights;
  const float epsilon = ann->learning_rate / num_data;
  unsigned int i = first_weight;
  
  for(; i != past_end; i++)
  {
    weights[i] += train_slopes[i] * epsilon;
    train_slopes[i] = 0.0;
  }
}
*/