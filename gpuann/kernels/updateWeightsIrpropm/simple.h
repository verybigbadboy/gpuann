
#include <kernels/updateWeights/updateWeights.h>
#include <fann.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <common/math.h>

__global__ void gpuann_fann_update_weights_irpropm_gpu_kernel(unsigned int weightsToUpdateCount, fann_type *weights, fann_type *slopes, fann_type *prevSlopes, fann_type *prevSteps,
                                                              float increaseFactor, float decreaseFactor, const float deltaMin, const float deltaMax,
                                                              unsigned int totalWeightsCount)
{
  unsigned int tid = threadIdx.x;
  unsigned int blockIndex = blockIdx.x;
  unsigned int blockSize = blockDim.x;
  unsigned int instance = blockIdx.y;
  unsigned int weightIndex = tid + blockIndex * blockSize;
  unsigned int weightIndexInstanced = weightIndex + totalWeightsCount * instance;

  if(weightIndex < weightsToUpdateCount)
  {
    fann_type prevStep = fann_max(prevSteps[weightIndexInstanced], (fann_type) 0.0001);    /* prevStep may not be zero because then the training will stop */
    fann_type slope = slopes[weightIndexInstanced];
    fann_type prevSlope = prevSlopes[weightIndexInstanced];
    fann_type nextStep;
    
    float sameSign = prevSlope * slope;
    
    if(sameSign >= 0.0)
      nextStep = fann_min(prevStep * increaseFactor, deltaMax);
    else
    {
      nextStep = fann_max(prevStep * decreaseFactor, deltaMin);
      slope = 0;
    }
    
    if(slope < 0)
    {
      weights[weightIndexInstanced] -= nextStep;
      if(weights[weightIndexInstanced] < -1500)
        weights[weightIndexInstanced] = -1500;
    }
    else
    {
      weights[weightIndexInstanced] += nextStep;
      if(weights[weightIndexInstanced] > 1500)
        weights[weightIndexInstanced] = 1500;
    }
    
    prevSteps[weightIndexInstanced] = nextStep;
    prevSlopes[weightIndexInstanced] = slope;
    slopes[weightIndexInstanced] = 0.0;
  }
}

void gpuann_fann_update_weights_irpropm_implementation(gpuann &data, unsigned int first_weight, unsigned int past_end)
{
  const fann *ann = data._fann;

  unsigned int instanceCount = data._instanceCount;
  unsigned int weightsToUpdateCount = past_end - first_weight;

  unsigned int threadCount = 256;
  dim3 dimBlock(threadCount, 1, 1);
  dim3 dimGrid(weightsToUpdateCount / threadCount + 1, instanceCount, 1);

  gpuann_fann_update_weights_irpropm_gpu_kernel<<<dimGrid, dimBlock>>> (weightsToUpdateCount,
                                                                        &(data.d_weightsArray[first_weight]),
                                                                        &(data.d_trainSlopes[first_weight]),
                                                                        &(data.d_prevTrainSlopes[first_weight]),
                                                                        &(data.d_prevSteps[first_weight]),
                                                                        ann->rprop_increase_factor,
                                                                        ann->rprop_decrease_factor,
                                                                        ann->rprop_delta_min,
                                                                        ann->rprop_delta_max,
                                                                        data._weightsCountPerInstance
  );
}

/*
void fann_update_weights_irpropm(struct fann *ann, unsigned int first_weight, unsigned int past_end)
{
  fann_type *train_slopes = ann->train_slopes;
  fann_type *weights = ann->weights;
  fann_type *prev_steps = ann->prev_steps;
  fann_type *prev_train_slopes = ann->prev_train_slopes;

  fann_type prev_step, slope, prev_slope, next_step, same_sign;

  float increase_factor = ann->rprop_increase_factor;    
  float decrease_factor = ann->rprop_decrease_factor;    
  float delta_min = ann->rprop_delta_min;    
  float delta_max = ann->rprop_delta_max;    

  unsigned int i = first_weight;

  for(; i != past_end; i++)
  {
    prev_step = fann_max(prev_steps[i], (fann_type) 0.0001);  
    slope = train_slopes[i];
    prev_slope = prev_train_slopes[i];

    same_sign = prev_slope * slope;

    if(same_sign >= 0.0)
      next_step = fann_min(prev_step * increase_factor, delta_max);
    else
    {
      next_step = fann_max(prev_step * decrease_factor, delta_min);
      slope = 0;
    }

    if(slope < 0)
    {
      weights[i] -= next_step;
      if(weights[i] < -1500)
        weights[i] = -1500;
    }
    else
    {
      weights[i] += next_step;
      if(weights[i] > 1500)
        weights[i] = 1500;
    }

    prev_steps[i] = next_step;
    prev_train_slopes[i] = slope;
    train_slopes[i] = 0.0;
  }
}
*/