
#include <kernels/updateWeights/updateWeights.h>
#include <fann.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <common/math.h>

__global__ void gpuann_fann_update_weights_quickprop_gpu_kernel(unsigned int weightsToUpdateCount, fann_type *weights, fann_type *slopes, fann_type *prevSlopes, fann_type *prevSteps,
                                                                    float epsilon, float decay, const float mu,
                                                                    unsigned int totalWeightsCount)
{
  unsigned int tid = threadIdx.x;
  unsigned int blockIndex = blockIdx.x;
  unsigned int blockSize = blockDim.x;
  unsigned int instance = blockIdx.y;
  unsigned int weightIndex = tid + blockIndex * blockSize;
  unsigned int weightIndexInstanced = weightIndex + totalWeightsCount * instance;

  if(weightIndex < totalWeightsCount)
  {
    fann_type w, prevStep, slope, prevSlope, nextStep;
    w = weights[weightIndexInstanced];
    prevStep = prevSteps[weightIndexInstanced];
    slope = slopes[weightIndexInstanced] + decay * w;
    prevSlope = prevSlopes[weightIndexInstanced];
    nextStep = 0.0;

    /* The step must always be in direction opposite to the slope. */
    if(prevStep > 0.001)
    {
      /* If last step was positive...  */
      if(slope > 0.0) /*  Add in linear term if current slope is still positive. */
        nextStep += epsilon * slope;

      /*If current slope is close to or larger than prev slope...  */
      if(slope > ((float) (mu / (1.0 + mu))  * prevSlope))
        nextStep += mu * prevStep;    /* Take maximum size negative step. */
        else
          nextStep += prevStep * slope / (prevSlope - slope);    /* Else, use quadratic estimate. */
    }
    else if(prevStep < -0.001)
    {
      /* If last step was negative...  */
      if(slope < 0.0) /*  Add in linear term if current slope is still negative. */
        nextStep += epsilon * slope;

      /* If current slope is close to or more neg than prev slope... */
      if(slope < ((float) (mu / (1.0 + mu))  * prevSlope))
        nextStep += mu * prevStep;    /* Take maximum size negative step. */
        else
          nextStep += prevStep * slope / (prevSlope - slope);    /* Else, use quadratic estimate. */
    }
    else /* Last step was zero, so use only linear term. */
      nextStep += epsilon * slope;

    prevSteps[weightIndexInstanced] = nextStep;

    w += nextStep;

    if(w > 1500)
      weights[weightIndexInstanced] = 1500;
    else if(w < -1500)
      weights[weightIndexInstanced] = -1500;
    else
      weights[weightIndexInstanced] = w;


    prevSlopes[weightIndexInstanced] = slope;
    slopes[weightIndexInstanced] = 0.0;
  }
}

void gpuann_fann_update_weights_quickprop_implementation(gpuann &data, unsigned int num_data, unsigned int first_weight, unsigned int past_end)
{
  const fann *ann = data._fann;

  unsigned int instanceCount = data._instanceCount;
  unsigned int weightsToUpdateCount = past_end - first_weight;

  unsigned int threadCount = 256;
  dim3 dimBlock(threadCount, 1, 1);
  dim3 dimGrid(weightsToUpdateCount / threadCount + 1, instanceCount, 1);

  gpuann_fann_update_weights_quickprop_gpu_kernel<<<dimGrid, dimBlock>>> (weightsToUpdateCount,
                                                                          &(data.d_weightsArray[first_weight]),
                                                                          &(data.d_trainSlopes[first_weight]),
                                                                          &(data.d_prevTrainSlopes[first_weight]),
                                                                          &(data.d_prevSteps[first_weight]),
                                                                          ann->learning_rate / num_data,
                                                                          ann->quickprop_decay,
                                                                          ann->quickprop_mu,
                                                                          data._weightsCountPerInstance
    
  );
}

/*
void fann_update_weights_quickprop(struct fann *ann, unsigned int num_data, unsigned int first_weight, unsigned int past_end)
{
  fann_type *train_slopes = ann->train_slopes;
  fann_type *weights = ann->weights;
  fann_type *prev_steps = ann->prev_steps;
  fann_type *prev_train_slopes = ann->prev_train_slopes;

  fann_type w, prev_step, slope, prev_slope, next_step;

  float epsilon = ann->learning_rate / num_data;
  float decay = ann->quickprop_decay;
  float mu = ann->quickprop_mu;

  unsigned int i = first_weight;

  for(; i != past_end; i++)
  {
    w = weights[i];
    prev_step = prev_steps[i];
    slope = train_slopes[i] + decay * w;
    prev_slope = prev_train_slopes[i];
    next_step = 0.0;

    if(prev_step > 0.001)
    {

      if(slope > 0.0) 
        next_step += epsilon * slope;

      if(slope > ((float) (mu / (1.0 + mu))  * prev_slope))
        next_step += mu * prev_step;
      else
        next_step += prev_step * slope / (prev_slope - slope);
    }
    else if(prev_step < -0.001)
    {
      if(slope < 0.0)
        next_step += epsilon * slope;

      if(slope < ((float) (mu / (1.0 + mu))  * prev_slope))
        next_step += mu * prev_step;
      else
        next_step += prev_step * slope / (prev_slope - slope);
    }
    else
      next_step += epsilon * slope;

    prev_steps[i] = next_step;

    w += next_step;

    if(w > 1500)
      weights[i] = 1500;
    else if(w < -1500)
      weights[i] = -1500;
    else
      weights[i] = w;


    prev_train_slopes[i] = slope;
    train_slopes[i] = 0.0;
  }
}
*/

