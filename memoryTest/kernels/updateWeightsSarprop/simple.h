
#include <kernels/updateWeights/updateWeights.h>
#include <fann.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <common/math.h>

__global__ void gpuann_fann_update_weights_sarprop_gpu_kernel(unsigned int weightsToUpdateCount, fann_type *weights, fann_type *slopes, fann_type *prevSlopes, fann_type *prevSteps,
                                                              const unsigned int epoch,
                                                              const float increase_factor,
                                                              const float decrease_factor,
                                                              const float delta_min,
                                                              const float delta_max,
                                                              const float weight_decay_shift,
                                                              const float step_error_threshold_factor,
                                                              const float step_error_shift,
                                                              const float T,
                                                              const float MSE,
                                                              const float RMSE,
                                                              const unsigned int totalWeightsCount)
{
  unsigned int tid = threadIdx.x;
  unsigned int blockIndex = blockIdx.x;
  unsigned int blockSize = blockDim.x;
  unsigned int instance = blockIdx.y;
  unsigned int weightIndex = tid + blockIndex * blockSize;
  unsigned int weightIndexInstanced = weightIndex + totalWeightsCount * instance;

  if(weightIndex < totalWeightsCount)
  {
    fann_type prev_step, slope, prev_slope, next_step = 0, same_sign;

    /* TODO: confirm whether 1x10^-6 == delta_min is really better */
    prev_step = fann_max(prevSteps[weightIndexInstanced], (fann_type) 0.000001);    /* prev_step may not be zero because then the training will stop */
    /* calculate SARPROP slope; TODO: better as new error function? (see SARPROP paper)*/
    slope = -slopes[weightIndexInstanced] - weights[weightIndexInstanced] * (fann_type)fann_exp2(-T * epoch + weight_decay_shift);

    /* TODO: is prev_train_slopes[i] 0.0 in the beginning? */
    prev_slope = prevSlopes[weightIndexInstanced];

    same_sign = prev_slope * slope;

    if(same_sign > 0.0)
    {
      next_step = fann_min(prev_step * increase_factor, delta_max);
      /* TODO: are the signs inverted? see differences between SARPROP paper and iRprop */
      if (slope < 0.0)
        weights[weightIndexInstanced] += next_step;
      else
        weights[weightIndexInstanced] -= next_step;
    }
    else if(same_sign < 0.0)
    {
      if(prev_step < step_error_threshold_factor * MSE)
        next_step = prev_step * decrease_factor;// + (float)rand() / RAND_MAX * RMSE * (fann_type)fann_exp2(-T * epoch + step_error_shift); TODO RAND
        else
          next_step = fann_max(prev_step * decrease_factor, delta_min);
        
        slope = 0.0;
    }
    else
    {
      if(slope < 0.0)
        weights[weightIndexInstanced] += prev_step;
      else
        weights[weightIndexInstanced] -= prev_step;
    }

    prevSteps[weightIndexInstanced] = next_step;
    prevSlopes[weightIndexInstanced] = slope;
    slopes[weightIndexInstanced] = 0.0;
  }
}

void gpuann_fann_update_weights_sarprop_implementation(gpuann &data, unsigned int epoch, unsigned int first_weight, unsigned int past_end)
{
  const fann *ann = data._fann;
  
  unsigned int instanceCount = data._instanceCount;
  unsigned int weightsToUpdateCount = past_end - first_weight;

  float increase_factor = ann->rprop_increase_factor;
  float decrease_factor = ann->rprop_decrease_factor;
  
  float delta_min = 0.000001f;
  float delta_max = ann->rprop_delta_max;
  float weight_decay_shift = ann->sarprop_weight_decay_shift;
  float step_error_threshold_factor = ann->sarprop_step_error_threshold_factor;
  float step_error_shift = ann->sarprop_step_error_shift;
  float T = ann->sarprop_temperature;

  //fix meeee! TODO
  float MSE = 1; //fann_get_MSE(ann);
  float RMSE = (float)sqrt(MSE);
  
  unsigned int threadCount = 256;
  dim3 dimBlock(threadCount, 1, 1);
  dim3 dimGrid(weightsToUpdateCount / threadCount + 1, instanceCount, 1);
  
  gpuann_fann_update_weights_sarprop_gpu_kernel<<<dimGrid, dimBlock>>> (weightsToUpdateCount,
                                                                        &(data.d_weightsArray[first_weight]),
                                                                        &(data.d_trainSlopes[first_weight]),
                                                                        &(data.d_prevTrainSlopes[first_weight]),
                                                                        &(data.d_prevSteps[first_weight]),
                                                                        epoch,
                                                                        increase_factor,
                                                                        decrease_factor,
                                                                        delta_min,
                                                                        delta_max,
                                                                        weight_decay_shift,
                                                                        step_error_threshold_factor,
                                                                        step_error_shift,
                                                                        T,
                                                                        MSE,
                                                                        RMSE,
                                                                        data._weightsCountPerInstance);
}

/*
void fann_update_weights_sarprop(struct fann *ann, unsigned int epoch, unsigned int first_weight, unsigned int past_end)
{
  fann_type *train_slopes = ann->train_slopes;
  fann_type *weights = ann->weights;
  fann_type *prev_steps = ann->prev_steps;
  fann_type *prev_train_slopes = ann->prev_train_slopes;
  
  fann_type prev_step, slope, prev_slope, next_step = 0, same_sign;
  
  float increase_factor = ann->rprop_increase_factor;
  float decrease_factor = ann->rprop_decrease_factor;
  
  float delta_min = 0.000001f;
  float delta_max = ann->rprop_delta_max;
  float weight_decay_shift = ann->sarprop_weight_decay_shift;
  float step_error_threshold_factor = ann->sarprop_step_error_threshold_factor;
  float step_error_shift = ann->sarprop_step_error_shift;
  float T = ann->sarprop_temperature;
  float MSE = fann_get_MSE(ann);
  float RMSE = (float)sqrt(MSE);
  
  unsigned int i = first_weight;
  
  
  for(; i != past_end; i++)
  {
    prev_step = fann_max(prev_steps[i], (fann_type) 0.000001);

    slope = -train_slopes[i] - weights[i] * (fann_type)fann_exp2(-T * epoch + weight_decay_shift);
    

    prev_slope = prev_train_slopes[i];
    
    same_sign = prev_slope * slope;
    
    if(same_sign > 0.0)
    {
      next_step = fann_min(prev_step * increase_factor, delta_max);

      if (slope < 0.0)
        weights[i] += next_step;
      else
        weights[i] -= next_step;
    }
    else if(same_sign < 0.0)
    {
      if(prev_step < step_error_threshold_factor * MSE)
        next_step = prev_step * decrease_factor + (float)rand() / RAND_MAX * RMSE * (fann_type)fann_exp2(-T * epoch + step_error_shift);
      else
        next_step = fann_max(prev_step * decrease_factor, delta_min);
      
      slope = 0.0;
    }
    else
    {
      if(slope < 0.0)
        weights[i] += prev_step;
      else
        weights[i] -= prev_step;
    }

    prev_steps[i] = next_step;
    prev_train_slopes[i] = slope;
    train_slopes[i] = 0.0;
  }
}
*/
