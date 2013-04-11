
#ifndef BASE_DERIVED_H
#define BASE_DERIVED_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <fann.h>

template <unsigned int activationFunction>
__device__ inline fann_type gpuann_fann_activation_derived(fann_type steepness, fann_type value, fann_type sum)
{
  switch (activationFunction)
  {
    case FANN_LINEAR:
    case FANN_LINEAR_PIECE:
    case FANN_LINEAR_PIECE_SYMMETRIC:
      return (fann_type) fann_linear_derive(steepness, value);
    case FANN_SIGMOID:
    case FANN_SIGMOID_STEPWISE:
      value = fann_clip(value, 0.01f, 0.99f);
      return (fann_type) fann_sigmoid_derive(steepness, value);
    case FANN_SIGMOID_SYMMETRIC:
    case FANN_SIGMOID_SYMMETRIC_STEPWISE:
      value = fann_clip(value, -0.98f, 0.98f);
      return (fann_type) fann_sigmoid_symmetric_derive(steepness, value);
    case FANN_GAUSSIAN:
      /* value = fann_clip(value, 0.01f, 0.99f); */
      return (fann_type) fann_gaussian_derive(steepness, value, sum);
    case FANN_GAUSSIAN_SYMMETRIC:
      /* value = fann_clip(value, -0.98f, 0.98f); */
      return (fann_type) fann_gaussian_symmetric_derive(steepness, value, sum);
    case FANN_ELLIOT:
      value = fann_clip(value, 0.01f, 0.99f);
      return (fann_type) fann_elliot_derive(steepness, value, sum);
    case FANN_ELLIOT_SYMMETRIC:
      value = fann_clip(value, -0.98f, 0.98f);
      return (fann_type) fann_elliot_symmetric_derive(steepness, value, sum);
    case FANN_SIN_SYMMETRIC:
      return (fann_type) fann_sin_symmetric_derive(steepness, sum);
    case FANN_COS_SYMMETRIC:
      return (fann_type) fann_cos_symmetric_derive(steepness, sum);
    case FANN_SIN:
      return (fann_type) fann_sin_derive(steepness, sum);
    case FANN_COS:
      return (fann_type) fann_cos_derive(steepness, sum);
    case FANN_THRESHOLD:
      return 0; //TODO fann_error(NULL, FANN_E_CANT_TRAIN_ACTIVATION);
  }
  return 0;
}


#endif //BASE_DERIVED_H