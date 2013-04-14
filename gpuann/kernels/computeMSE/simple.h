#include <kernels/computeMSE/run.h>
#include <fann.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <common/math.h>
#include <base/derived.h>

template <unsigned int activationFunction>
__device__ inline fann_type gpuann_fann_update_MSE(fann_type neuron_diff)
{


  switch (activationFunction)
  {
    case FANN_LINEAR_PIECE_SYMMETRIC:
    case FANN_THRESHOLD_SYMMETRIC:
    case FANN_SIGMOID_SYMMETRIC:
    case FANN_SIGMOID_SYMMETRIC_STEPWISE:
    case FANN_ELLIOT_SYMMETRIC:
    case FANN_GAUSSIAN_SYMMETRIC:
    case FANN_SIN_SYMMETRIC:
    case FANN_COS_SYMMETRIC:
      neuron_diff /= (fann_type)2.0;
      break;
    case FANN_THRESHOLD:
    case FANN_LINEAR:
    case FANN_SIGMOID:
    case FANN_SIGMOID_STEPWISE:
    case FANN_GAUSSIAN:
    case FANN_GAUSSIAN_STEPWISE:
    case FANN_ELLIOT:
    case FANN_LINEAR_PIECE:
    case FANN_SIN:
    case FANN_COS:
      break;
  }

  /*
  MSE not implemented TODO

  float neuron_diff2 = (float) (neuron_diff * neuron_diff);
  ann->MSE_value += neuron_diff2;

  if(fann_abs(neuron_diff) >= ann->bit_fail_limit)
  {
    ann->num_bit_fail++;
  }
  */

  return neuron_diff;
}

template <unsigned int layerActivationFunction, unsigned int trainErrorFunction>
__device__ inline void gpuann_fann_compute_MSE_implementation_gpu_kernel(unsigned int layerSize, fann_type *neuronValues, fann_type *desiredOutput, fann_type *trainErrors, fann_type *sums, fann_type layerSteepness, unsigned int totalNeuronsCount)
{
  unsigned int tid = threadIdx.x;
  unsigned int instance = blockIdx.y;
  unsigned int neuronIndex = tid + instance * totalNeuronsCount;

  if(tid < layerSize)
  {
    fann_type neuronDiff = desiredOutput[tid + layerSize * instance] - neuronValues[neuronIndex];

    neuronDiff = gpuann_fann_update_MSE<layerActivationFunction>(neuronDiff);

    if(trainErrorFunction != 0)
    {
      /* TODO make switch when more functions */
      if(neuronDiff < -.9999999)
        neuronDiff = -17.0;
      else if(neuronDiff > .9999999)
        neuronDiff = 17.0;
      else
        neuronDiff = (fann_type) log((1.0 + neuronDiff) / (1.0 - neuronDiff));
    }

    trainErrors[neuronIndex] = gpuann_fann_activation_derived<layerActivationFunction>(layerSteepness, neuronValues[neuronIndex], sums[neuronIndex]) * neuronDiff;
  }
}

template <unsigned int layerActivationFunction>
__device__ inline void gpuann_fann_compute_MSE_implementation_gpu_kernel_trainErrorFunction(unsigned int layerSize, fann_type *neuronValues, fann_type *desiredOutput, fann_type *trainErrors, fann_type *sums, fann_type layerSteepness, unsigned int totalNeuronsCount, unsigned int trainErrorFunction)
{
  switch (trainErrorFunction)
  {
    case 0:
      gpuann_fann_compute_MSE_implementation_gpu_kernel<layerActivationFunction, 0>(layerSize, neuronValues, desiredOutput, trainErrors, sums, layerSteepness, totalNeuronsCount);
      break;
    default:
      gpuann_fann_compute_MSE_implementation_gpu_kernel<layerActivationFunction, 1>(layerSize, neuronValues, desiredOutput, trainErrors, sums, layerSteepness, totalNeuronsCount);
      break;
  }
}

#define gpuann_fann_compute_MSE_implementation_gpu_kernel_trainErrorFunction_layerActivationFunction_case(X)   case X: \
      gpuann_fann_compute_MSE_implementation_gpu_kernel_trainErrorFunction<X>(layerSize, neuronValues, desiredOutput, trainErrors, sums, layerSteepness, totalNeuronsCount, trainErrorFunction); \
      break;

__global__ void gpuann_fann_compute_MSE_implementation_gpu_kernel_trainErrorFunction_layerActivationFunction(unsigned int layerSize, fann_type *neuronValues, fann_type *desiredOutput, fann_type *trainErrors, fann_type *sums, fann_type layerSteepness, unsigned int totalNeuronsCount, unsigned int trainErrorFunction, unsigned int layerActivationFunction)
{
  switch (layerActivationFunction)
  {
    gpuann_fann_compute_MSE_implementation_gpu_kernel_trainErrorFunction_layerActivationFunction_case(0);
    gpuann_fann_compute_MSE_implementation_gpu_kernel_trainErrorFunction_layerActivationFunction_case(1);
    gpuann_fann_compute_MSE_implementation_gpu_kernel_trainErrorFunction_layerActivationFunction_case(2);
    gpuann_fann_compute_MSE_implementation_gpu_kernel_trainErrorFunction_layerActivationFunction_case(3);
    gpuann_fann_compute_MSE_implementation_gpu_kernel_trainErrorFunction_layerActivationFunction_case(4);
    gpuann_fann_compute_MSE_implementation_gpu_kernel_trainErrorFunction_layerActivationFunction_case(5);
    gpuann_fann_compute_MSE_implementation_gpu_kernel_trainErrorFunction_layerActivationFunction_case(6);
    gpuann_fann_compute_MSE_implementation_gpu_kernel_trainErrorFunction_layerActivationFunction_case(7);
    gpuann_fann_compute_MSE_implementation_gpu_kernel_trainErrorFunction_layerActivationFunction_case(8);
    gpuann_fann_compute_MSE_implementation_gpu_kernel_trainErrorFunction_layerActivationFunction_case(9);
    gpuann_fann_compute_MSE_implementation_gpu_kernel_trainErrorFunction_layerActivationFunction_case(10);
    gpuann_fann_compute_MSE_implementation_gpu_kernel_trainErrorFunction_layerActivationFunction_case(11);
    gpuann_fann_compute_MSE_implementation_gpu_kernel_trainErrorFunction_layerActivationFunction_case(12);
    gpuann_fann_compute_MSE_implementation_gpu_kernel_trainErrorFunction_layerActivationFunction_case(13);
    gpuann_fann_compute_MSE_implementation_gpu_kernel_trainErrorFunction_layerActivationFunction_case(14);
    gpuann_fann_compute_MSE_implementation_gpu_kernel_trainErrorFunction_layerActivationFunction_case(15);
  }
}

void gpuann_fann_compute_MSE_implementation_gpu(gpuann &data, fann_type * d_desired_output)
{
  const fann * ann = data._fann;

  fann_neuron * firstNeuronFromLastLayer = (ann->last_layer - 1)->first_neuron;
  unsigned int firstNeuronIndex = firstNeuronFromLastLayer - ann->first_layer->first_neuron;
  unsigned int layerSize = ann->num_output;
  unsigned int instanceCount = data._instanceCount;
  fann_type layerSteepness = firstNeuronFromLastLayer->activation_steepness;
  unsigned int layerActivationFunction = firstNeuronFromLastLayer->activation_function;
  unsigned int trainErrorFunction = ann->train_error_function;

  unsigned int threadCount = pow2roundup(layerSize);
  dim3 dimBlock(threadCount, 1, 1);
  dim3 dimGrid(1, instanceCount, 1);

  gpuann_fann_compute_MSE_implementation_gpu_kernel_trainErrorFunction_layerActivationFunction<<<dimGrid, dimBlock>>>
    ( layerSize
    , &(data.d_valuesArray[firstNeuronIndex])
    , d_desired_output
    , &(data.d_trainErrorsArray[firstNeuronIndex])
    , &(data.d_sumArray[firstNeuronIndex])
    , layerSteepness
    , data._neuronsCountPerInstance
    , trainErrorFunction
    , layerActivationFunction
  );
}