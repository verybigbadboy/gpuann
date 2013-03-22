
#include <kernels/updateWeights/updateWeights.h>
#include <fann.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <common/math.h>


__global__ void gpuann_fann_update_weights_implementation_gpu_kernel(unsigned int neuronInputCount, fann_type *prevNeuronValues, fann_type *trainErrors, fann_type *weights, fann_type *prevWeightsDeltas, float learningMomentum, float learningRate
, unsigned int totalNeuronsCount, unsigned int totalWeightsCount)
{
  unsigned int tid = threadIdx.x;
  unsigned int neuronIndex = blockIdx.x;
  unsigned int instance = blockIdx.y;
  unsigned int weightIndex     = tid + neuronIndex * neuronInputCount + instance * totalWeightsCount;
  unsigned int prevNeuronIndex = tid + instance * totalNeuronsCount;

  if(tid < neuronInputCount)
  {
    fann_type tmpError = trainErrors[neuronIndex + instance * totalNeuronsCount] * learningRate;

    fann_type delta_w = tmpError * prevNeuronValues[prevNeuronIndex] + learningMomentum * prevWeightsDeltas[weightIndex];
    weights[weightIndex] += delta_w;
    prevWeightsDeltas[weightIndex] = delta_w;
  }
}

void gpuann_fann_update_weights_implementation(gpuann &data)
{
  const fann *ann = data._fann;

  unsigned int instanceCount = data._instanceCount;
  float learningRate = ann->learning_rate;
  float learningMomentum = ann->learning_momentum;

  struct fann_neuron *neuronsArray = ann->first_layer->first_neuron;
  fann_layer *firstLayer = ann->first_layer;
  const fann_layer *lastLayer = ann->last_layer;
  fann_layer *layerIt;

  for(layerIt = (firstLayer + 1); layerIt != lastLayer; layerIt++)
  {
    fann_neuron *neuronIt = layerIt->first_neuron;

    unsigned int layerSize = layerIt->last_neuron - layerIt->first_neuron;
    unsigned int layerNeuronInputCount = neuronIt->last_con - neuronIt->first_con;
    unsigned int prevLayerNeuronArrayShift = (layerIt - 1)->first_neuron - neuronsArray;
    unsigned int currentLayerNeuronArrayShift = layerIt->first_neuron - neuronsArray;
    unsigned int weightsArrayShift = neuronIt->first_con;

    unsigned int threadCount = pow2roundup(layerNeuronInputCount);

    dim3 dimBlock(threadCount, 1, 1); //TODO remove bias
    dim3 dimGrid(layerSize - 1, instanceCount, 1);

    gpuann_fann_update_weights_implementation_gpu_kernel<<<dimGrid, dimBlock>>>(
           layerNeuronInputCount,
           &(data.d_valuesArray[prevLayerNeuronArrayShift]),
           &(data.d_trainErrorsArray[currentLayerNeuronArrayShift]),
           &(data.d_weightsArray[weightsArrayShift]),
           &(data.d_prevWeightsDeltas[weightsArrayShift]),
           learningMomentum,
           learningRate,
           data._neuronsCountPerInstance,
           data._weightsCountPerInstance);
  }
}

void gpuann_fann_update_weights_implementation(struct fann *ann)
{
  struct fann_neuron *neuron_it, *last_neuron, *prev_neurons;
  fann_type tmp_error, delta_w, *weights;
  struct fann_layer *layer_it;
  unsigned int i;
  unsigned int num_connections;

  /* store some variabels local for fast access */
  const float learning_rate = ann->learning_rate;
  const float learning_momentum = ann->learning_momentum;
  struct fann_neuron *first_neuron = ann->first_layer->first_neuron;
  struct fann_layer *first_layer = ann->first_layer;
  const struct fann_layer *last_layer = ann->last_layer;
  fann_type *error_begin = ann->train_errors;
  fann_type *deltas_begin, *weights_deltas;

  deltas_begin = ann->prev_weights_deltas;
  prev_neurons = first_neuron;
  for(layer_it = (first_layer + 1); layer_it != last_layer; layer_it++)
  {
    last_neuron = layer_it->last_neuron;
    prev_neurons = (layer_it - 1)->first_neuron;

    for(neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++)
    {
      tmp_error = error_begin[neuron_it - first_neuron] * learning_rate;
      num_connections = neuron_it->last_con - neuron_it->first_con;
      weights = ann->weights + neuron_it->first_con;
      weights_deltas = deltas_begin + neuron_it->first_con;
      for(i = 0; i != num_connections; i++)
      {
        delta_w = tmp_error * prev_neurons[i].value + learning_momentum * weights_deltas[i];
        weights[i] += delta_w;
        weights_deltas[i] = delta_w;
      }
    }
  }
}