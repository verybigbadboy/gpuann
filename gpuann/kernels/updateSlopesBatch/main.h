#include <kernels/updateSlopesBatch/updateSlopesBatch.h>
#include <fann.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <common/math.h>
#include <string>

#include <kernels/updateSlopesBatch/simple.h>
#include <kernels/updateSlopesBatch/multineuron.h>
#include <kernels/updateSlopesBatch/bigNeuron.h>


void gpuann_fann_update_slopes_batch_implementation(gpuann &data, fann_layer *layerBegin, fann_layer *layerEnd)
{
  const fann *ann = data._fann;
  unsigned int instanceCount = data._instanceCount;
  fann_neuron *firstNeuron = ann->first_layer->first_neuron;
  fann_layer *layerIt;

  for(layerIt = layerBegin; layerIt <= layerEnd; ++layerIt)
  {
    unsigned int layerSize = layerIt->last_neuron - layerIt->first_neuron;
    unsigned int layerNeuronShift = layerIt->first_neuron - firstNeuron;
    unsigned int layerWeightShift = layerIt->first_neuron->first_con;

    fann_layer  *prevLayer = layerIt - 1;
    fann_neuron *prevLayerFirstNeuron = prevLayer->first_neuron;
    unsigned int prevLayerNeuronShift = prevLayerFirstNeuron - firstNeuron;
    unsigned int prevLayerSize = prevLayer->last_neuron - prevLayerFirstNeuron;

    if(prevLayerSize < 32)
    {
      gpuann_fann_update_slopes_batch_multineuron_implementation(prevLayerSize,
                                                                 layerSize,
                                                                 &(data.d_trainErrorsArray[layerNeuronShift]),
                                                                 &(data.d_trainSlopes[layerWeightShift]),
                                                                 &(data.d_valuesArray[prevLayerNeuronShift]),
                                                                 data._neuronsCountPerInstance,
                                                                 data._weightsCountPerInstance,
                                                                 instanceCount);
    }
    else
    {
      if(prevLayerSize > 1024 && 0)
      {
        //tested with 4k, it works slower
        gpuann_fann_update_slopes_batch_big_neurons_implementation(prevLayerSize,
                                                                   layerSize,
                                                                   &(data.d_trainErrorsArray[layerNeuronShift]),
                                                                   &(data.d_trainSlopes[layerWeightShift]),
                                                                   &(data.d_valuesArray[prevLayerNeuronShift]),
                                                                   data._neuronsCountPerInstance,
                                                                   data._weightsCountPerInstance,
                                                                   instanceCount);
      }
      else
      {
        gpuann_fann_update_slopes_batch_simple_implementation(prevLayerSize,
                                                              layerSize,
                                                              &(data.d_trainErrorsArray[layerNeuronShift]),
                                                              &(data.d_trainSlopes[layerWeightShift]),
                                                              &(data.d_valuesArray[prevLayerNeuronShift]),
                                                              data._neuronsCountPerInstance,
                                                              data._weightsCountPerInstance,
                                                              instanceCount);
      }
    }
  }
}