#include "neuralNetworkTypeCheck.h"
#include <fann.h>
#include <string>

void check(const fann * ann)
{
  if(ann->network_type != FANN_NETTYPE_LAYER || ann->connection_rate < 1)
  {
    throw new std::string("Neural network must have FANN_NETTYPE_LAYER type\n");
  }

  struct fann_layer *last_layer = ann->last_layer;

  for(struct fann_layer *layer_it = ann->first_layer + 1; layer_it != last_layer; layer_it++)
  {
    struct fann_neuron * last_neuron = layer_it->last_neuron;
    struct fann_neuron * neuron_it = layer_it->first_neuron;

    unsigned int layerActivationFunction = neuron_it->activation_function;
    fann_type    layerSteepness = neuron_it->activation_steepness;
    unsigned int layerNeuronInputCount = neuron_it->last_con - neuron_it->first_con;

    for(neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++)
    {
      if(layerActivationFunction != neuron_it->activation_function)
      {
        throw std::string("Layer must have one type activation function for each neuron\n");
      }
    }

    for(neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++)
    {
      if(layerSteepness != neuron_it->activation_steepness)
      {
        throw std::string("Layer must have one type activation steepness for each neuron\n");
      }
    }

    for(neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++)
    {
      unsigned int count = (neuron_it->last_con - neuron_it->first_con);
      if(layerNeuronInputCount != count && count != 0)
      {
        throw std::string("Layer must have same input count for each neuron\n");
      }
    }
  }
}
