#ifndef COMMON_FANNHELPER_H
#define COMMON_FANNHELPER_H

#include <fann.h>

fann *createANN(unsigned int num_hiden_layers, unsigned int num_neurons_hidden_per_layer, bool trained);

#endif // COMMON_FANNHELPER_H