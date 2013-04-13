#ifndef TESTING_TESTING_H
#define TESTING_TESTING_H

#include <fann.h>

fann *createSpecificTrainedFann(unsigned int num_hiden_layers, unsigned int num_neurons_hidden_per_layer);
bool runTest(struct fann *ann, fann_type * input);
void runTests(struct fann *ann, bool fullreport = false);
void fulltest();

#endif // TESTING_TESTING_H