#ifndef TESTING_TESTING_H
#define TESTING_TESTING_H

#include <fann.h>

void printannsum(struct fann *ann);
void printannvalue(struct fann *ann);
void print2arrays(unsigned int size, fann_type *f, fann_type *s);
fann *createSpecificTrainedFann(unsigned int num_hiden_layers, unsigned int num_neurons_hidden_per_layer);
bool runTest(struct fann *ann, fann_type * input, const char * testName, bool fullreport = false);
void runTests(struct fann *ann, bool fullreport = false);
void fulltest();

#endif // TESTING_TESTING_H