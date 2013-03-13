
#ifndef GPUANNDATA_H
#define GPUANNDATA_H

#include <fann.h>

struct gpuann
{
  const fann * _fann;
  fann_type *weightsArray;
  fann_type *valuesArray;
  fann_type *sumArray;
  fann_type *d_weightsArray;
  fann_type *d_valuesArray;
  fann_type *d_sumArray;
};


#endif // GPUANNDATA_H
