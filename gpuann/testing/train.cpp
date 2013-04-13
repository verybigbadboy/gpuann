
#include <testing/train.h>
#include <testing/util.h>
#include <base/gpuannDataCreator.h>
#include <gpuann.h>
#include <gpuannTrain.h>

#include <string>

void testOneEpochTrain(fann *ann, fann_train_data* train, unsigned int trainingAlgorithm, const std::string &header)
{
  printf("TEST: One Epoch %20s ", header.c_str());
  bool passed = true;

  fann *gpunn = fann_copy(ann);
  fann *cpunn = fann_copy(ann);
  gpunn->training_algorithm = (fann_train_enum)trainingAlgorithm;
  cpunn->training_algorithm = (fann_train_enum)trainingAlgorithm;

  gpuann_fann_train_on_data(gpunn, train, 1);

  fann_train_epoch(cpunn, train);

  fann_type *cpuValuesArray = (fann_type *)malloc(cpunn->total_neurons * sizeof(fann_type));
  fann_type *gpuValuesArray = (fann_type *)malloc(cpunn->total_neurons * sizeof(fann_type));  

  fann *tmpnn = cpunn;
  struct fann_neuron * last_neuron = (tmpnn->last_layer - 1)->last_neuron;
  fann_neuron *neuronsArray = tmpnn->first_layer->first_neuron;
  struct fann_neuron * neuron_it   = tmpnn->first_layer->first_neuron;

  for(; neuron_it != last_neuron; neuron_it++)
  {
    unsigned int currentNeuronShift  = neuron_it - neuronsArray;
    cpuValuesArray[currentNeuronShift] = neuron_it->value;
  }

  tmpnn = gpunn;
  last_neuron = (tmpnn->last_layer - 1)->last_neuron;
  neuronsArray = tmpnn->first_layer->first_neuron;
  neuron_it   = tmpnn->first_layer->first_neuron;

  for(; neuron_it != last_neuron; neuron_it++)
  {
    unsigned int currentNeuronShift  = neuron_it - neuronsArray;
    gpuValuesArray[currentNeuronShift] = neuron_it->value;
  }

  passed &= isAlmostSameArrays(gpuValuesArray, cpuValuesArray, cpunn->total_neurons, true, "VALUES:");
  passed &= isAlmostSameArrays(gpunn->weights, cpunn->weights, cpunn->total_connections, true, "WEIGHTS:");

  fann_destroy(gpunn);
  fann_destroy(cpunn);

  if(passed)
    printf("PASSED\n");
  else
    printf("FAILED\n");
}


void testTrainMethods(fann *ann, fann_train_data* train)
{
  testOneEpochTrain(ann, train, FANN_TRAIN_INCREMENTAL, "Incremental Train");
  testOneEpochTrain(ann, train, FANN_TRAIN_BATCH,       "Batch Train");
  testOneEpochTrain(ann, train, FANN_TRAIN_QUICKPROP,   "QuickProp Train");
  testOneEpochTrain(ann, train, FANN_TRAIN_RPROP,       "RProp Train");
  testOneEpochTrain(ann, train, FANN_TRAIN_SARPROP,     "SarProp Train");
  
  gpuann data;
  gpuannTrainData trainData;
  
  creategpuannTrainData(trainData, train);
  
  (ann->first_layer->last_neuron - 1)->value = 1;
  creategpuann(data, ann);
  loadgpuann(data, ann);
  
  gpuann_fann_train_epoch_incremental(data, trainData);
  gpuann_fann_train_epoch_batch(data, trainData);
  gpuann_fann_train_epoch_sarprop(data, trainData);
  gpuann_fann_train_epoch_irpropm(data, trainData);
  gpuann_fann_train_epoch_quickprop(data, trainData);

  gpuann_fann_train_epoch_batch_parralel(data, trainData);
  gpuann_fann_train_epoch_sarprop_parralel(data, trainData);
  gpuann_fann_train_epoch_irpropm_parralel(data, trainData);
  gpuann_fann_train_epoch_quickprop_parralel(data, trainData);

  savegpuann(data, ann);

  removegpuann(data);
  removegpuannTrainData(trainData);
}
