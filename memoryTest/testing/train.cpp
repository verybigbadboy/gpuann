
#include <testing/train.h>
#include <base/gpuannDataCreator.h>
#include <gpuannTrain.h>

void testTrainMethods(fann *ann, fann_train_data* train)
{
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

  removegpuann(data);
  removegpuannTrainData(trainData);
}
