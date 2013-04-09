
#include <testing/train.h>
#include <base/gpuannDataCreator.h>

void print2arrays(unsigned int size, fann_type *f, fann_type *s)
{
  printf("ololo\n");
  for(unsigned int i = 0; i < size; ++i)
  {
    printf("%10.3f %10.3f\n", f[i], s[i]);
  }
}

float fann_train_epoch_batch(struct fann *ann, struct fann_train_data *data)
{
  unsigned int i;
  
  fann_reset_MSE(ann);
  
  for(i = 0; i < data->num_data; i++)
  {
    fann_run(ann, data->input[i]);
    fann_compute_MSE(ann, data->output[i]);
    fann_backpropagate_MSE(ann);
    fann_update_slopes_batch(ann, ann->first_layer + 1, ann->last_layer - 1);
  }
  
  fann_update_weights_batch(ann, data->num_data, 0, ann->total_connections);
  
  return fann_get_MSE(ann);
}

float fann_train_epoch_sarprop(struct fann *ann, struct fann_train_data *data)
{
  unsigned int i;
  
  if(ann->prev_train_slopes == NULL)
  {
    fann_clear_train_arrays(ann);
  }
  
  fann_reset_MSE(ann);
  
  for(i = 0; i < data->num_data; i++)
  {
    fann_run(ann, data->input[i]);
    fann_compute_MSE(ann, data->output[i]);
    fann_backpropagate_MSE(ann);
    fann_update_slopes_batch(ann, ann->first_layer + 1, ann->last_layer - 1);
  }
  
  fann_update_weights_sarprop(ann, ann->sarprop_epoch, 0, ann->total_connections);
  
  ++(ann->sarprop_epoch);
  
  return fann_get_MSE(ann);
}

float fann_train_epoch_irpropm(struct fann *ann, struct fann_train_data *data)
{
  unsigned int i;
  
  if(ann->prev_train_slopes == NULL)
  {
    fann_clear_train_arrays(ann);
  }
  
  fann_reset_MSE(ann);
  
  for(i = 0; i < data->num_data; i++)
  {
    fann_run(ann, data->input[i]);
    fann_compute_MSE(ann, data->output[i]);
    fann_backpropagate_MSE(ann);
    fann_update_slopes_batch(ann, ann->first_layer + 1, ann->last_layer - 1);
  }
  
  fann_update_weights_irpropm(ann, 0, ann->total_connections);
  
  return fann_get_MSE(ann);
}

float fann_train_epoch_quickprop(struct fann *ann, struct fann_train_data *data)
{
  unsigned int i;
  
  if(ann->prev_train_slopes == NULL)
  {
    fann_clear_train_arrays(ann);
  }
  
  fann_reset_MSE(ann);
  
  for(i = 0; i < data->num_data; i++)
  {
    fann_run(ann, data->input[i]);
    fann_compute_MSE(ann, data->output[i]);
    fann_backpropagate_MSE(ann);
    fann_update_slopes_batch(ann, ann->first_layer + 1, ann->last_layer - 1);
  }
  fann_update_weights_quickprop(ann, data->num_data, 0, ann->total_connections);
  
  return fann_get_MSE(ann);
}

void test(fann *ann, fann_train_data* train)
{
  gpuann data;
  gpuannTrainData trainData;
  debugGpuann dump[10];
  
  creategpuannTrainData(trainData, train);
  
  (ann->first_layer->last_neuron - 1)->value = 1; ///TODO WHY?
  creategpuann(data, ann);
  loadgpuann(data, ann);
  
  createDump(data, dump[0]);
  
  /*
   *  gpuann_fann_train_epoch_incremental(data, trainData);
   *  createDump(data, dump[1]);
   * 
   *  fann_train(ann, train->input[0], train->output[0]);
   *  fann_train(ann, train->input[1], train->output[1]);
   *  fann_train(ann, train->input[2], train->output[2]);
   *  fann_train(ann, train->input[3], train->output[3]);
   * 
   *  print2arrays(data._neuronsCountPerInstance, ann->train_errors, dump[1].d_trainErrorsArray);
   *  print2arrays(data._weightsCountPerInstance, ann->weights, dump[1].d_weightsArray);
   */
  
  /*
   *  fann_train_epoch_batch(ann, train);
   *  gpuann_fann_train_epoch_batch(data, trainData);
   *  createDump(data, dump[2]);
   *  print2arrays(data._weightsCountPerInstance, ann->weights, dump[2].d_weightsArray);
   *  //print2arrays(data._weightsCountPerInstance, ann->train_slopes, dump[2].d_trainSlopes);
   */
  
  /*
   *  gpuann_fann_train_epoch_sarprop(data, trainData);
   *  createDump(data, dump[3]);
   *  fann_train_epoch_sarprop(ann, train);
   *  print2arrays(data._weightsCountPerInstance, ann->weights, dump[3].d_weightsArray);
   */
  
  /*
   *  gpuann_fann_train_epoch_irpropm(data, trainData);
   *  createDump(data, dump[4]);
   *  fann_train_epoch_irpropm(ann, train);
   *  print2arrays(data._weightsCountPerInstance, ann->weights, dump[4].d_weightsArray);
   */
  /*
   *  gpuann_fann_train_epoch_quickprop(data, trainData);
   *  createDump(data, dump[5]);
   *  fann_train_epoch_quickprop(ann, train);
   *  print2arrays(data._weightsCountPerInstance, ann->weights, dump[5].d_weightsArray);
   */
  removegpuann(data);
  removegpuannTrainData(trainData);
}
