#include <base/gpuannDataCreator.h>

#include <cuda.h>
#include <cuda_runtime.h>

void creategpuann(gpuann& nn, const fann *ann, unsigned int instanceCount)
{
  nn.d_sumArray     = 0;
  nn.d_valuesArray  = 0;
  nn.d_weightsArray = 0;
  nn._sarpropEpoch = 0;
  nn._instanceCount = instanceCount;

  nn._fann = ann;

  unsigned int neuronCount = ann->total_neurons;
  unsigned int weightsCount = ((ann->last_layer - 1)->last_neuron - 1)->last_con;

  nn.h_tmp_valuesArray  = (fann_type *)malloc(neuronCount * sizeof(fann_type));
  nn.h_tmp_sumArray     = (fann_type *)malloc(neuronCount * sizeof(fann_type));

  cudaMalloc((void **)&(nn.d_sumArray),          instanceCount * neuronCount * sizeof(fann_type));
  cudaMalloc((void **)&(nn.d_valuesArray),       instanceCount * neuronCount * sizeof(fann_type));
  cudaMalloc((void **)&(nn.d_trainErrorsArray),  instanceCount * neuronCount * sizeof(fann_type));
  cudaMalloc((void **)&(nn.d_weightsArray),      instanceCount * weightsCount * sizeof(fann_type));
  cudaMalloc((void **)&(nn.d_prevWeightsDeltas), instanceCount * weightsCount * sizeof(fann_type));
  cudaMalloc((void **)&(nn.d_trainSlopes), instanceCount * weightsCount * sizeof(fann_type));
  cudaMalloc((void **)&(nn.d_prevTrainSlopes), instanceCount * weightsCount * sizeof(fann_type));
  cudaMalloc((void **)&(nn.d_prevSteps), instanceCount * weightsCount * sizeof(fann_type));

  cudaMemset(nn.d_sumArray, 0, instanceCount * neuronCount * sizeof(fann_type));
  cudaMemset(nn.d_valuesArray, 0, instanceCount * neuronCount * sizeof(fann_type));
  cudaMemset(nn.d_trainErrorsArray, 0, instanceCount * neuronCount * sizeof(fann_type));
  cudaMemset(nn.d_weightsArray, 0, instanceCount * weightsCount * sizeof(fann_type));
  cudaMemset(nn.d_prevWeightsDeltas, 0, instanceCount * weightsCount * sizeof(fann_type));
  cudaMemset(nn.d_trainSlopes, 0, instanceCount * weightsCount * sizeof(fann_type));
  cudaMemset(nn.d_prevTrainSlopes, 0, instanceCount * weightsCount * sizeof(fann_type));
  cudaMemset(nn.d_prevSteps, 0, instanceCount * weightsCount * sizeof(fann_type));

  nn._neuronsCountPerInstance = neuronCount;
  nn._weightsCountPerInstance = weightsCount;
}

void removegpuann(gpuann& nn)
{
  free(nn.h_tmp_sumArray);
  free(nn.h_tmp_valuesArray);

  cudaFree(nn.d_sumArray);
  cudaFree(nn.d_valuesArray);
  cudaFree(nn.d_weightsArray);
  cudaFree(nn.d_trainErrorsArray);
  cudaFree(nn.d_prevWeightsDeltas);
  cudaFree(nn.d_trainSlopes);
  cudaFree(nn.d_prevTrainSlopes);
  cudaFree(nn.d_prevSteps);
}

void loadgpuann(gpuann& nn, const fann *ann, unsigned int instanceIndex)
{
  unsigned int neuronCount = ann->total_neurons;
  struct fann_neuron *neuronsArray = ann->first_layer->first_neuron;

  struct fann_layer *last_layer = ann->last_layer;
  struct fann_layer *layer_it   = ann->first_layer;

  struct fann_neuron * last_neuron = layer_it->last_neuron;
  struct fann_neuron * neuron_it   = layer_it->first_neuron;

  for(; neuron_it != last_neuron; neuron_it++)
  {
    unsigned int currentNeuronShift  = neuron_it - neuronsArray;
    nn.h_tmp_valuesArray[currentNeuronShift] = neuron_it->value;
    nn.h_tmp_sumArray[currentNeuronShift]    = neuron_it->sum;
  }

  for(layer_it += 1; layer_it != last_layer; layer_it++)
  {
    struct fann_neuron * last_neuron = layer_it->last_neuron;
    struct fann_neuron * neuron_it   = layer_it->first_neuron;
    for(; neuron_it != last_neuron; neuron_it++)
    {
      unsigned int currentNeuronShift  = neuron_it - neuronsArray;
      if(neuron_it->last_con - neuron_it->first_con == 0)
        neuron_it->value = 1;
      nn.h_tmp_valuesArray[currentNeuronShift] = neuron_it->value;
      nn.h_tmp_sumArray[currentNeuronShift]    = neuron_it->sum;
    }
  }

  unsigned int weightsCount = ((ann->last_layer - 1)->last_neuron - 1)->last_con;

  cudaMemcpyAsync(nn.d_sumArray + neuronCount * instanceIndex,     nn.h_tmp_sumArray,    neuronCount  * sizeof(fann_type), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(nn.d_valuesArray + neuronCount * instanceIndex,  nn.h_tmp_valuesArray, neuronCount  * sizeof(fann_type), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(nn.d_weightsArray + weightsCount * instanceIndex, ann->weights,        weightsCount * sizeof(fann_type), cudaMemcpyHostToDevice);

  if(ann->training_algorithm == FANN_TRAIN_RPROP)
  {
    float array[weightsCount];
    for(int i = 0; i < weightsCount; ++i)
      array[i] = ann->rprop_delta_zero;

    cudaMemcpyAsync(nn.d_prevSteps + weightsCount * instanceIndex, array, weightsCount * sizeof(fann_type), cudaMemcpyHostToDevice);
  }
}

void savegpuann(const gpuann& nn, fann *ann, unsigned int instanceIndex)
{
  unsigned int neuronCount = ann->total_neurons;
  unsigned int weightsCount = ((ann->last_layer - 1)->last_neuron - 1)->last_con;

  cudaMemcpyAsync(nn.h_tmp_sumArray,     nn.d_sumArray + neuronCount * instanceIndex,      neuronCount  * sizeof(fann_type), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(nn.h_tmp_valuesArray,  nn.d_valuesArray + neuronCount * instanceIndex,   neuronCount  * sizeof(fann_type), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(ann->weights,          nn.d_weightsArray + weightsCount * instanceIndex, weightsCount * sizeof(fann_type), cudaMemcpyDeviceToHost);
  cudaThreadSynchronize();

  struct fann_neuron *neuronsArray = ann->first_layer->first_neuron;
  struct fann_layer *last_layer = ann->last_layer;
  struct fann_layer *layer_it   = ann->first_layer;

  for(; layer_it != last_layer; layer_it++)
  {
    struct fann_neuron * last_neuron = layer_it->last_neuron;
    struct fann_neuron * neuron_it   = layer_it->first_neuron;
    for(; neuron_it != last_neuron; neuron_it++)
    {
      unsigned int currentNeuronShift  = neuron_it - neuronsArray;
      neuron_it->value = nn.h_tmp_valuesArray[currentNeuronShift];
      neuron_it->sum = nn.h_tmp_sumArray[currentNeuronShift];
    }
  }
}

void gpuann_loadInputs(gpuann& nn, fann_type *d_inputs, unsigned int instanceIndex)
{
  const fann *ann = nn._fann;
  cudaMemcpyAsync(nn.d_valuesArray + nn._neuronsCountPerInstance * instanceIndex,  d_inputs, ann->num_input * sizeof(fann_type), cudaMemcpyDeviceToDevice);
}

fann_type* gpuann_getOutputsDevicePointer(gpuann& nn, unsigned int instanceIndex)
{
  const fann *ann = nn._fann;
  unsigned int outputShift = (ann->last_layer - 1)->first_neuron - ann->first_layer->first_neuron;
  return nn.d_valuesArray + outputShift + nn._neuronsCountPerInstance * instanceIndex;
}

void createDump(gpuann &nn, debugGpuann &dnn)
{
  const fann *ann = nn._fann;

  unsigned int neuronCount = ann->total_neurons;
  unsigned int weightsCount = ((ann->last_layer - 1)->last_neuron - 1)->last_con;

  dnn.d_sumArray = new fann_type[neuronCount];
  dnn.d_valuesArray = new fann_type[neuronCount];
  dnn.d_trainErrorsArray = new fann_type[neuronCount];
  dnn.d_weightsArray = new fann_type[weightsCount];
  dnn.d_prevWeightsDeltas = new fann_type[weightsCount];
  dnn.d_trainSlopes = new fann_type[weightsCount];
  dnn.d_prevTrainSlopes = new fann_type[weightsCount];
  dnn.d_prevSteps = new fann_type[weightsCount];

  cudaThreadSynchronize();

  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }

  cudaMemcpy(dnn.d_sumArray,                  nn.d_sumArray,         neuronCount  * sizeof(fann_type), cudaMemcpyDeviceToHost);
  error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaMemcpy(dnn.d_valuesArray,               nn.d_valuesArray,      neuronCount  * sizeof(fann_type), cudaMemcpyDeviceToHost);
  error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaMemcpy(dnn.d_trainErrorsArray,          nn.d_trainErrorsArray, neuronCount  * sizeof(fann_type), cudaMemcpyDeviceToHost);
  error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaMemcpy(dnn.d_weightsArray,              nn.d_weightsArray,      weightsCount * sizeof(fann_type), cudaMemcpyDeviceToHost);
  error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  cudaMemcpy(dnn.d_prevWeightsDeltas,         nn.d_prevWeightsDeltas, weightsCount * sizeof(fann_type), cudaMemcpyDeviceToHost);
  error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  
  cudaMemcpy(dnn.d_trainSlopes,         nn.d_trainSlopes, weightsCount * sizeof(fann_type), cudaMemcpyDeviceToHost);
  error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  
  cudaMemcpy(dnn.d_prevTrainSlopes,         nn.d_prevTrainSlopes, weightsCount * sizeof(fann_type), cudaMemcpyDeviceToHost);
  error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
  
  cudaMemcpy(dnn.d_prevSteps,         nn.d_prevSteps, weightsCount * sizeof(fann_type), cudaMemcpyDeviceToHost);
  error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and exit
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }

  cudaThreadSynchronize();
}

void creategpuannTrainData(gpuannTrainData &trainData, fann_train_data *train)
{
  unsigned int dataCount = train->num_data;
  unsigned int inputCount = train->num_input;
  unsigned int outputCount = train->num_output;
  trainData._dataCount   = dataCount;
  trainData._inputCount  = inputCount;
  trainData._outputCount = outputCount;

  cudaMalloc((void **)&(trainData.d_input),  dataCount * inputCount  * sizeof(fann_type));
  cudaMalloc((void **)&(trainData.d_output), dataCount * outputCount * sizeof(fann_type));

  for(unsigned int i = 0; i < dataCount; ++i)
  {
    cudaMemcpyAsync(trainData.d_input  + i * inputCount,  train->input[i],  inputCount  * sizeof(fann_type), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(trainData.d_output + i * outputCount, train->output[i], outputCount * sizeof(fann_type), cudaMemcpyHostToDevice);
  }
}

void removegpuannTrainData(gpuannTrainData &trainData)
{
  cudaFree(trainData.d_input);
  cudaFree(trainData.d_output);
}


