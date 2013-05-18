#include <testing/testing.h>
#include <testing/train.h>
#include <testing/run.h>
#include <common/fannHelper.h>
#include <gpuann.h>


void testSpecificAnn(unsigned int hidenLayersCount, unsigned int neuronsPerLayer, bool btestTrainMethods)
{
  printf("Neural network type: %d %d\n", hidenLayersCount, neuronsPerLayer);
  fann *ann = createANN(hidenLayersCount, neuronsPerLayer, true);
  testRunMethods(ann);
  
  if(btestTrainMethods)
  {
    fann_train_data *data = fann_read_train_from_file("xor.data");

    testTrainMethods(ann, data);

    fann_destroy_train(data);
  }
  
  fann_destroy(ann);
}

void fulltest()
{
  for(int i = 1; i < 7; ++i)
  {
    for(int j = 30; j < 1024; j *=2)
    {
      testSpecificAnn(i, j);
    }
  }
}
