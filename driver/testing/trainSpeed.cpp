#include <testing/trainSpeed.h>
#include <gpuann.h>
#include <cuda_runtime_api.h>
#include <ctime>

void trainMethodsSpeedTestCPU(fann *ann, fann_train_data* train, unsigned int trainingAlgorithm, unsigned int epochCount)
{
  fann *cpunn = fann_copy(ann);
  cpunn->training_algorithm = (fann_train_enum)trainingAlgorithm;

  {
    clock_t start = clock();

    for(unsigned int i = 0; i < epochCount; ++i)
      fann_train_epoch(cpunn, train);

    clock_t ends = clock();
    printf("%10.5f ", (double) (ends - start) / CLOCKS_PER_SEC * 1000.);
  }

  fann_destroy(cpunn);
}

void trainMethodsSpeedTestGPU(fann *ann, fann_train_data* train, unsigned int trainingAlgorithm, unsigned int epochCount)
{
  fann *gpunn = fann_copy(ann);
  gpunn->training_algorithm = (fann_train_enum)trainingAlgorithm;

  {
    cudaEvent_t start, stop;
    float time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    gpuann_fann_parallel_train_on_data(gpunn, train, epochCount);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("%10.5f ", time);
  }

  fann_destroy(gpunn);
}

void trainMethodsSpeedTest(fann *ann, fann_train_data* train)
{
  unsigned int epochCount = 10000;

  trainMethodsSpeedTestGPU(ann, train, FANN_TRAIN_BATCH,     epochCount);
  trainMethodsSpeedTestGPU(ann, train, FANN_TRAIN_QUICKPROP, epochCount);
  trainMethodsSpeedTestGPU(ann, train, FANN_TRAIN_RPROP,     epochCount);
  trainMethodsSpeedTestGPU(ann, train, FANN_TRAIN_SARPROP,   epochCount);
  printf("\n");
}