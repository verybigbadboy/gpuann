#include <common/cuda/common.h>
#include <testing/testing.h>

int main(int argc, char** argv)
{
  cudaDeviceInit();

  fulltest();

  cudaDeviceReset();

  return 0;
}
