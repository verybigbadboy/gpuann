#include <common/cuda/common.h>
#include <testing/testing.h>

int main()
{
  cudaDeviceInit();

  fulltest();

  cudaDeviceReset();

  return 0;
}
