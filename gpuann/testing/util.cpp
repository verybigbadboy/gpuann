#include <testing/util.h>

bool isAlmostSame(fann_type& f1, fann_type& f2)
{
  if (f1 == f2)
    return true;

  double minValue = 0.01;

  if(fabs(f1) < minValue && fabs(f2) < minValue || fabs(f1 - f2) < minValue)
    return true;

  float relativeError, maxRelativeError = 0.1; //10 percent error max

  if (fabs(f2) > fabs(f1))
    relativeError = fabs((f1 - f2) / f2);
  else
    relativeError = fabs((f1 - f2) / f1);

  if (relativeError <= maxRelativeError)
    return true;

  return false;
}

static unsigned int failCountPrinted = 0;
static unsigned int maxFailCountPrinted = 30000000;

bool isAlmostSameArrays(fann_type *ar1, fann_type *ar2, unsigned int size, bool printErrors, const std::string& header)
{
  bool result = true;
  unsigned int i = 0;
  failCountPrinted = 0;
  for( ; i < size; ++i)
  {
    if(!isAlmostSame(ar1[i], ar2[i]))
    {
      if(printErrors)
      {
        if(result)
          printf("\n%s\n", header.c_str());
        printf("%6d %10.4f != %10.4f diff %10.4f %7.2f%\n", i, ar1[i], ar2[i], fabs(ar1[i] - ar2[i]), fabs(((ar1[i] - ar2[i])/ar2[i] * 100)) );

        if(failCountPrinted > maxFailCountPrinted)
        {
          int * tmp = 0;
          *tmp = 0;
        }
        failCountPrinted++;
      }
      result = false;
    }
  }
  if(failCountPrinted > 0)
    printf("Total failed: %d\n", failCountPrinted);
  return result;
}

void printannsum(struct fann *ann)
{
  struct fann_layer *last_layer = ann->last_layer;
  for(struct fann_layer *layer_it = ann->first_layer; layer_it != last_layer; layer_it++)
  {
    struct fann_neuron * last_neuron = layer_it->last_neuron;
    struct fann_neuron * neuron_it   = layer_it->first_neuron;
    for(; neuron_it < last_neuron; ++neuron_it)
    {
      if(neuron_it->last_con == neuron_it->first_con)
        printf("A");
      printf("%f ", neuron_it->sum);
    }
    printf("\n");
  }
}

void printannvalue(struct fann *ann)
{
  struct fann_layer *last_layer = ann->last_layer;
  for(struct fann_layer *layer_it = ann->first_layer; layer_it != last_layer; layer_it++)
  {
    struct fann_neuron * last_neuron = layer_it->last_neuron;
    struct fann_neuron * neuron_it   = layer_it->first_neuron;
    for(; neuron_it < last_neuron; ++neuron_it)
    {
      if(neuron_it->last_con == neuron_it->first_con)
        printf("A");
      printf("%f ", neuron_it->value);
    }
    printf("\n");
  }
}

void print2arrays(unsigned int size, fann_type *f, fann_type *s)
{
  printf("ololo\n");
  for(unsigned int i = 0; i < size; ++i)
  {
    printf("%10.3f %10.3f\n", f[i], s[i]);
  }
}