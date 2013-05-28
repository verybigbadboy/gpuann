
#ifndef CONFIGURATION_H
#define CONFIGURATION_H

static unsigned int parallelTrainInstanceCountMax = 32;
static bool         backpropationMultiNeuronImplementationEnabled = true;
static bool         backpropationParallelImplementationEnabled = true;
static unsigned int backpropationParallelImplementationBeginCount = 256;
static bool         straightforwardSmallNeuronImplementationEnabled = true;
static bool         updateSlopesBatchMultiNeuronImplementationEnabled = true;
static bool         updateSlopesBatchBigNeuronImplementationEnabled = false;
static unsigned int updateSlopesBatchMultiNeuronImplementationBeginCount = 1024;
static bool         minimalThreadCountPerBlockOptimization = true;


#endif // CONFIGURATION_H
