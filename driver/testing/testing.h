#ifndef TESTING_TESTING_H
#define TESTING_TESTING_H

#include <fann.h>

void testSpecificAnn(unsigned int hidenLayersCount, unsigned int neuronsPerLayer, bool testTrainMethods = true);
void fulltest();

#endif // TESTING_TESTING_H