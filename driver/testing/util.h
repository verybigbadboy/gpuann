#ifndef TESTING_UTIL_H
#define TESTING_UTIL_H

#include <fann.h>
#include <string>

bool isAlmostSame(const fann_type& f1, const fann_type& f2);
bool isAlmostSameArrays(fann_type *ar1, fann_type *ar2, unsigned int size, bool printErrors, const std::string& header);

void printannsum(struct fann *ann);
void printannvalue(struct fann *ann);
void print2arrays(unsigned int size, fann_type *f, fann_type *s);

#endif // TESTING_UTIL_H