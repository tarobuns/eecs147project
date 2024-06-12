// utils.h
#ifndef UTILS_H
#define UTILS_H

#include "network.h"

// utility function declarations
extern FILE* f;// declare the global file pointer
void InitializeRandoms();
INT RandomEqualINT(INT Low, INT High);
REAL RandomEqualREAL(REAL Low, REAL High);
void NormalizeSunspots();
void InitializeApplication(NET* Net);
void FinalizeApplication(NET* Net);
void SaveWeights(NET* Net);
void RestoreWeights(NET* Net);
void ReadSunspotData(const char* filename, REAL* data, int size);

#endif // UTILS_H
