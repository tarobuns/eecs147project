// network.h
#ifndef NETWORK_H
#define NETWORK_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef int BOOL;
typedef int INT;
typedef double REAL;

#define FALSE 0
#define TRUE 1
#define NOT !
#define AND &&
#define OR ||

#define MIN_REAL -HUGE_VAL
#define MAX_REAL +HUGE_VAL
#define MIN(x,y) ((x)<(y) ? (x) : (y))
#define MAX(x,y) ((x)>(y) ? (x) : (y))

#define LO 0.1
#define HI 0.9
#define BIAS 1

#define sqr(x) ((x)*(x))

#define NUM_LAYERS 3
#define N 30
#define M 1

#define FIRST_YEAR 1700
#define NUM_YEARS 280

#define TRAIN_LWB (N)
#define TRAIN_UPB (179)
#define TRAIN_YEARS (TRAIN_UPB - TRAIN_LWB + 1)
#define TEST_LWB (180)
#define TEST_UPB (259)
#define TEST_YEARS (TEST_UPB - TEST_LWB + 1)
#define EVAL_LWB (260)
#define EVAL_UPB (NUM_YEARS - 1)
#define EVAL_YEARS (EVAL_UPB - EVAL_LWB + 1)

typedef struct {
    INT Units;
    REAL* Output;
    REAL* Error;
    REAL** Weight;
    REAL** WeightSave;
    REAL** dWeight;
} LAYER;

typedef struct {
    LAYER** Layer;
    LAYER* InputLayer;
    LAYER* OutputLayer;
    REAL Alpha;
    REAL Eta;
    REAL Gain;
    REAL Error;
} NET;

// declare the global variables
extern REAL Sunspots[NUM_YEARS];
extern REAL Sunspots_[NUM_YEARS];
extern REAL Mean;
extern REAL TrainError;
extern REAL TrainErrorPredictingMean;
extern REAL TestError;
extern REAL TestErrorPredictingMean;



void GenerateNetwork(NET* Net);
void RandomWeights(NET* Net);
void SetInput(NET* Net, REAL* Input);
void GetOutput(NET* Net, REAL* Output);
void PropagateLayer(NET* Net, LAYER* Lower, LAYER* Upper);
void PropagateNet(NET* Net);
void ComputeOutputError(NET* Net, REAL* Target);
void BackpropagateLayer(NET* Net, LAYER* Upper, LAYER* Lower);
void BackpropagateNet(NET* Net);
void AdjustWeights(NET* Net);
void SimulateNet(NET* Net, REAL* Input, REAL* Output, REAL* Target, BOOL Training);
void TrainNet(NET* Net, INT Epochs);
void TestNet(NET* Net);
void EvaluateNet(NET* Net);

// declarations for CUDA functions
void PropagateLayerCUDA(NET* Net, LAYER* Lower, LAYER* Upper);
void BackpropagateLayerCUDA(NET* Net, LAYER* Upper, LAYER* Lower);

#endif // NETWORK_H
