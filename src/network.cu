// network.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "network.h"
#include "utils.h"
#include <omp.h> // Ensure this is included for OpenMP

REAL Sunspots[NUM_YEARS];
REAL Sunspots_[NUM_YEARS];
REAL Mean;
REAL TrainError;
REAL TrainErrorPredictingMean;
REAL TestError;
REAL TestErrorPredictingMean;

const int Units[] = { };

// initialization
void GenerateNetwork(NET* Net) {
    INT l, i;

    Net->Layer = (LAYER**) calloc(NUM_LAYERS, sizeof(LAYER*));

    for (l = 0; l < NUM_LAYERS; l++) {
        Net->Layer[l] = (LAYER*) malloc(sizeof(LAYER));

        Net->Layer[l]->Units = Units[l];
        Net->Layer[l]->Output = (REAL*) calloc(Units[l] + 1, sizeof(REAL));
        Net->Layer[l]->Error = (REAL*) calloc(Units[l] + 1, sizeof(REAL));
        Net->Layer[l]->Weight = (REAL**) calloc(Units[l] + 1, sizeof(REAL*));
        Net->Layer[l]->WeightSave = (REAL**) calloc(Units[l] + 1, sizeof(REAL*));
        Net->Layer[l]->dWeight = (REAL**) calloc(Units[l] + 1, sizeof(REAL*));
        Net->Layer[l]->Output[0] = BIAS;

        if (l != 0) {
            for (i = 1; i <= Units[l]; i++) {
                Net->Layer[l]->Weight[i] = (REAL*) calloc(Units[l - 1] + 1, sizeof(REAL));
                Net->Layer[l]->WeightSave[i] = (REAL*) calloc(Units[l - 1] + 1, sizeof(REAL));
                Net->Layer[l]->dWeight[i] = (REAL*) calloc(Units[l - 1] + 1, sizeof(REAL));
            }
        }
    }
    Net->InputLayer = Net->Layer[0];
    Net->OutputLayer = Net->Layer[NUM_LAYERS - 1];
    Net->Alpha = 0.9;
    Net->Eta = 0.25;
    Net->Gain = 1;
}

// initialize weights randomly
void RandomWeights(NET* Net) {
    INT l, i, j;

    for (l = 1; l < NUM_LAYERS; l++) {
        for (i = 1; i <= Net->Layer[l]->Units; i++) {
            for (j = 0; j <= Net->Layer[l - 1]->Units; j++) {
                Net->Layer[l]->Weight[i][j] = RandomEqualREAL(-0.5, 0.5);
            }
        }
    }
}

// set input values
void SetInput(NET* Net, REAL* Input) {
    INT i;

    for (i = 1; i <= Net->InputLayer->Units; i++) {
        Net->InputLayer->Output[i] = Input[i - 1];
    }
}

// get output values
void GetOutput(NET* Net, REAL* Output) {
    INT i;

    for (i = 1; i <= Net->OutputLayer->Units; i++) {
        Output[i - 1] = Net->OutputLayer->Output[i];
    }
}


// propagate signals through a layer
void PropagateLayer(NET* Net, LAYER* Lower, LAYER* Upper) {
    INT i, j;
    REAL Sum;

    for (i = 1; i <= Upper->Units; i++) {
        Sum = 0;
        for (j = 0; j <= Lower->Units; j++) {
            Sum += Upper->Weight[i][j] * Lower->Output[j];
        }
        Upper->Output[i] = 1 / (1 + exp(-Net->Gain * Sum));
    }
}

// propagate signals through the network
void PropagateNet(NET* Net) {
    INT l;

    for (l = 0; l < NUM_LAYERS - 1; l++) {
        PropagateLayerCUDA(Net, Net->Layer[l], Net->Layer[l + 1]);
    }
}

// compute output error
void ComputeOutputError(NET* Net, REAL* Target) {
    INT i;
    REAL Out, Err;

    Net->Error = 0;
    for (i = 1; i <= Net->OutputLayer->Units; i++) {
        Out = Net->OutputLayer->Output[i];
        Err = Target[i - 1] - Out;
        Net->OutputLayer->Error[i] = Net->Gain * Out * (1 - Out) * Err;
        Net->Error += 0.5 * sqr(Err);
    }
}

// backpropagate error through a layer
void BackpropagateLayer(NET* Net, LAYER* Upper, LAYER* Lower) {
    INT i, j;
    REAL Out, Err;

    for (i = 1; i <= Lower->Units; i++) {
        Out = Lower->Output[i];
        Err = 0;
        for (j = 1; j <= Upper->Units; j++) {
            Err += Upper->Weight[j][i] * Upper->Error[j];
        }
        Lower->Error[i] = Net->Gain * Out * (1 - Out) * Err;
    }
}

// backpropagate error through the network
void BackpropagateNet(NET* Net) {
    INT l;

    for (l = NUM_LAYERS - 1; l > 1; l--) {
        BackpropagateLayerCUDA(Net, Net->Layer[l], Net->Layer[l - 1]);
    }
}

// adjust weights based on errors
void AdjustWeights(NET* Net) {
    INT l, i, j;
    REAL Out, Err, dWeight;

    for (l = 1; l < NUM_LAYERS; l++) {
        for (i = 1; i <= Net->Layer[l]->Units; i++) {
            for (j = 0; j <= Net->Layer[l - 1]->Units; j++) {
                Out = Net->Layer[l - 1]->Output[j];
                Err = Net->Layer[l]->Error[i];
                dWeight = Net->Layer[l]->dWeight[i][j];
                Net->Layer[l]->Weight[i][j] += Net->Eta * Err * Out + Net->Alpha * dWeight;
                Net->Layer[l]->dWeight[i][j] = Net->Eta * Err * Out;
            }
        }
    }
}

// simulate the network
void SimulateNet(NET* Net, REAL* Input, REAL* Output, REAL* Target, BOOL Training) {
    SetInput(Net, Input);
    PropagateNet(Net);
    GetOutput(Net, Output);

    ComputeOutputError(Net, Target);
    if (Training) {
        BackpropagateNet(Net);
        AdjustWeights(Net);
    }
}

// train the network
void TrainNet(NET* Net, INT Epochs) {
    INT Year, n;
    REAL Output[M];

    #pragma omp parallel for private(Year, Output) schedule(dynamic)
    for (n = 0; n < Epochs * TRAIN_YEARS; n++) {
        Year = RandomEqualINT(TRAIN_LWB, TRAIN_UPB);
        SimulateNet(Net, &(Sunspots[Year - N]), Output, &(Sunspots[Year]), TRUE);
    }
}

// test the network
void TestNet(NET* Net) {
    INT Year;
    REAL Output[M];

    TrainError = 0;
    for (Year = TRAIN_LWB; Year <= TRAIN_UPB; Year++) {
        SimulateNet(Net, &(Sunspots[Year - N]), Output, &(Sunspots[Year]), FALSE);
        TrainError += Net->Error;
    }
    TestError = 0;
    for (Year = TEST_LWB; Year <= TEST_UPB; Year++) {
        SimulateNet(Net, &(Sunspots[Year - N]), Output, &(Sunspots[Year]), FALSE);
        TestError += Net->Error;
    }
    fprintf(f, "\nNMSE is %0.3f on Training Set and %0.3f on Test Set",
        TrainError / TrainErrorPredictingMean,
        TestError / TestErrorPredictingMean);
}

// evaluate the network
void EvaluateNet(NET* Net) {
    INT Year;
    REAL Output[M];
    REAL Output_[M];

    fprintf(f, "\n\n\n");
    fprintf(f, "Year    Sunspots    Open-Loop Prediction    Closed-Loop Prediction\n");
    fprintf(f, "\n");
    for (Year = EVAL_LWB; Year <= EVAL_UPB; Year++) {
        SimulateNet(Net, &(Sunspots[Year - N]), Output, &(Sunspots[Year]), FALSE);
        SimulateNet(Net, &(Sunspots_[Year - N]), Output_, &(Sunspots_[Year]), FALSE);
        Sunspots_[Year] = Output_[0];
        fprintf(f, "%d       %0.3f                   %0.3f                     %0.3f\n",
            FIRST_YEAR + Year,
            Sunspots[Year],
            Output[0],
            Output_[0]);
    }
}
