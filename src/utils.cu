// utils.cu
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "utils.h"

// declare the global file pointer
FILE* f;

// initialize random seed
void InitializeRandoms() {
    srand(4711);
}

// generate a random integer between Low and High (inclusive)
INT RandomEqualINT(INT Low, INT High) {
    return rand() % (High - Low + 1) + Low;
}

// read the sunspot data
void ReadSunspotData(const char* filename, REAL* data, int size) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file %s\n", filename);
        exit(1);
    }

    for (int i = 0; i < size; i++) {
        if (fscanf(file, "%lf", &data[i]) != 1) {
            fprintf(stderr, "Error reading data at index %d\n", i);
            fclose(file);
            exit(1);
        }
    }

    fclose(file);
}

// generate a random real number between Low and High
REAL RandomEqualREAL(REAL Low, REAL High) {
    return ((REAL) rand() / RAND_MAX) * (High - Low) + Low;
}

// normalize sunspots data
void NormalizeSunspots() { 
    INT Year;
    REAL Min, Max;

    Min = MAX_REAL;
    Max = MIN_REAL;
    for (Year = 0; Year < NUM_YEARS; Year++) {
        Min = MIN(Min, Sunspots[Year]);
        Max = MAX(Max, Sunspots[Year]);
    }
    Mean = 0;
    for (Year = 0; Year < NUM_YEARS; Year++) {
        Sunspots_[Year] = Sunspots[Year] = ((Sunspots[Year] - Min) / (Max - Min)) * (HI - LO) + LO;
        Mean += Sunspots[Year] / NUM_YEARS;
    }
}

// initialize the application
void InitializeApplication(NET* Net) {
    INT Year, i;
    REAL Out, Err;

    Net->Alpha = 0.5;
    Net->Eta = 0.05;
    Net->Gain = 1;

    // read sunspot data from file
    ReadSunspotData("../data/sunspots.data", Sunspots, NUM_YEARS);

    NormalizeSunspots();
    TrainErrorPredictingMean = 0;
    for (Year = TRAIN_LWB; Year <= TRAIN_UPB; Year++) {
        for (i = 0; i < M; i++) {
            Out = Sunspots[Year + i];
            Err = Mean - Out;
            TrainErrorPredictingMean += 0.5 * sqr(Err);
        }
    }
    TestErrorPredictingMean = 0;
    for (Year = TEST_LWB; Year <= TEST_UPB; Year++) {
        for (i = 0; i < M; i++) {
            Out = Sunspots[Year + i];
            Err = Mean - Out;
            TestErrorPredictingMean += 0.5 * sqr(Err);
        }
    }
    f = fopen("BPN.txt", "w");
}

// finalize the application
void FinalizeApplication(NET* Net) {
    fclose(f);
}

// save weights
void SaveWeights(NET* Net) { // deleted from network.cu
    INT l, i, j;

    for (l = 1; l < NUM_LAYERS; l++) {
        for (i = 1; i <= Net->Layer[l]->Units; i++) {
            for (j = 0; j <= Net->Layer[l - 1]->Units; j++) {
                Net->Layer[l]->WeightSave[i][j] = Net->Layer[l]->Weight[i][j];
            }
        }
    }
}

// restore weights
void RestoreWeights(NET* Net) { // deleted from network.cu
    INT l, i, j;

    for (l = 1; l < NUM_LAYERS; l++) {
        for (i = 1; i <= Net->Layer[l]->Units; i++) {
            for (j = 0; j <= Net->Layer[l - 1]->Units; j++) {
                Net->Layer[l]->Weight[i][j] = Net->Layer[l]->WeightSave[i][j];
            }
        }
    }
}
