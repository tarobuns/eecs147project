//main.cu
#include <iostream>
#include <chrono>
#include "network.h"
#include "utils.h"

#define MAX_EPOCHS 1000
#define EARLY_STOPPING_PATIENCE 10

int main() {
    NET Net;
    BOOL Stop;
    REAL MinTestError;
    int epoch = 0;
    int no_improvement_epochs = 0;

    std::cout << "Starting network simulation..." << std::endl;

    // initialize random seed for reproducibility
    InitializeRandoms();
    std::cout << "Done initializing randoms" << std::endl;

    // generate the network structure
    GenerateNetwork(&Net);
    std::cout << "Done generating network" << std::endl;

    // initialize weights randomly
    RandomWeights(&Net);
    std::cout << "Done initializing weights randomly" << std::endl;

    // initialize the application-specific settings
    InitializeApplication(&Net);
    std::cout << "Done initializing application-specific settings" << std::endl;

    Stop = FALSE;
    MinTestError = MAX_REAL;

    auto start_total = std::chrono::high_resolution_clock::now();

    do {
        auto start_epoch = std::chrono::high_resolution_clock::now();

        std::cout << "Beginning to train network for epoch " << epoch << std::endl;
        // train the network
        TrainNet(&Net, 10);
        std::cout << "Done training network for epoch " << epoch << std::endl;

        // test the network to check performance
        TestNet(&Net);
        std::cout << "Done testing network for epoch " << epoch << std::endl;

        auto end_epoch = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> epoch_duration = end_epoch - start_epoch;
        std::cout << "Epoch " << epoch << " completed in " << epoch_duration.count() << " seconds." << std::endl;

        // print NMSE
        std::cout << "NMSE is " << (TrainError / TrainErrorPredictingMean)
                  << " on Training Set and " << (TestError / TestErrorPredictingMean);

        // save weights if the test error improves
        if (TestError < MinTestError) {
            std::cout << " - saving Weights ..." << std::endl;
            MinTestError = TestError;
            SaveWeights(&Net);
            no_improvement_epochs = 0;  // Reset the counter for early stopping
        }
        // stop training if the test error worsens significantly or early stopping condition is met
        else if (TestError > 1.2 * MinTestError) {
            std::cout << " - stopping Training and restoring Weights ..." << std::endl;
            Stop = TRUE;
            RestoreWeights(&Net);
        } else {
            no_improvement_epochs++;
            std::cout << std::endl;
            if (no_improvement_epochs >= EARLY_STOPPING_PATIENCE) {
                std::cout << " - early stopping due to no improvement ..." << std::endl;
                Stop = TRUE;
            }
        }

        // increment epoch count
        epoch += 10;

        // stop if maximum epochs are reached
        if (epoch >= MAX_EPOCHS) {
            std::cout << " - maximum epochs reached, stopping Training ..." << std::endl;
            Stop = TRUE;
        }

    } while (!Stop);

    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration = end_total - start_total;

    // final test and evaluation of the network
    TestNet(&Net);
    std::cout << "Done with final test" << std::endl;
    EvaluateNet(&Net);
    std::cout << "Done with final evaluation" << std::endl;

    // finalize the application
    FinalizeApplication(&Net);
    std::cout << "Done finalizing application" << std::endl;

    std::cout << "Total Execution Time: " << total_duration.count() << " seconds" << std::endl;

    return 0;
}
