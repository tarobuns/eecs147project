// network_kernel.cu
#include "network.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 256

#define cudaCheckError() {                                          \
    cudaError_t e=cudaGetLastError();                                \
    if(e!=cudaSuccess) {                                             \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE);                                          \
    }                                                                \
}

// the kernel for forward propagation
__global__ void PropagateLayerKernel(REAL* Weight, REAL* Input, REAL* Output, int InputSize, int OutputSize, REAL Gain) {
    __shared__ REAL SharedInput[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < InputSize) {
        SharedInput[tid] = Input[tid];
    }
    __syncthreads();

    if (idx < OutputSize) {
        REAL Sum = 0.0;
        for (int j = 0; j < InputSize; j++) {
            Sum += Weight[idx * InputSize + j] * SharedInput[j];
        }
        Output[idx] = 1.0 / (1.0 + exp(-Gain * Sum));
    }
}

// Kernel for backpropagation
__global__ void BackpropagateLayerKernel(REAL* Weight, REAL* Output, REAL* Error, REAL* NextError, int InputSize, int OutputSize, REAL Gain) {
    __shared__ REAL SharedOutput[BLOCK_SIZE];
    __shared__ REAL SharedNextError[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < OutputSize) {
        SharedOutput[tid] = Output[tid];
        SharedNextError[tid] = NextError[tid];
    }
    __syncthreads();

    if (idx < InputSize) {
        REAL Err = 0.0;
        for (int j = 0; j < OutputSize; j++) {
            Err += Weight[j * InputSize + idx] * SharedNextError[j];
        }
        Error[idx] = Gain * SharedOutput[idx] * (1.0 - SharedOutput[idx]) * Err;
    }
}

// using CUDA to propogate through a layer
void PropagateLayerCUDA(NET* Net, LAYER* Lower, LAYER* Upper) {
    int InputSize = Lower->Units + 1; // +1 for the bias
    int OutputSize = Upper->Units;

    // allocate memory on the device
    REAL *d_Weight, *d_Input, *d_Output;
    cudaMalloc(&d_Weight, OutputSize * InputSize * sizeof(REAL)); cudaCheckError();
    cudaMalloc(&d_Input, InputSize * sizeof(REAL)); cudaCheckError();
    cudaMalloc(&d_Output, OutputSize * sizeof(REAL)); cudaCheckError();

    // opy data to the device
    cudaMemcpy(d_Weight, Upper->Weight[1], OutputSize * InputSize * sizeof(REAL), cudaMemcpyHostToDevice); cudaCheckError();
    cudaMemcpy(d_Input, Lower->Output, InputSize * sizeof(REAL), cudaMemcpyHostToDevice); cudaCheckError();

    // define the number of threads and blocks
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (OutputSize + threadsPerBlock - 1) / threadsPerBlock;

    // launch the kernel
    PropagateLayerKernel<<<blocksPerGrid, threadsPerBlock>>>(d_Weight, d_Input, d_Output, InputSize, OutputSize, Net->Gain); cudaCheckError();

    // synchronize the device
    cudaDeviceSynchronize(); cudaCheckError();

    // copy the result back to the host
    cudaMemcpy(Upper->Output + 1, d_Output, OutputSize * sizeof(REAL), cudaMemcpyDeviceToHost); cudaCheckError();

    // free device memory
    cudaFree(d_Weight); cudaCheckError();
    cudaFree(d_Input); cudaCheckError();
    cudaFree(d_Output); cudaCheckError();
}

// function to backpropagate through a layer using CUDA
void BackpropagateLayerCUDA(NET* Net, LAYER* Upper, LAYER* Lower) {
    int InputSize = Lower->Units + 1; // +1 for the bias
    int OutputSize = Upper->Units;

    // allocate memory on the device
    REAL *d_Weight, *d_Output, *d_Error, *d_NextError;
    cudaMalloc(&d_Weight, OutputSize * InputSize * sizeof(REAL)); cudaCheckError();
    cudaMalloc(&d_Output, InputSize * sizeof(REAL)); cudaCheckError();
    cudaMalloc(&d_Error, InputSize * sizeof(REAL)); cudaCheckError();
    cudaMalloc(&d_NextError, OutputSize * sizeof(REAL)); cudaCheckError();

    // copy data to the device
    cudaMemcpy(d_Weight, Upper->Weight[1], OutputSize * InputSize * sizeof(REAL), cudaMemcpyHostToDevice); cudaCheckError();
    cudaMemcpy(d_Output, Lower->Output, InputSize * sizeof(REAL), cudaMemcpyHostToDevice); cudaCheckError();
    cudaMemcpy(d_NextError, Upper->Error + 1, OutputSize * sizeof(REAL), cudaMemcpyHostToDevice); cudaCheckError();

    // define the number of threads and blocks
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (InputSize + threadsPerBlock - 1) / threadsPerBlock;

    // launch the kernel
    BackpropagateLayerKernel<<<blocksPerGrid, threadsPerBlock>>>(d_Weight, d_Output, d_Error, d_NextError, InputSize, OutputSize, Net->Gain); cudaCheckError();

    // synchronize the device
    cudaDeviceSynchronize(); cudaCheckError();

    // copy the result back to the host
    cudaMemcpy(Lower->Error + 1, d_Error, InputSize * sizeof(REAL), cudaMemcpyDeviceToHost); cudaCheckError();

    // free device memory
    cudaFree(d_Weight); cudaCheckError();
    cudaFree(d_Output); cudaCheckError();
    cudaFree(d_Error); cudaCheckError();
    cudaFree(d_NextError); cudaCheckError();
}
