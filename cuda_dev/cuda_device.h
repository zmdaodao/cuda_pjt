#ifndef __CUDA_DEVICE_H__
#define __CUDA_DEVICE_H__


#include <stdio.h> 
#include <stdlib.h> 
#include <math.h>
#include <assert.h>

#include <cuda_runtime.h>


typedef struct 
{
    
    /* data */
}sGpuProp;


// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition)    \
{    cudaError_t error = condition; \
    if(error != cudaSuccess)       \
    {                              \
        printf("cudaMalloc returned %d\n-> %s\n",    \
        static_cast<int>(error), cudaGetErrorString(error));    \
        printf("Result = FAIL\n");    \
        exit(EXIT_FAILURE);         \
    }                               \
}                                   \

void CheckDevice();

int GetDeviceNums();

int SetRunDevice(int id);

int GetDeviceProp();

#endif
