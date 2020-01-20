#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <iostream>

#include "typed-def.h"

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


typedef struct
{
	float global_MB;
	int const_KB;
	int share_perblock_KB;
	int text_KB;
	int blocks_per;
	int threads_per;
	int sms;
	int cores_per;
	int id;
	int wrap_size;
	int l2cache_KB;
	float clock_rate_MHz;
	float mem_clock_rate_MHz;
	int mem_bus_width_bit;


	/* data */
}sGpuProp;

class nvidia_dev
{
public:
	nvidia_dev();
	~nvidia_dev();

	int get_devnums();
	bool set_devid(int id);

	bool get_dev_prop(sGpuProp *devProp, int device_id);

private:
	void check_device();

private:
	int m_devices;
	sGpuProp** m_dev_prop_ptr;

};

class CudaTimer
{
public:
	CudaTimer(const std::string& name)
		: name_(name)
	{
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, NULL);
	}

	~CudaTimer()
	{
		cudaEventRecord(stop, NULL);
		cudaEventSynchronize(stop);
		float msecTotal = 0.0f;
		cudaEventElapsedTime(&msecTotal, start, stop);

		std::cout << name_ << ": " << msecTotal << "ms" << std::endl;
		cudaEventDestroy(stop);
		cudaEventDestroy(start);
	}

private:
	std::string name_;
	cudaEvent_t start;
	cudaEvent_t stop;
};
#define CUDA_TIMER(name) CudaTimer cudatimer__(name); 

