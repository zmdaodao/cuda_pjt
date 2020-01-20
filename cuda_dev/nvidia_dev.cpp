#include "nvidia_dev.h"
#include <memory.h>
#include <iostream>

using namespace std;

// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor)
{
	// Defines for GPU Architecture types (using the SM version to determine
	// the # of cores per SM
	typedef struct {
		int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
		// and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] = {
		{0x30, 192},  // Kepler Generation (SM 3.0) GK10x class
		{0x32, 192},  // Kepler Generation (SM 3.2) GK10x class
		{0x35, 192},  // Kepler Generation (SM 3.5) GK11x class
		{0x37, 192},  // Kepler Generation (SM 3.7) GK21x class
		{0x50, 128},  // Maxwell Generation (SM 5.0) GM10x class
		{0x52, 128},  // Maxwell Generation (SM 5.2) GM20x class
		{0x53, 128},  // Maxwell Generation (SM 5.3) GM20x class
		{0x60, 64},   // Pascal Generation (SM 6.0) GP100 class
		{0x61, 128},  // Pascal Generation (SM 6.1) GP10x class
		{0x62, 128},  // Pascal Generation (SM 6.2) GP10x class
		{0x70, 64},   // Volta Generation (SM 7.0) GV100 class
		{0x72, 64},   // Volta Generation (SM 7.2) GV11b class
		{-1, -1} };

	int index = 0;

	while (nGpuArchCoresPerSM[index].SM != -1)
	{
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
		{
			return nGpuArchCoresPerSM[index].Cores;
		}

		index++;
	}

	// If we don't find the values, we default use the previous one
	// to run properly
	printf(
		"MapSMtoCores for SM %d.%d is undefined."
		"  Default to use %d Cores/SM\n",
		major, minor, nGpuArchCoresPerSM[index - 1].Cores);

	return nGpuArchCoresPerSM[index - 1].Cores;
}

nvidia_dev::nvidia_dev()
{
	m_devices = 0;
	m_dev_prop_ptr = nullptr;

	CUDA_CHECK(cudaGetDeviceCount(&m_devices));
	if (m_devices != 0)
	{
		printf("host has:%d devices\n", m_devices);

		m_dev_prop_ptr = new sGpuProp* [m_devices];
		if (m_dev_prop_ptr == nullptr)
		{
			printf("ErrOr!! new m_dev_prop_ptr error\n");

			return;
		}

		for (int i = 0; i < m_devices; i++)
		{
			m_dev_prop_ptr[i] = new sGpuProp[sizeof(sGpuProp)];
			if (m_dev_prop_ptr[i] == nullptr)
			{
				printf("ErrOr!! new m_dev_prop_ptr[i] error\n");

				return;
			}
			memset(m_dev_prop_ptr[i], 0x00, sizeof(sGpuProp));
		}

		check_device();
		
	}
	else
	{
		printf("There are no available device(s) that support CUDA\n");
	}
}

nvidia_dev::~nvidia_dev()
{
	printf("~nvidia_dev\n");

	if (m_dev_prop_ptr != nullptr)
	{
		delete[]m_dev_prop_ptr ;
	}

	m_devices = 0;
}

int nvidia_dev::get_devnums()
{
	return m_devices;
}

bool nvidia_dev::set_devid(int id)
{
	if (id >= m_devices)
	{
		printf("ERROR!! id out-or-range devices.maybe used id is(0~%d)\n", m_devices-1);

		return false;
	}

	cudaSetDevice(id);
	
	return true;

}

bool nvidia_dev::get_dev_prop(sGpuProp* devProp, int device_id)
{
	if (device_id >= m_devices)
	{
		printf("ERROR!! id out-or-range devices.maybe used id is(0~%d)\n", m_devices - 1);

		return false;
	}

	memcpy(devProp, &m_dev_prop_ptr[device_id], sizeof(sGpuProp));

	return true;
}


void nvidia_dev::check_device()
{
	int32_t dev, driverVersion = 0, runtimeVersion = 0;
	for (dev = 0; dev < m_devices; ++dev)
	{
		cudaSetDevice(dev);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);

		m_dev_prop_ptr[dev]->global_MB = static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f);
		m_dev_prop_ptr[dev]->const_KB = deviceProp.totalConstMem / (1024);
		m_dev_prop_ptr[dev]->share_perblock_KB = deviceProp.sharedMemPerBlock / 1024;
		//m_dev_prop_ptr[dev].text_MB = ;
		m_dev_prop_ptr[dev]->blocks_per = deviceProp.maxThreadsPerMultiProcessor / deviceProp.maxThreadsPerBlock;
		m_dev_prop_ptr[dev]->threads_per = deviceProp.maxThreadsPerBlock;
		m_dev_prop_ptr[dev]->sms = deviceProp.multiProcessorCount;
		m_dev_prop_ptr[dev]->cores_per = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
		m_dev_prop_ptr[dev]->id = dev;
		m_dev_prop_ptr[dev]->l2cache_KB = deviceProp.l2CacheSize / 1024;
		m_dev_prop_ptr[dev]->clock_rate_MHz = static_cast<float>(deviceProp.clockRate * 1e-3f);
		m_dev_prop_ptr[dev]->mem_clock_rate_MHz = static_cast<float>(deviceProp.memoryClockRate * 1e-3f);
		m_dev_prop_ptr[dev]->wrap_size = deviceProp.warpSize;
		m_dev_prop_ptr[dev]->mem_bus_width_bit = deviceProp.memoryBusWidth;

		printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

		cudaDriverGetVersion(&driverVersion);
		cudaRuntimeGetVersion(&runtimeVersion);
		printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
			driverVersion / 1000, (driverVersion % 100) / 10,
			runtimeVersion / 1000, (runtimeVersion % 100) / 10);
		printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
			deviceProp.major, deviceProp.minor);

		char msg[256];
		snprintf(msg, sizeof(msg),
			"  Total amount of global memory:                 %.0f MBytes "
			"(%llu bytes)\n",
			static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
			(unsigned long long)deviceProp.totalGlobalMem);
		printf("%s", msg);

		printf("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
			deviceProp.multiProcessorCount,
			_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
			_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) *
			deviceProp.multiProcessorCount);
		printf(
			"  GPU Max Clock rate:                            %.0f MHz (%0.2f "
			"GHz)\n",
			deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);

		printf("  Memory Clock rate:                             %.0f Mhz\n",
			deviceProp.memoryClockRate * 1e-3f);
		printf("  Memory Bus Width:                              %d-bit\n",
			deviceProp.memoryBusWidth);

		if (deviceProp.l2CacheSize) {
			printf("  L2 Cache Size:                                 %d bytes\n",
				deviceProp.l2CacheSize);
		}

		printf(
			"  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, "
			"%d), 3D=(%d, %d, %d)\n",
			deviceProp.maxTexture1D, deviceProp.maxTexture2D[0],
			deviceProp.maxTexture2D[1], deviceProp.maxTexture3D[0],
			deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
		printf(
			"  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
			deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
		printf(
			"  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d "
			"layers\n",
			deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
			deviceProp.maxTexture2DLayered[2]);

		printf("  Total amount of constant memory:               %lu bytes\n",
			deviceProp.totalConstMem);
		printf("  Total amount of shared memory per block:       %lu bytes\n",
			deviceProp.sharedMemPerBlock);
		printf("  Total number of registers available per block: %d\n",
			deviceProp.regsPerBlock);
		printf("  Warp size:                                     %d\n",
			deviceProp.warpSize);
		printf("  Maximum number of threads per multiprocessor:  %d\n",
			deviceProp.maxThreadsPerMultiProcessor);
		printf("  Maximum number of threads per block:           %d\n",
			deviceProp.maxThreadsPerBlock);
		printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
			deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
			deviceProp.maxThreadsDim[2]);
		printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
			deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
			deviceProp.maxGridSize[2]);
		printf("  Maximum memory pitch:                          %lu bytes\n",
			deviceProp.memPitch);
		printf("  Texture alignment:                             %lu bytes\n",
			deviceProp.textureAlignment);
		printf(
			"  Concurrent copy and kernel execution:          %s with %d copy "
			"engine(s)\n",
			(deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
		printf("  Run time limit on kernels:                     %s\n",
			deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
		printf("  Integrated GPU sharing Host Memory:            %s\n",
			deviceProp.integrated ? "Yes" : "No");
		printf("  Support host page-locked memory mapping:       %s\n",
			deviceProp.canMapHostMemory ? "Yes" : "No");
		printf("  Alignment requirement for Surfaces:            %s\n",
			deviceProp.surfaceAlignment ? "Yes" : "No");
		printf("  Device has ECC support:                        %s\n",
			deviceProp.ECCEnabled ? "Enabled" : "Disabled");

		printf("  Device supports Unified Addressing (UVA):      %s\n",
			deviceProp.unifiedAddressing ? "Yes" : "No");
		printf("  Device supports Compute Preemption:            %s\n",
			deviceProp.computePreemptionSupported ? "Yes" : "No");
		printf("  Supports Cooperative Kernel Launch:            %s\n",
			deviceProp.cooperativeLaunch ? "Yes" : "No");
		printf("  Supports MultiDevice Co-op Kernel Launch:      %s\n",
			deviceProp.cooperativeMultiDeviceLaunch ? "Yes" : "No");
		printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n",
			deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);

		const char* sComputeMode[] = {
			"Default (multiple host threads can use ::cudaSetDevice() with device "
			"simultaneously)",
			"Exclusive (only one host thread in one process is able to use "
			"::cudaSetDevice() with this device)",
			"Prohibited (no host thread can use ::cudaSetDevice() with this "
			"device)",
			"Exclusive Process (many threads in one process is able to use "
			"::cudaSetDevice() with this device)",
			"Unknown",
			NULL };
		printf("  Compute Mode:\n");
		printf("     < %s >\n\n", sComputeMode[deviceProp.computeMode]);
	}

}

