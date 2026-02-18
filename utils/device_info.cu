#include <cuda_runtime.h>
#include "check_cuda.h"
#include <fstream>
#include <cstdio>

int main(){

    int dev = 0; cudaDeviceProp prop; CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

    std::ofstream ofs("device_info.txt");
    if (!ofs) { fprintf(stderr, "Failed to open device_info.txt\n"); return 1; }

    ofs << "Device " << dev << ": " << prop.name << "\n";
    ofs << "Compute Capability:" << prop.major << "." << prop.minor << "\n";
    ofs << "SM count:" << prop.multiProcessorCount << "\n";
    ofs << "Max Threads Per Block:" << prop.maxThreadsPerBlock << "\n";
    ofs << "Max Threads Dim: [" << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << "]\n";
	ofs << "Max Grid Size: [" << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << "]\n";
	ofs << "Warp Size: " << prop.warpSize << "\n";
	ofs << "Shared Mem Per Block: " << (size_t)prop.sharedMemPerBlock << " bytes\n";
	ofs << "Shared Mem Per Multiprocessor: " << (size_t)prop.sharedMemPerMultiprocessor << " bytes\n";
	ofs << "Registers Per Block: " << prop.regsPerBlock << "\n";
	ofs << "Regs Per Multiprocessor: " << prop.regsPerMultiprocessor << "\n";
	ofs << "Max Blocks Per Multiprocessor: " << prop.maxBlocksPerMultiProcessor << "\n";
	ofs << "Max Threads Per Multiprocessor: " << prop.maxThreadsPerMultiProcessor << "\n";
	ofs << "Memory Clock Rate: " << prop.memoryClockRate << " kHz\n";
	ofs << "Memory Bus Width: " << prop.memoryBusWidth << " bits\n";
	ofs << "Total Global Memory: " << (size_t)prop.totalGlobalMem << " bytes\n";
	ofs << "L2 Cache Size: " << prop.l2CacheSize << " bytes\n";
	ofs << "Concurrent Kernels: " << (prop.concurrentKernels ? "Yes" : "No") << "\n";
	ofs << "Cooperative Launch: " << (prop.cooperativeLaunch ? "Yes" : "No") << "\n";
	ofs << "Max Shared Mem Optin Per Block: " << (size_t)prop.sharedMemPerBlockOptin << " bytes\n";

	ofs.close();
	return 0;
}