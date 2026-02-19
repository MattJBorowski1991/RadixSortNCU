#ifndef CUDA_TIMER_H
#define CUDA_TIMER_H

#include <cuda_runtime.h>
#include <iostream>

class CudaTimer {
public:
    CudaTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }

    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start() {
        cudaEventRecord(start_);
    }

    void stop() {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
    }

    float elapsedMilliseconds() const {
        float ms;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }

private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
};

#endif // CUDA_TIMER_H