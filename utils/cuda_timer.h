// Simple RAII CUDA event timer
#ifndef RADIX_SORT_NCU_CUDA_TIMER_H
#define RADIX_SORT_NCU_CUDA_TIMER_H

#include <cuda_runtime.h>

struct CudaTimer {
    cudaEvent_t start;
    cudaEvent_t stop;

    CudaTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // record start event (default stream)
    void startEvent(cudaStream_t stream = 0) {
        cudaEventRecord(start, stream);
    }

    // record stop event (default stream)
    void stopEvent(cudaStream_t stream = 0) {
        cudaEventRecord(stop, stream);
    }

    // wait for the stop event to complete
    void sync() {
        cudaEventSynchronize(stop);
    }

    // get elapsed milliseconds between last start/stop
    float elapsedMs() const {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

#endif // RADIX_SORT_NCU_CUDA_TIMER_H
