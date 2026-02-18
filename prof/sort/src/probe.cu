#include <iostream>
#include <cuda_runtime.h>
int main(){
  int dev=0; cudaGetDevice(&dev);
  cudaDeviceProp p; cudaGetDeviceProperties(&p, dev);
  std::cout<<p.name<<" maxThreadsPerBlock="<<p.maxThreadsPerBlock<<"\n";
  return 0;
}
