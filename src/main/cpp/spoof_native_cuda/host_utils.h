
#ifndef HOST_UTILS_H
#define HOST_UTILS_H

#pragma once

#include <cuda.h>

#define CHECK_CUDA(call)                                                  \
  do {                                                                    \
    CUresult status = call;                                               \
    if (status != CUDA_SUCCESS) {                                         \
      const char* str;                                                    \
      cuGetErrorName(status, &str);                                       \
      std::cout << "(CUDA) returned " << str;                             \
      std::cout << " (" << __FILE__ << ":" << __LINE__ << ":" << __func__ \
                << "())" << std::endl;                                    \
    }                                                                     \
  } while (0)

#define CHECK_CUDART(call)                                                \
  do {                                                                    \
    cudaError_t status = call;                                            \
    if (status != cudaSuccess) {                                          \
      std::cout << "(CUDART) returned " << cudaGetErrorString(status);    \
      std::cout << " (" << __FILE__ << ":" << __LINE__ << ":" << __func__ \
                << "())" << std::endl;                                    \
    }                                                                     \
  } while (0)

#endif // HOST_UTILS_H