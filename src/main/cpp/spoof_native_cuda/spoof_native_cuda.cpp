
#include "spoof_native_cuda.h"

#include <filesystem>
#include <iostream>

size_t SpoofCudaContext::initialize_cuda(uint32_t device_id) {
  std::cout << "initializing cuda device " << device_id << std::endl;

  SpoofCudaContext *ctx = new SpoofCudaContext();
  cudaSetDevice(device_id);

  return reinterpret_cast<size_t>(ctx);
}

void SpoofCudaContext::destroy_cuda(SpoofCudaContext *ctx, uint32_t device_id) {
  std::cout << "destroying cuda context " << ctx << " of device " << device_id
            << std::endl;
  delete ctx;
  ctx = nullptr;
  cudaDeviceReset();
}

bool SpoofCudaContext::compile_cuda(const std::string &src,
                                    const std::string &name) {
  std::cout << "compiling cuda kernel " << name << std::endl;
  std::cout << src << std::endl;

  std::cout << "cwd: " << std::filesystem::current_path() << std::endl;

  jitify::Program program = kernel_cache.program(
      src, 0,
      {"-I./src/main/cpp/kernels/", "-I/usr/local/cuda/include",
       "-I/usr/local/cuda/include/cuda/std/detail/libcxx/include/"});

  // ToDo: agg types
  ops.insert(std::make_pair(
      name, SpoofOperator({std::move(program), SpoofOperator::FULL_AGG})));

  return true;
}
