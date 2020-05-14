
#include "spoof_native_cuda.h"

#include <iostream>

size_t SpoofCudaContext::initialize_cuda(uint32_t device_id) {
  std::cout << "initializing cuda device " << device_id << std::endl;

  SpoofCudaContext *ctx = new SpoofCudaContext();

  return reinterpret_cast<size_t>(ctx);
}

void SpoofCudaContext::destroy_cuda(SpoofCudaContext *ctx, uint32_t device_id) {
  std::cout << "destroying cuda context " << ctx << " of device " << device_id
            << std::endl;
}

bool SpoofCudaContext::compile_cuda(const std::string &src,
                                    const std::string &name) {
  std::cout << "compiling cuda kernel " << name << std::endl;
  std::cout << src << std::endl;

  jitify::Program program = kernel_cache.program(src);

  // ToDo: agg type
  ops.insert(std::make_pair(
      name, SpoofOperator({std::move(program), SpoofOperator::NO_AGG})));


  return true;
}

