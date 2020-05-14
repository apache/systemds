
#include "spoof_native_cuda.h"

#include <iostream>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    if (call != CUDA_SUCCESS) {                                                \
      const char *str;                                                         \
      cuGetErrorName(call, &str);                                              \
      std::cout << "(CUDA) returned " << str;                                  \
      std::cout << " (" << __FILE__ << ":" << __LINE__ << ":" << __func__      \
                << "())" << std::endl;                                         \
      return false;                                                            \
    }                                                                          \
  } while (0)

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

  std::cout << "\n bla \n" << name << std::endl;

  jitify::Program program = kernel_cache.program(src);

  std::cout << "\n schlubb \n" << name << std::endl;
  // ToDo: agg type
  ops.insert(std::make_pair(
      name, SpoofOperator({std::move(program), SpoofOperator::NO_AGG})));

  std::cout << "\n blubb \n" << name << std::endl;

  return true;
}

// template <>
// bool SpoofCudaContext::execute_kernel(const std::string &name, double
// *in_ptr,
//                                      long side_ptr, long out_ptr,
//                                      long scalars_ptr, long m, long n,
//                                      long grix) {
//  std::cout << " double instance " << std::endl;
//}

// const char *program_source = "my_program\n"
//                             "template<int N, typename T>\n"
//                             "__global__\n"
//                             "void my_kernel(T* data) {\n"
//                             "    T data0 = data[0];\n"
//                             "    for( int i=0; i<N-1; ++i ) {\n"
//                             "        data[0] *= data0;\n"
//                             "    }\n"
//                             "}\n";
// static jitify::JitCache kernel_cache;
// jitify::Program program = kernel_cache.program(program_source);
//// ...set up data etc.
// double h_data = 5;
// double *d_data;
// cudaMalloc((void **)&d_data, sizeof(double));
// cudaMemcpy(d_data, &h_data, sizeof(double), cudaMemcpyHostToDevice);
////  dim3 grid(1);
////  dim3 block(1);
// using jitify::reflection::type_of;
// program.kernel("my_kernel")
//    .instantiate(3, type_of(*d_data))
//    .configure(grid, block)
//    .launch(d_data);
