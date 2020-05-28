
#include <cstdint>
#include <map>
#include <string>

#include "jitify.hpp"

using jitify::reflection::type_of;

struct SpoofOperator {
  enum AggType { NO_AGG, ROW_AGG, COL_AGG, FULL_AGG };

  jitify::Program program;
  AggType agg_type;
};

class SpoofCudaContext {

  jitify::JitCache kernel_cache;
  std::map<const std::string, SpoofOperator> ops;

public:
    // ToDo: make launch config more adaptive
    // num threads
    const int NT = 256;

    // values / thread
    const int VT = 4;


  static size_t initialize_cuda(uint32_t device_id);

  static void destroy_cuda(SpoofCudaContext *ctx, uint32_t device_id);

  bool compile_cuda(const std::string &src, const std::string &name);

  template <typename T>
  T execute_kernel(const std::string &name, T **in_ptrs, int num_inputs,
                   T **side_ptrs, int num_sides, T *out_ptr, T *scalars_ptr,
                   int num_scalars, int m, int n, int grix) {

    T result = 0.0;
    size_t dev_buf_size;
    T** d_sides;
    T* d_scalars;

    auto o = ops.find(name);
    if (o != ops.end()) {
        SpoofOperator* op = &(o->second);

        // num ctas
        int NB = std::ceil((m * n + NT * VT - 1) / (NT * VT));
        dim3 grid(NB, 1, 1);
        dim3 block(NT, 1, 1);
        std::cout << "launching spoof kernel " << name << " with " << NT * NB << " threads in " << NB << " blocks" << std::endl;

        if (num_sides > 0) {
            dev_buf_size = sizeof(T*) * num_sides;
            cudaMalloc((void**)&d_sides, dev_buf_size);
            cudaMemcpy(d_sides, side_ptrs, dev_buf_size, cudaMemcpyHostToDevice);
        }


        if (num_scalars > 0 || op->agg_type == SpoofOperator::FULL_AGG) {
            if (op->agg_type == SpoofOperator::FULL_AGG)
                dev_buf_size = sizeof(T) * num_scalars + 1;
            else
                dev_buf_size = sizeof(T) * num_scalars;

            cudaMalloc((void**)&d_scalars, dev_buf_size);
            cudaMemcpy(d_scalars, scalars_ptr, dev_buf_size, cudaMemcpyHostToDevice);
        }

        op->program.kernel(name)
            .instantiate(type_of(result), VT)
            .configure(grid, block)
            .launch(in_ptrs[0], d_sides, out_ptr, d_scalars, m, n, grix, 0, 0);

        if (op->agg_type == SpoofOperator::FULL_AGG)
            cudaMemcpy(&result, (scalars_ptr + num_scalars * sizeof(T)), sizeof(T), cudaMemcpyDeviceToHost);

        if(num_scalars > 0)
            cudaFree(d_scalars);

        if(side_ptrs > 0)
            cudaFree(d_sides);

    } 
    else {
      std::cout << "kernel " << name << " not found." << std::endl;
      return result;
    }
    return result;
  }
};
