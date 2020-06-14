
#include <cmath>
#include <cstdint>
#include <map>
#include <string>

#include "jitify.hpp"

#include "host_utils.h"

using jitify::reflection::type_of;

struct SpoofOperator {
  enum class AggType : int { NO_AGG, ROW_AGG, COL_AGG, FULL_AGG, NONE };
  enum class AggOp : int {SUM, SUM_SQ, MIN, MAX, NONE };

  jitify::Program program;
  AggType agg_type;
  AggOp agg_op;

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
    T **d_sides;
    T *d_scalars;
    T *d_temp_agg_buf;
    uint32_t N = m * n;

    auto o = ops.find(name);
    if (o != ops.end()) {
      SpoofOperator *op = &(o->second);

      if (num_sides > 0) {
        dev_buf_size = sizeof(T *) * num_sides;
        CHECK_CUDART(cudaMalloc((void **)&d_sides, dev_buf_size));
        CHECK_CUDART(cudaMemcpy(d_sides, side_ptrs, dev_buf_size, cudaMemcpyHostToDevice));
      }

      if (num_scalars > 0) {
        dev_buf_size = sizeof(T) * num_scalars;
        CHECK_CUDART(cudaMalloc((void **)&d_scalars, dev_buf_size));
        CHECK_CUDART(cudaMemcpy(d_scalars, scalars_ptr, dev_buf_size, cudaMemcpyHostToDevice));
      }

      switch (op->agg_type) {
      case SpoofOperator::AggType::FULL_AGG: {
        // num ctas
        int NB = std::ceil((N + NT * 2 - 1) / (NT * 2));
        dim3 grid(NB, 1, 1);
        dim3 block(NT, 1, 1);
        unsigned int shared_mem_size = NT * sizeof(T);

        dev_buf_size = sizeof(T) * NB;
        CHECK_CUDART(cudaMalloc((void **)&d_temp_agg_buf, dev_buf_size));

        // ToDo: connect output to SystemDS logging facilities
        std::cout << "launching spoof cellwise kernel " << name << " with "
                  << NT * NB << " threads in " << NB << " blocks and "
                  << shared_mem_size
                  << " bytes of shared memory for full aggregation of "
                  << N << " elements"
                  << std::endl;

        CHECK_CUDA(op->program.kernel(name)
            .instantiate(type_of(result))
            .configure(grid, block, shared_mem_size)
            .launch(in_ptrs[0], d_sides, d_temp_agg_buf, d_scalars, m, n, grix,
                    0, 0));

        // ToDo: block aggregation
        //        while (NB > 1) {
        //          std::cout << "launching spoof cellwise kernel " << name << "
        //          with "
        //                    << NT * NB << " threads in " << NB << " blocks and
        //                    "
        //                    << shared_mem_size
        //                    << " bytes of shared memory for full aggregation"
        //                    << std::endl;

        //          op->program.kernel(name)
        //              .instantiate(type_of(result))
        //              .configure(grid, block, shared_mem_size)
        //              .launch(d_temp_agg_buf, d_temp_agg_buf, NB);
        //        }

        CHECK_CUDART(cudaMemcpy(&result, d_temp_agg_buf, sizeof(T), cudaMemcpyDeviceToHost));
        CHECK_CUDART(cudaFree(d_temp_agg_buf));
        break;
      }
      case SpoofOperator::AggType::NO_AGG: 
      default: {
        // num ctas
        int NB = std::ceil((N + NT * VT - 1) / (NT * VT));
        dim3 grid(NB, 1, 1);
        dim3 block(NT, 1, 1);
        std::cout << "launching spoof cellwise kernel " << name << " with " << NT * NB
                  << " threads in " << NB << " blocks without aggregation for " 
                  << N << " elements"
                  << std::endl;

        CHECK_CUDA(op->program.kernel(name)
            .instantiate(type_of(result), VT)
            .configure(grid, block)
            .launch(in_ptrs[0], d_sides, out_ptr, d_scalars, m, n, grix, 0, 0));
      }
      }

      if (num_scalars > 0)
        CHECK_CUDART(cudaFree(d_scalars));

      if (num_sides > 0)
        CHECK_CUDART(cudaFree(d_sides));
    } 
    else {
      std::cout << "kernel " << name << " not found." << std::endl;
      return result;
    }
    return result;
  }
};
