
#include <cstdint>
#include <map>
#include <string>

#include "jitify.hpp"

class SpoofCudaContext {

  jitify::JitCache kernel_cache;
  std::map<const std::string, jitify::Program> program_map;
  std::vector<jitify::Program> program_vec;

public:
  static long initialize_cuda(uint32_t device_id);

  static void destroy_cuda(SpoofCudaContext *ctx, uint32_t device_id);

  bool compile_cuda(const std::string &src, const std::string &name);

  //  template <typename T>
  //  bool execute_kernel(const std::string &name, T *in_ptr, long side_ptr,
  //                      long out_ptr, long scalars_ptr, long m, long n,
  //                      long grix);

  template <typename T>
  bool execute_kernel(const std::string &name, T **in_ptr, T **side_ptr,
                      T **out_ptr, T **scalars_ptr, long num_scalars, long m,
                      long n, long grix) {

    auto p = program_map.find(name);
    jitify::Program *pr = &(p->second);

    std::cout << "launching kernel " << name << std::endl;

    if (p != program_map.end()) {

      std::cout << "p->first=" << p->first << std::endl;
      std::cout << "p->second=" << &(p->second) << std::endl;
      std::cout << "p=" << &(p) << std::endl;

      dim3 grid(1);
      dim3 block(m, n);
      std::cout << "launching " << block.x << "x" << block.y
                << "==" << block.x * block.y << " threads" << std::endl;
      using jitify::reflection::type_of;

      double tmp;
      pr->kernel(name)
          //                   .instantiate(type_of(*in_ptr))
          .instantiate(type_of(tmp))
          .configure(grid, block)
          //          .launch(reinterpret_cast<double **>(in_ptr), (double
          //          **)side_ptr,
          //                  (double **)out_ptr, (double **)scalars_ptr, m, n,
          //                  grix, 0, 0);

          .launch(in_ptr, side_ptr, out_ptr, 0, m, n, grix, 0, 0);

    } else {
      std::cout << "kernel " << name << " not found." << std::endl;
    }
    return true;
  }
};
