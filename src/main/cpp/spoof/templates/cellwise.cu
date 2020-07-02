%TMP%
// CellType: %TYPE%
// AggOp: %AGG_OP_NAME%
// SparseSafe: %SPARSE_SAFE%
// SEQ: %SEQ%
#include "utils.cuh"
#include "agg_ops.cuh"
#include "spoof_utils.cuh"
#include "reduction.cuh"

template<typename T>
struct SpoofCellwiseOp {
   T**b; T* scalars; 
   int m, n, grix_;

   SpoofCellwiseOp(T** b, T* scalars, int m, int n, int grix) : 
       b(b), scalars(scalars), m(m), n(n), grix_(grix) {}

   __device__  __forceinline__ T operator()(T a, int idx) const {
        int rix = idx / n;
        int cix = idx % n;
        int grix = grix_ + rix;
%BODY_dense%
        return %OUT%;
   }
};

template<typename T>
__global__ void %TMP% (T *a, T** b, T* c, T* scalars, int m, int n, int grix) {
   %AGG_OP%<T> agg_op;
   SpoofCellwiseOp<T> spoof_op(b, scalars, m, n, grix);
   %TYPE%<T, %AGG_OP%<T>, SpoofCellwiseOp<T>>(a, c, m, n, %INITIAL_VALUE%, agg_op, spoof_op);
};
