/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/**********************************
When updating a kernel or adding a new one,
please compile the ptx file and commit it:
nvcc -w -ptx -arch=sm_30 --std c++11 SystemML.cu
***********************************/

#include <cfloat>
#include <cmath>

extern "C" __global__ void double2float_f(double *A, float *ret, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < N) {
    // TODO: Use __double2float_rd or __double2float_rn  or __double2float_ru or
    // __double2float_rz after
    ret[tid] = (float)A[tid];
  }
}

extern "C" __global__ void float2double_f(float *A, double *ret, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < N) {
    ret[tid] = (double)A[tid];
  }
}

/**
 * This method performs an im2col operation on sparse input image
 *
 * @params inVal input val pointer
 * @params inRowPtr input row pointer
 * @params colInd input col index pointer
 * @param ret output matrix allocated on the GPU
 * @param NCHW  value of N*C*H*W
 * @param CHW value of C*H*W
 * @param HW value of H*W
 * @param W image height
 * @param R filter height
 * @param S filter width
 * @param P height of conv2d output
 * @param Q width of conv2d output
 * @param PQ value of P*Q
 * @param RS value of R*S
 * @param NPQ value of N*P*Q
 * @param stride_h stride height
 * @param stride_w stride width
 * @param pad_h padding height
 * @param pad_w padding width
 */
template <typename T>
__device__ void sparse_dense_im2col(T *inVal, int *inRowPtr, int *colInd, T *ret,
  int nnz, int N, int CHW, int HW, int W,
  int R, int S, int P, int Q,  int PQ, int RS, int NPQ,
  int stride_h, int stride_w, int pad_h, int pad_w) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < nnz) {
  	T value = inVal[tid];
  	int n = 0;
    while (inRowPtr[n+1] <= tid) {
      n++;
    }
  	int chw = colInd[tid];
  	int c = chw / HW;
  	int hw = chw % HW;
  	int h = hw / W;
  	int w = hw % W;
  	
  	// Constraints: for(int r = 0; r < R; r++) { if(0 <= p && p < P && (h - r + pad_h) % stride_h == 0) { ... } }
	// Constraint 1: p >= 0 and p = (h - r + pad_h)  / stride_h
	// Therefore,  r <= h + pad_h 
	// Constraint 2: p < P and p = (h - r + pad_h)  / stride_h
	// Therefore,  h + pad_h - P*stride_h < r
	// Math.max(0, h + pad_h - P*stride_h + 1) <= r <= Math.min(R-1, h + pad_h)
	int rMin = max(0, h + pad_h - P*stride_h + 1);
	int rMax = min(R-1, h + pad_h);
	int sMin = max(0, w + pad_w - Q*stride_w + 1);
	int sMax = min(S-1, w + pad_w);
	// Constraint 3: (h - r + pad_h) % stride_h == 0
	while((h - rMin + pad_h) % stride_h != 0 && rMin <= rMax) rMin++;
	while((w - sMin + pad_w) % stride_w != 0 && sMin <= sMax) sMin++;
	
	for(int r = rMin; r <= rMax; r += stride_h) {
		// Only append value if h == h, where h = (r - pad_h) + p*stride_h and 0 <= p < P
		// Therefore, p = (h - r + pad_h)  / stride_h. Use the same logic for q.
		int p = (h - r + pad_h)  / stride_h;
		int npQ = n*PQ + p*Q;
		int outRowIndex = c*RS + r*S;
		for(int s = sMin; s <= sMax; s += stride_w) {
			int q = (w - s + pad_w)  / stride_w;
			// chw -> [crs, npq]
			ret[(outRowIndex + s)*NPQ + npQ + q] = value;
		}
	}
  }
}

extern "C" __global__ void sparse_dense_im2col_d(double *inVal, int *inRowPtr, int *colInd, double *ret,
  int nnz, int N, int CHW, int HW, int W,
  int R, int S, int P, int Q,  int PQ, int RS, int NPQ,
  int stride_h, int stride_w, int pad_h, int pad_w) {
  sparse_dense_im2col(inVal, inRowPtr, colInd, ret, nnz, N, CHW, HW, W, R, S, P, Q, PQ, RS, NPQ, stride_h, stride_w, pad_h, pad_w);
}

extern "C" __global__ void sparse_dense_im2col_f(float *inVal, int *inRowPtr, int *colInd, float *ret,
  int nnz, int N, int CHW, int HW, int W,
  int R, int S, int P, int Q, int PQ, int RS, int NPQ,
  int stride_h, int stride_w, int pad_h, int pad_w) {
  sparse_dense_im2col(inVal, inRowPtr, colInd, ret, nnz, N, CHW, HW, W, R, S, P, Q, PQ, RS, NPQ, stride_h, stride_w, pad_h, pad_w);
}

/**
 * This method performs an im2col operation on dense input image
 *
 * @param input input matrix allocated on the GPU
 * @param ret output matrix allocated on the GPU
 * @param NCHW  value of N*C*H*W
 * @param CHW value of C*H*W
 * @param HW value of H*W
 * @param W image height
 * @param R filter height
 * @param S filter width
 * @param P height of conv2d output
 * @param Q width of conv2d output
 * @param PQ value of P*Q
 * @param RS value of R*S
 * @param NPQ value of N*P*Q
 * @param stride_h stride height
 * @param stride_w stride width
 * @param pad_h padding height
 * @param pad_w padding width
 */
template <typename T>
__device__ void dense_dense_im2col(T *input, T *ret,
  int NCHW, int CHW, int HW, int W,
  int R, int S, int P, int Q,  int PQ, int RS, int NPQ,
  int stride_h, int stride_w, int pad_h, int pad_w) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < NCHW) {
  	T value = input[tid];
  	int n = tid / CHW;
  	int chw = tid % CHW;
  	int c = chw / HW;
  	int hw = chw % HW;
  	int h = hw / W;
  	int w = hw % W;
  	
  	// Constraints: for(int r = 0; r < R; r++) { if(0 <= p && p < P && (h - r + pad_h) % stride_h == 0) { ... } }
	// Constraint 1: p >= 0 and p = (h - r + pad_h)  / stride_h
	// Therefore,  r <= h + pad_h 
	// Constraint 2: p < P and p = (h - r + pad_h)  / stride_h
	// Therefore,  h + pad_h - P*stride_h < r
	// Math.max(0, h + pad_h - P*stride_h + 1) <= r <= Math.min(R-1, h + pad_h)
	int rMin = max(0, h + pad_h - P*stride_h + 1);
	int rMax = min(R-1, h + pad_h);
	int sMin = max(0, w + pad_w - Q*stride_w + 1);
	int sMax = min(S-1, w + pad_w);
	// Constraint 3: (h - r + pad_h) % stride_h == 0
	while((h - rMin + pad_h) % stride_h != 0 && rMin <= rMax) rMin++;
	while((w - sMin + pad_w) % stride_w != 0 && sMin <= sMax) sMin++;
	
	for(int r = rMin; r <= rMax; r += stride_h) {
		// Only append value if h == h, where h = (r - pad_h) + p*stride_h and 0 <= p < P
		// Therefore, p = (h - r + pad_h)  / stride_h. Use the same logic for q.
		int p = (h - r + pad_h)  / stride_h;
		int npQ = n*PQ + p*Q;
		int outRowIndex = c*RS + r*S;
		for(int s = sMin; s <= sMax; s += stride_w) {
			int q = (w - s + pad_w)  / stride_w;
			// chw -> [crs, npq]
			ret[(outRowIndex + s)*NPQ + npQ + q] = value;
		}
	}
  }
}

extern "C" __global__ void dense_dense_im2col_d(double *input, double *ret,
  int NCHW, int CHW, int HW, int W,
  int R, int S, int P, int Q,  int PQ, int RS, int NPQ,
  int stride_h, int stride_w, int pad_h, int pad_w) {
  dense_dense_im2col(input, ret, NCHW, CHW, HW, W, R, S, P, Q, PQ, RS, NPQ, stride_h, stride_w, pad_h, pad_w);
}

extern "C" __global__ void dense_dense_im2col_f(float *input, float *ret,
  int NCHW, int CHW, int HW, int W,
  int R, int S, int P, int Q, int PQ, int RS, int NPQ,
  int stride_h, int stride_w, int pad_h, int pad_w) {
  dense_dense_im2col(input, ret, NCHW, CHW, HW, W, R, S, P, Q, PQ, RS, NPQ, stride_h, stride_w, pad_h, pad_w);
}

/**
 * This method performs a reorg operation of matrix with dimensions [K, NPQ]
 * and returns a matrix with dimensions [N, KPQ]
 *
 * @param knpqPtr input matrix allocated on the GPU
 * @param ret output matrix allocated on the GPU
 * @param NKPQ length of input and output matrix
 * @param NPQ the number of columns of input matrix
 * @param KPQ the number of columns of output matrix
 * @param PQ value of P*Q
 */
template <typename T>
__device__ void reorg_knpq(T *knpqPtr, T *ret, int NKPQ, int NPQ, int KPQ, int PQ) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < NKPQ) {
  	int k = tid / NPQ;
  	int npq = tid % NPQ;
  	int n = npq / PQ;
  	int pq = npq % PQ;
    ret[n*KPQ + k*PQ + pq] = knpqPtr[tid];
  }
}

extern "C" __global__ void reorg_knpq_d(double *knpqPtr, double *ret, int NKPQ, int NPQ, int KPQ, int PQ) {
  reorg_knpq(knpqPtr, ret, NKPQ, NPQ, KPQ, PQ);
}

extern "C" __global__ void reorg_knpq_f(float *knpqPtr, float *ret, int NKPQ, int NPQ, int KPQ, int PQ) {
  reorg_knpq(knpqPtr, ret, NKPQ, NPQ, KPQ, PQ);
}

/**
 * Performs a slice operation where the input matrix is sparse and the output
 * matrix is dense.
 * This function avoids unnecessary sparse to dense conversion of the input
 * matrix.
 * Parallelization: rows of output matrix.
 *
 * @params inVal input val pointer
 * @params inRowPtr input row pointer
 * @params colInd input col index pointer
 * @params ret dense output pointer
 * @param rl row lower
 * @param ru row upper
 * @param cl column lower
 * @param cu column upper
 * @param retClen number of columns of output matrix
 */
template <typename T>
__device__ void slice_sparse_dense_row(T *inVal, int *inRowPtr, int *colInd,
                                       T *ret, int rl, int ru, int cl, int cu,
                                       int retClen) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int rowIndex = index + rl;
  if (rowIndex <= ru) {
    /*
     * TODO: Alternative approach: use dynamic parallelism. We are skipping this
*for now to avoid
     * the complexity of two-step separate compilation and linking process.
     *
     * extern "C"
     * __global__ void slice_sparse_dense_row_helper(double* inVal, int*
*inRowPtr, int* colInd, double* ret,
     *     int rl, int ru, int cl, int cu, int retClen, int start, int end, int
*index) {
     *  int i = blockIdx.x * blockDim.x + threadIdx.x + start;
     * 	// Only slice if the index falls into the given range
     * 	if(i < end && cl <= colInd[i] && colInd[i] <= cu) {
     * 		ret[ index*retClen + (colInd[i] - cl) ] = inVal[i];
     * 	}
     * }
     *
     * int size = inRowPtr[rowIndex+1] - inRowPtr[rowIndex];
     * double numThreads = (double)min(size, MAX_NUM_THREADS_CHILD_KERNEL);
     * slice_sparse_dense_row_helper
     * <<< ceil(numThreads/MAX_NUM_THREADS_CHILD_KERNEL), MAX_NUM_THREADS_CHILD_KERNEL>>>
     * (inVal, inRowPtr, colInd, ret, rl, ru, cl, cu, retClen, inRowPtr[rowIndex],
     *	inRowPtr[rowIndex+1], index);
     *
     * Two-step compilation and linking process in JCudaKernels's constructor:
     * cuLinkAddFile(linkState, CUjitInputType.CU_JIT_INPUT_LIBRARY,
     * "/usr/local/cuda/lib64/libcudadevrt.a", jitOptions);
     */
    // Iterate over elements of the row 'rowIndex'.
    for (int i = inRowPtr[rowIndex]; i < inRowPtr[rowIndex + 1]; i++) {
      // Only slice if the index falls into the given range
      if (cl <= colInd[i] && colInd[i] <= cu) {
        ret[index * retClen + (colInd[i] - cl)] = inVal[i];
      }
    }
  }
}

extern "C" __global__ void slice_sparse_dense_row_d(double *inVal,
                                                    int *inRowPtr, int *colInd,
                                                    double *ret, int rl, int ru,
                                                    int cl, int cu,
                                                    int retClen) {
  slice_sparse_dense_row(inVal, inRowPtr, colInd, ret, rl, ru, cl, cu, retClen);
}

extern "C" __global__ void slice_sparse_dense_row_f(float *inVal, int *inRowPtr,
                                                    int *colInd, float *ret,
                                                    int rl, int ru, int cl,
                                                    int cu, int retClen) {
  slice_sparse_dense_row(inVal, inRowPtr, colInd, ret, rl, ru, cl, cu, retClen);
}

/**
 * Performs a slice operation where the input matrix is sparse and the output
 * matrix is dense.
 * This function avoids unnecessary sparse to dense conversion of the input
 * matrix.
 * Parallelization: subset of number of non-zeroes of input matrix.
 *
 * @params inVal input val pointer
 * @params inRowPtr input row pointer
 * @params colInd input col index pointer
 * @params ret dense output pointer
 * @param rl row lower
 * @param ru row upper
 * @param cl column lower
 * @param cu column upper
 * @param retClen number of columns of output matrix
 */
template <typename T>
__device__ void slice_sparse_dense_nnz(T *inVal, int *inRowPtr, int *colInd,
                                       T *ret, int rl, int ru, int cl, int cu,
                                       int retClen) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int i = tid + inRowPtr[rl];

  // Only slice if the index falls into the given range
  if (i < inRowPtr[ru + 1] && cl <= colInd[i] && colInd[i] <= cu) {
    // Find the row index for corresponding non-zero value 'i'.
    int rowIndex = rl;
    while (inRowPtr[rowIndex + 1] <= i) {
      rowIndex++;
    }
    ret[(rowIndex - rl) * retClen + (colInd[i] - cl)] = inVal[i];
  }
}

extern "C" __global__ void slice_sparse_dense_nnz_d(double *inVal,
                                                    int *inRowPtr, int *colInd,
                                                    double *ret, int rl, int ru,
                                                    int cl, int cu,
                                                    int retClen) {
  slice_sparse_dense_nnz(inVal, inRowPtr, colInd, ret, rl, ru, cl, cu, retClen);
}

extern "C" __global__ void slice_sparse_dense_nnz_f(float *inVal, int *inRowPtr,
                                                    int *colInd, float *ret,
                                                    int rl, int ru, int cl,
                                                    int cu, int retClen) {
  slice_sparse_dense_nnz(inVal, inRowPtr, colInd, ret, rl, ru, cl, cu, retClen);
}

/**
 * Performs a slice operation where the input matrix is dense and the output
 * matrix is dense.
 *
 * @params in dense input pointer
 * @params ret dense output pointer
 * @param rl row lower
 * @param ru row upper
 * @param cl column lower
 * @param cu column upper
 * @param inClen number of columns of input matrix
 * @param retRlen number of rows of output matrix
 * @param retClen number of columns of output matrix
 */
template <typename T>
__device__ void slice_dense_dense(T *in, T *ret, int rl, int ru, int cl, int cu,
                                  int inClen, int retRlen, int retClen) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int ix = tid / retClen;
  int iy = tid % retClen;
  if (ix < retRlen && iy < retClen) {
    int inIndex = (ix + rl) * inClen + cl + iy;
    ret[tid] = in[inIndex];
  }
}

extern "C" __global__ void slice_dense_dense_d(double *in, double *ret, int rl,
                                               int ru, int cl, int cu,
                                               int inClen, int retRlen,
                                               int retClen) {
  slice_dense_dense(in, ret, rl, ru, cl, cu, inClen, retRlen, retClen);
}

extern "C" __global__ void slice_dense_dense_f(float *in, float *ret, int rl,
                                               int ru, int cl, int cu,
                                               int inClen, int retRlen,
                                               int retClen) {
  slice_dense_dense(in, ret, rl, ru, cl, cu, inClen, retRlen, retClen);
}

/**
 * Does a copy of upper to lower triangle of the given matrix
 * @param ret the input and output array allocated on the GPU
 * @param dim the number of rows of the square matrix ret
 * @param N total number of elements of the matrix
 */
template <typename T>
__device__ void copy_u2l_dense(T *ret, int dim, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int ix = tid / dim;
  int iy = tid % dim;
  int id_dest = iy * dim + ix;
  if (iy > ix && id_dest < N) {
    // TODO: Potential to reduce the number of threads by half
    int id_src = tid;
    ret[id_dest] = ret[id_src];
  }
}

extern "C" __global__ void copy_u2l_dense_d(double *ret, int dim, int N) {
  copy_u2l_dense(ret, dim, N);
}

extern "C" __global__ void copy_u2l_dense_f(float *ret, int dim, int N) {
  copy_u2l_dense(ret, dim, N);
}

// Use this method in templates to fetch the maximum value for a given datatype
template <typename T>
__forceinline__ __device__ T MAX() {
  return T();
}
template <>
__forceinline__ __device__ float MAX <float>() {
  return FLT_MAX;
}
template <>
__forceinline__ __device__ double MAX <double>() {
  return DBL_MAX;
}

// op = {0=plus, 1=minus, 2=multiply, 3=divide, 4=power,
// 5=less, 6=lessequal, 7=greater, 8=greaterequal, 9=equal, 10=notequal,
// 11=min, 12=max, 13=and, 14=or, 15=minus1multiply, 16=minusnz,
// 17=modulus, 18=integer division}
template <typename T>
__forceinline__ __device__ T binaryOp(T x, T y, int op) {
  switch (op) {
    case 0:
      return x + y;
    case 1:
      return x - y;
    case 2:
      return x * y;
    case 3:
      return x / y;
    case 4:
      return pow(x, y);
    case 5:
      return (x < y) == 0 ? 0.0 : 1.0;
    case 6:
      return (x <= y) == 0 ? 0.0 : 1.0;
    case 7:
      return (x > y) == 0 ? 0.0 : 1.0;
    case 8:
      return (x >= y) == 0 ? 0.0 : 1.0;
    case 9:
      return (x == y) == 0 ? 0.0 : 1.0;
    case 10:
      return (x != y) == 0 ? 0.0 : 1.0;
    case 11:
      return min(x, y);
    case 12:
      return max(x, y);
    case 13:
      return ((int)llrint(x) & (int)llrint(y)) == 0 ? 0.0 : 1.0;
    case 14:
      return ((int)llrint(x) | (int)llrint(y)) == 0 ? 0.0 : 1.0;
    case 15:
      return 1 - x * y;
    case 16:
      return (x != 0.0 ? x - y : 0.0);
    case 17: {
      if (y == 0.0 || y == -0.0) {
        return nan("");
      }
      T v = x / y;
      // Check for v being NaN (v != v) or if it is infinity
      if (isnan(v) || isinf(v)) {
        return v;
      } else {
        v = floor(v);
      }
      return x - v * y;
    }
    case 18: {
      T v = x / y;
      if (isnan(v) || isinf(v)) {
        return v;
      } else {
        return floor(v);
      }
    }
    default:
      return MAX<T>();
  }
}

/**
 * Performs forward pass for relu: ret = max(A, 0)
 *
 * @param A input array allocated on the GPU
 * @param ret output array allocated on the GPU
 * @param rlen the number of rows
 * @param clen the number of columns
 */
template <typename T>
__device__ void relu(T *A, T *ret, int rlen, int clen) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int ix = tid / clen;
  int iy = tid % clen;
  if (ix < rlen && iy < clen) {
    ret[tid] = max(0.0, A[tid]);
  }
}

extern "C" __global__ void relu_d(double *A, double *ret, int rlen, int clen) {
  relu(A, ret, rlen, clen);
}

extern "C" __global__ void relu_f(float *A, float *ret, int rlen, int clen) {
  relu(A, ret, rlen, clen);
}

/**
 * This method computes the backpropagation errors for previous layer of relu
 * operation
 *
 * @param X input activation array allocated on the GPU
 * @param dout errors from previous layer
 * @param ret output array allocated on the GPU
 * @param rlen the number of rows
 * @param clen the number of columns
 */
template <typename T>
__device__ void relu_backward(T *X, T *dout, T *ret, int rlen, int clen) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int ix = tid / clen;
  int iy = tid % clen;
  if (ix < rlen && iy < clen) {
    ret[tid] = X[tid] > 0 ? dout[tid] : 0;
  }
}

extern "C" __global__ void relu_backward_d(double *X, double *dout, double *ret,
                                           int rlen, int clen) {
  relu_backward(X, dout, ret, rlen, clen);
}

extern "C" __global__ void relu_backward_f(float *X, float *dout, float *ret,
                                           int rlen, int clen) {
  relu_backward(X, dout, ret, rlen, clen);
}

/**
 * Performs inplace addition: ret += input
 *
 * @param input rhs input array allocated on the GPU
 * @param ret the input and output array allocated on the GPU
 * @param rlen the number of rows
 * @param clen the number of columns
 */
template <typename T>
__device__ void inplace_add(T *input, T *ret, int rlen, int clen) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int ix = tid / clen;
  int iy = tid % clen;
  if (ix < rlen && iy < clen) {
    ret[tid] += input[tid];
  }
}

extern "C" __global__ void inplace_add_d(double *input, double *ret, int rlen,
                                         int clen) {
  inplace_add(input, ret, rlen, clen);
}

extern "C" __global__ void inplace_add_f(float *input, float *ret, int rlen,
                                         int clen) {
  inplace_add(input, ret, rlen, clen);
}

// Performs the operation corresponding to the DML script:
// ones = matrix(1, rows=1, cols=Hout*Wout)
// output = input + matrix(bias %*% ones, rows=1, cols=F*Hout*Wout)
// This operation is often followed by conv2d and hence we have introduced
// bias_add(input, bias) built-in function
template <typename T>
__device__ void bias_add(T *input, T *bias, T *ret, int rlen, int clen,
                         int PQ) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int ix = tid / clen;
  int iy = tid % clen;
  if (ix < rlen && iy < clen) {
    int biasIndex = iy / PQ;
    ret[tid] = input[tid] + bias[biasIndex];
  }
}

extern "C" __global__ void bias_add_d(double *input, double *bias, double *ret,
                                      int rlen, int clen, int PQ) {
  bias_add(input, bias, ret, rlen, clen, PQ);
}

extern "C" __global__ void bias_add_f(float *input, float *bias, float *ret,
                                      int rlen, int clen, int PQ) {
  bias_add(input, bias, ret, rlen, clen, PQ);
}

// Performs the operation "ret <- A + alpha*B", where B is a vector
template <typename T>
__device__ void daxpy_matrix_vector(T *A, T *B, double alpha, T *ret, int rlenA,
                                    int clenA, int rlenB, int clenB) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int ix = tid / clenA;
  int iy = tid % clenA;
  if (ix < rlenA && iy < clenA) {
    int index = ix * clenA + iy;
    if (rlenB == 1) {
      ret[index] = A[index] + alpha * B[iy];
    } else {
      ret[index] = A[index] + alpha * B[ix];
    }
  }
}

extern "C" __global__ void daxpy_matrix_vector_d(double *A, double *B,
                                                 double alpha, double *ret,
                                                 int rlenA, int clenA,
                                                 int rlenB, int clenB) {
  daxpy_matrix_vector(A, B, alpha, ret, rlenA, clenA, rlenB, clenB);
}

extern "C" __global__ void daxpy_matrix_vector_f(float *A, float *B,
                                                 double alpha, float *ret,
                                                 int rlenA, int clenA,
                                                 int rlenB, int clenB) {
  daxpy_matrix_vector(A, B, alpha, ret, rlenA, clenA, rlenB, clenB);
}

// Performs similar operation as bias_add except elementwise multiplication
// instead of add
template <typename T>
__device__ void bias_multiply(T *input, T *bias, T *ret, int rlen, int clen,
                              int PQ) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int ix = tid / clen;
  int iy = tid % clen;
  if (ix < rlen && iy < clen) {
    int biasIndex = iy / PQ;
    ret[tid] = input[tid] * bias[biasIndex];
  }
}

extern "C" __global__ void bias_multiply_d(double *input, double *bias,
                                           double *ret, int rlen, int clen,
                                           int PQ) {
  bias_multiply(input, bias, ret, rlen, clen, PQ);
}

extern "C" __global__ void bias_multiply_f(float *input, float *bias,
                                           float *ret, int rlen, int clen,
                                           int PQ) {
  bias_multiply(input, bias, ret, rlen, clen, PQ);
}

/**
 * Performs a binary cellwise arithmetic operation on 2 matrices.
 * Either both matrices are of equal size or one of them is a vector or both
 * are.
 * @param A                 first input matrix allocated on GPU
 * @param B                 second input matrix allocated on GPU
 * @param C                 output allocated on GPU
 * @param maxRlen           maximum of the row lengths of A and B
 * @param maxClen           maximum of the column lengths of A and B
 * @param vectorAStatus     if A is a row vector, column vector or neither
 * @param vectorBStatus     if B is a row vector, column vector or neither
 * @param op                the numeric code of the arithmetic operation to
 * perform
 *
 */
template <typename T>
__device__ void matrix_matrix_cellwise_op(T *A, T *B, T *C, int maxRlen,
                                          int maxClen, int vectorAStatus,
                                          int vectorBStatus, int op) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int ix = tid / maxClen;
  int iy = tid % maxClen;

  if (ix < maxRlen && iy < maxClen) {
    int outIndex = ix * maxClen + iy;
    int aIndex = outIndex;
    int bIndex = outIndex;
    if (vectorAStatus == 1)
      aIndex = ix;  // clen == 1
    else if (vectorAStatus == 2)
      aIndex = iy;  // rlen == 1
    if (vectorBStatus == 1)
      bIndex = ix;  // clen == 1
    else if (vectorBStatus == 2)
      bIndex = iy;  // rlen == 1
    C[outIndex] = binaryOp(A[aIndex], B[bIndex], op);
    // printf("C[%d] = A[%d](%f) B[%d](%f) (%d %d)\n", outIndex, aIndex,
    // A[aIndex], bIndex,  B[bIndex], (ix+1), (iy+1));
    __syncthreads();
  }
}

extern "C" __global__ void matrix_matrix_cellwise_op_d(
    double *A, double *B, double *C, int maxRlen, int maxClen,
    int vectorAStatus, int vectorBStatus, int op) {
  matrix_matrix_cellwise_op(A, B, C, maxRlen, maxClen, vectorAStatus,
                            vectorBStatus, op);
}

extern "C" __global__ void matrix_matrix_cellwise_op_f(
    float *A, float *B, float *C, int maxRlen, int maxClen, int vectorAStatus,
    int vectorBStatus, int op) {
  matrix_matrix_cellwise_op(A, B, C, maxRlen, maxClen, vectorAStatus,
                            vectorBStatus, op);
}

/**
 * Performs an arithmetic operation between a matrix and a scalar.
 * C = s op A or C = A op s (where A is the matrix, s is the scalar and op is
 * the operation)
 * @param A             input matrix allocated on GPU
 * @param scalar        scalar input
 * @param C             output matrix allocated on GPU
 * @param size          number of elements in matrix A
 * @param op            number code of the arithmetic operation to perform
 * @param isLeftScalar  whether the scalar is on the left side
 */
template <typename T>
__device__ void matrix_scalar_op(T *A, T scalar, T *C, int size, int op,
                                 int isLeftScalar) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    if (isLeftScalar) {
      C[index] = binaryOp(scalar, A[index], op);
    } else {
      C[index] = binaryOp(A[index], scalar, op);
    }
  }
  __syncthreads();
}

extern "C" __global__ void matrix_scalar_op_d(double *A, double scalar,
                                              double *C, int size, int op,
                                              int isLeftScalar) {
  matrix_scalar_op(A, scalar, C, size, op, isLeftScalar);
}

extern "C" __global__ void matrix_scalar_op_f(float *A, double scalar, float *C,
                                              int size, int op,
                                              int isLeftScalar) {
  matrix_scalar_op(A, (float)scalar, C, size, op, isLeftScalar);
}

/**
 * Sets all elements (fills) of a double array of given length with a given
 * scalar value
 * @param A         array to be filled
 * @param scalar    value to fill array with
 * @param lenA      length of array A
 */
template <typename T>
__device__ void fill(T *A, T scalar, int lenA) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < lenA) {
    A[index] = scalar;
  }
}

extern "C" __global__ void fill_d(double *A, double scalar, int lenA) {
  fill(A, scalar, lenA);
}

extern "C" __global__ void fill_f(float *A, double scalar, int lenA) {
  fill(A, (float)scalar, lenA);
}

/**
 * Appends Matrix B to the right side of Matrix A into a new matrix C
 *         | 1 2 3 4 |   | 8 8 8 |     | 1 2 3 4 8 8 8 |
 * cbind ( | 9 8 7 6 | , | 7 7 7 | ) = | 9 8 7 6 7 7 7 |
 *         | 4 3 2 1 |   | 9 9 9 |     | 4 3 2 1 9 9 9 |
 * @param A      input matrix A allocated on the GPU
 * @param B      input matrix B allocated on the GPU
 * @param C      input matrix C allocated on the GPU
 * @param rowsA  rows in A
 * @param colsA  columns in A
 * @param rowsB  rows in B
 * @param colsB  columns in B
 */
template <typename T>
__device__ void cbind(T *A, T *B, T *C, int rowsA, int colsA, int rowsB,
                      int colsB) {
  int maxClen = max(colsA, colsB);
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int ix = tid / maxClen;
  int iy = tid % maxClen;

  int colsC = colsA + colsB;
  int rowsC = rowsA;

  // Copy an element of A into C into the appropriate location
  if (ix < rowsA && iy < colsA) {
    T elemA = A[ix * colsA + iy];
    C[ix * colsC + iy] = elemA;
  }

  // Copy an element of B into C into the appropriate location
  if (ix < rowsB && iy < colsB) {
    T elemB = B[ix * colsB + iy];
    C[ix * colsC + (iy + colsA)] = elemB;
  }
}

extern "C" __global__ void cbind_d(double *A, double *B, double *C, int rowsA,
                                   int colsA, int rowsB, int colsB) {
  cbind(A, B, C, rowsA, colsA, rowsB, colsB);
}

extern "C" __global__ void cbind_f(float *A, float *B, float *C, int rowsA,
                                   int colsA, int rowsB, int colsB) {
  cbind(A, B, C, rowsA, colsA, rowsB, colsB);
}

/**
 * Appends Matrix B to the bottom of Matrix A into a new matrix C
 *         | 2 3 4 |   | 8 8 8 |     | 2 3 4 |
 * rbind ( | 8 7 6 | , | 7 7 7 | ) = | 8 7 6 |
 *         | 3 2 1 |                 | 3 2 1 |
                                     | 8 8 8 |
                                     | 7 7 7 |
 * @param A      input matrix A allocated on the GPU
 * @param B      input matrix B allocated on the GPU
 * @param C      input matrix C allocated on the GPU
 * @param rowsA  rows in A
 * @param colsA  columns in A
 * @param rowsB  rows in B
 * @param colsB  columns in B
 */
template <typename T>
__device__ void rbind(T *A, T *B, T *C, int rowsA, int colsA, int rowsB,
                      int colsB) {
  int maxClen = max(colsA, colsB);
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int ix = tid / maxClen;
  int iy = tid % maxClen;

  int rowsC = rowsA + rowsB;
  int colsC = colsA;

  // Copy an element of A into C into the appropriate location
  if (ix < rowsA && iy < colsA) {
    T elemA = A[ix * colsA + iy];
    C[ix * colsC + iy] = elemA;
  }

  // Copy an element of B into C into the appropriate location
  if (ix < rowsB && iy < colsB) {
    T elemB = B[ix * colsB + iy];
    C[(ix + rowsA) * colsC + iy] = elemB;
  }
}

extern "C" __global__ void rbind_d(double *A, double *B, double *C, int rowsA,
                                   int colsA, int rowsB, int colsB) {
  rbind(A, B, C, rowsA, colsA, rowsB, colsB);
}

extern "C" __global__ void rbind_f(float *A, float *B, float *C, int rowsA,
                                   int colsA, int rowsB, int colsB) {
  rbind(A, B, C, rowsA, colsA, rowsB, colsB);
}

/**
 * Does a reduce operation over all elements of the array.
 * This method has been adapted from the Reduction sample in the NVIDIA CUDA
 * Samples (v8.0)
 * and the Reduction example available through jcuda.org
 * When invoked initially, all blocks partly compute the reduction operation
 * over the entire array
 * and writes it to the output/temporary array. A second invokation needs to
 * happen to get the
 * reduced value.
 * The number of threads, blocks and amount of shared memory is calculated in a
 * specific way.
 * Please refer to the NVIDIA CUDA Sample or the SystemML code that invokes this
 * method to see
 * how its done.
 * The template-ized version of this function is similar to what is found in
 * NVIDIA CUB
 *
 * @param ReductionOp       Type of the functor object that implements the
 * reduction operation
 */
template <typename ReductionOp, typename T>
__device__ void reduce(
    T *g_idata,  ///< input data stored in device memory (of size n)
    T *g_odata,  ///< output/temporary array stored in device memory (of size n)
    unsigned int n,  ///< size of the input and temporary/output arrays
    ReductionOp
        reduction_op,  ///< Reduction operation to perform (functor object)
    T initialValue)    ///< initial value for the reduction variable
{
  // extern __shared__ T sdata[];
  extern __shared__ __align__(sizeof(T)) unsigned char my_sdata[];
  T *sdata = reinterpret_cast<T *>(my_sdata);

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  unsigned int gridSize = blockDim.x * 2 * gridDim.x;

  T v = initialValue;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while (i < n) {
    v = reduction_op(v, g_idata[i]);
    // ensure we don't read out of bounds
    if (i + blockDim.x < n) v = reduction_op(v, g_idata[i + blockDim.x]);
    i += gridSize;
  }

  // each thread puts its local sum into shared memory
  sdata[tid] = v;
  __syncthreads();

  // do reduction in shared mem
  if (blockDim.x >= 1024) {
    if (tid < 512) {
      sdata[tid] = v = reduction_op(v, sdata[tid + 512]);
    }
    __syncthreads();
  }
  if (blockDim.x >= 512) {
    if (tid < 256) {
      sdata[tid] = v = reduction_op(v, sdata[tid + 256]);
    }
    __syncthreads();
  }
  if (blockDim.x >= 256) {
    if (tid < 128) {
      sdata[tid] = v = reduction_op(v, sdata[tid + 128]);
    }
    __syncthreads();
  }
  if (blockDim.x >= 128) {
    if (tid < 64) {
      sdata[tid] = v = reduction_op(v, sdata[tid + 64]);
    }
    __syncthreads();
  }

  if (tid < 32) {
    // now that we are using warp-synchronous programming (below)
    // we need to declare our shared memory volatile so that the compiler
    // doesn't reorder stores to it and induce incorrect behavior.
    volatile T *smem = sdata;
    if (blockDim.x >= 64) {
      smem[tid] = v = reduction_op(v, smem[tid + 32]);
    }
    if (blockDim.x >= 32) {
      smem[tid] = v = reduction_op(v, smem[tid + 16]);
    }
    if (blockDim.x >= 16) {
      smem[tid] = v = reduction_op(v, smem[tid + 8]);
    }
    if (blockDim.x >= 8) {
      smem[tid] = v = reduction_op(v, smem[tid + 4]);
    }
    if (blockDim.x >= 4) {
      smem[tid] = v = reduction_op(v, smem[tid + 2]);
    }
    if (blockDim.x >= 2) {
      smem[tid] = v = reduction_op(v, smem[tid + 1]);
    }
  }

  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/**
 * Does a reduce (sum) over each row of the array.
 * This kernel must be launched with as many blocks as there are rows.
 * The intuition for this kernel is that each block does a reduction over a
 * single row.
 * The maximum number of blocks that can launched (as of compute capability 3.0)
 * is 2^31 - 1
 * This works out fine for SystemML, since the maximum elements in a Java array
 * can be 2^31 - c (some small constant)
 * If the matrix is "fat" and "short", i.e. there are small number of rows and a
 * large number of columns,
 * there could be under-utilization of the hardware.
 * The template-ized version of this function is similar to what is found in
 * NVIDIA CUB
 * @param ReductionOp       Type of the functor object that implements the
 * reduction operation
 * @param AssignmentOp      Type of the functor object that is used to modify
 * the value before writing it to its final location in global memory for each
 * row
 */
template <typename ReductionOp, typename AssignmentOp, typename T>
__device__ void reduce_row(
    T *g_idata,  ///< input data stored in device memory (of size rows*cols)
    T *g_odata,  ///< output/temporary array store in device memory (of size
    /// rows*cols)
    unsigned int rows,  ///< rows in input and temporary/output arrays
    unsigned int cols,  ///< columns in input and temporary/output arrays
    ReductionOp
        reduction_op,  ///< Reduction operation to perform (functor object)
    AssignmentOp assignment_op,  ///< Operation to perform before assigning this
    /// to its final location in global memory for
    /// each row
    T initialValue) {  ///< initial value for the reduction variable
  // extern __shared__ T sdata[];
  extern __shared__ __align__(sizeof(T)) unsigned char my_sdata[];
  T *sdata = reinterpret_cast<T *>(my_sdata);

  // one block per row
  if (blockIdx.x >= rows) {
    return;
  }

  unsigned int block = blockIdx.x;
  unsigned int tid = threadIdx.x;
  unsigned int i = tid;
  unsigned int block_offset = block * cols;

  T v = initialValue;
  while (i < cols) {
    v = reduction_op(v, g_idata[block_offset + i]);
    i += blockDim.x;
  }

  // each thread puts its local sum into shared memory
  sdata[tid] = v;
  __syncthreads();

  // do reduction in shared mem
  if (blockDim.x >= 1024) {
    if (tid < 512) {
      sdata[tid] = v = reduction_op(v, sdata[tid + 512]);
    }
    __syncthreads();
  }
  if (blockDim.x >= 512) {
    if (tid < 256) {
      sdata[tid] = v = reduction_op(v, sdata[tid + 256]);
    }
    __syncthreads();
  }
  if (blockDim.x >= 256) {
    if (tid < 128) {
      sdata[tid] = v = reduction_op(v, sdata[tid + 128]);
    }
    __syncthreads();
  }
  if (blockDim.x >= 128) {
    if (tid < 64) {
      sdata[tid] = v = reduction_op(v, sdata[tid + 64]);
    }
    __syncthreads();
  }

  if (tid < 32) {
    // now that we are using warp-synchronous programming (below)
    // we need to declare our shared memory volatile so that the compiler
    // doesn't reorder stores to it and induce incorrect behavior.
    volatile T *smem = sdata;
    if (blockDim.x >= 64) {
      smem[tid] = v = reduction_op(v, smem[tid + 32]);
    }
    if (blockDim.x >= 32) {
      smem[tid] = v = reduction_op(v, smem[tid + 16]);
    }
    if (blockDim.x >= 16) {
      smem[tid] = v = reduction_op(v, smem[tid + 8]);
    }
    if (blockDim.x >= 8) {
      smem[tid] = v = reduction_op(v, smem[tid + 4]);
    }
    if (blockDim.x >= 4) {
      smem[tid] = v = reduction_op(v, smem[tid + 2]);
    }
    if (blockDim.x >= 2) {
      smem[tid] = v = reduction_op(v, smem[tid + 1]);
    }
  }

  // write result for this block to global mem, modify it with assignment op
  if (tid == 0) g_odata[block] = assignment_op(sdata[0]);
}

/**
 * Does a column wise reduction.
 * The intuition is that there are as many global threads as there are columns
 * Each global thread is responsible for a single element in the output vector
 * This of course leads to a under-utilization of the GPU resources.
 * For cases, where the number of columns is small, there can be unused SMs
 *
 * The template-ized version of this function is similar to what is found in
 * NVIDIA CUB
 * @param ReductionOp       Type of the functor object that implements the
 * reduction operation
 * @param AssignmentOp      Type of the functor object that is used to modify
 * the value before writing it to its final location in global memory for each
 * column
 */
template <typename ReductionOp, typename AssignmentOp, typename T>
__device__ void reduce_col(
    T *g_idata,  ///< input data stored in device memory (of size rows*cols)
    T *g_odata,  ///< output/temporary array store in device memory (of size
    /// rows*cols)
    unsigned int rows,  ///< rows in input and temporary/output arrays
    unsigned int cols,  ///< columns in input and temporary/output arrays
    ReductionOp
        reduction_op,  ///< Reduction operation to perform (functor object)
    AssignmentOp assignment_op,  ///< Operation to perform before assigning this
    /// to its final location in global memory for
    /// each column
    T initialValue)  ///< initial value for the reduction variable
{
  unsigned int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_tid >= cols) {
    return;
  }

  unsigned int i = global_tid;
  unsigned int grid_size = cols;
  T val = initialValue;

  while (i < rows * cols) {
    val = reduction_op(val, g_idata[i]);
    i += grid_size;
  }
  g_odata[global_tid] = assignment_op(val);
}

/**
 * Functor op for assignment op. This is a dummy/identity op.
 */
template <typename T>
struct IdentityOp {
  __device__ __forceinline__ T operator()(T a) const { return a; }
};

/**
 * Functor op for summation operation
 */
template <typename T>
struct SumOp {
  __device__ __forceinline__ T operator()(T a, T b) const { return a + b; }
};

/**
 * Do a summation over all elements of an array/matrix
 * @param g_idata   input data stored in device memory (of size n)
 * @param g_odata   output/temporary array stored in device memory (of size n)
 * @param n         size of the input and temporary/output arrays
 */
template <typename T>
__device__ void reduce_sum(T *g_idata, T *g_odata, unsigned int n) {
  SumOp<T> op;
  reduce<SumOp<T>, T>(g_idata, g_odata, n, op, (T)0.0);
}

extern "C" __global__ void reduce_sum_d(double *g_idata, double *g_odata,
                                        unsigned int n) {
  reduce_sum(g_idata, g_odata, n);
}

extern "C" __global__ void reduce_sum_f(float *g_idata, float *g_odata,
                                        unsigned int n) {
  reduce_sum(g_idata, g_odata, n);
}

/**
 * Do a summation over all rows of a matrix
 * @param g_idata   input matrix stored in device memory (of size rows * cols)
 * @param g_odata   output vector stored in device memory (of size rows)
 * @param rows      number of rows in input matrix
 * @param cols      number of columns in input matrix
 */
template <typename T>
__device__ void reduce_row_sum(T *g_idata, T *g_odata, unsigned int rows,
                               unsigned int cols) {
  SumOp<T> op;
  IdentityOp<T> aop;
  reduce_row<SumOp<T>, IdentityOp<T>, T>(g_idata, g_odata, rows, cols, op, aop,
                                         0.0);
}

extern "C" __global__ void reduce_row_sum_d(double *g_idata, double *g_odata,
                                            unsigned int rows,
                                            unsigned int cols) {
  reduce_row_sum(g_idata, g_odata, rows, cols);
}

extern "C" __global__ void reduce_row_sum_f(float *g_idata, float *g_odata,
                                            unsigned int rows,
                                            unsigned int cols) {
  reduce_row_sum(g_idata, g_odata, rows, cols);
}

/**
 * Do a summation over all columns of a matrix
 * @param g_idata   input matrix stored in device memory (of size rows * cols)
 * @param g_odata   output vector stored in device memory (of size cols)
 * @param rows      number of rows in input matrix
 * @param cols      number of columns in input matrix
 */
template <typename T>
__device__ void reduce_col_sum(T *g_idata, T *g_odata, unsigned int rows,
                               unsigned int cols) {
  SumOp<T> op;
  IdentityOp<T> aop;
  reduce_col<SumOp<T>, IdentityOp<T>, T>(g_idata, g_odata, rows, cols, op, aop,
                                         (T)0.0);
}

extern "C" __global__ void reduce_col_sum_d(double *g_idata, double *g_odata,
                                            unsigned int rows,
                                            unsigned int cols) {
  reduce_col_sum(g_idata, g_odata, rows, cols);
}

extern "C" __global__ void reduce_col_sum_f(float *g_idata, float *g_odata,
                                            unsigned int rows,
                                            unsigned int cols) {
  reduce_col_sum(g_idata, g_odata, rows, cols);
}

/**
 * Functor op for max operation
 */
template <typename T>
struct MaxOp {
  __device__ __forceinline__ T operator()(T a, T b) const { return fmax(a, b); }
};

template <>
struct MaxOp<float> {
  __device__ __forceinline__ float operator()(float a, float b) const {
    return fmaxf(a, b);
  }
};

/**
 * Do a max over all elements of an array/matrix
 * @param g_idata   input data stored in device memory (of size n)
 * @param g_odata   output/temporary array stode in device memory (of size n)
 * @param n         size of the input and temporary/output arrays
 */
template <typename T>
__device__ void reduce_max(T *g_idata, T *g_odata, unsigned int n) {
  MaxOp<T> op;
  reduce<MaxOp<T>, T>(g_idata, g_odata, n, op, -MAX<T>());
}

extern "C" __global__ void reduce_max_d(double *g_idata, double *g_odata,
                                        unsigned int n) {
  reduce_max(g_idata, g_odata, n);
}

extern "C" __global__ void reduce_max_f(float *g_idata, float *g_odata,
                                        unsigned int n) {
  reduce_max(g_idata, g_odata, n);
}

/**
 * Do a max over all rows of a matrix
 * @param g_idata   input matrix stored in device memory (of size rows * cols)
 * @param g_odata   output vector stored in device memory (of size rows)
 * @param rows      number of rows in input matrix
 * @param cols      number of columns in input matrix
 */
template <typename T>
__device__ void reduce_row_max(T *g_idata, T *g_odata, unsigned int rows,
                               unsigned int cols) {
  MaxOp<T> op;
  IdentityOp<T> aop;
  reduce_row<MaxOp<T>, IdentityOp<T>, T>(g_idata, g_odata, rows, cols, op, aop,
                                         -MAX<T>());
}

extern "C" __global__ void reduce_row_max_d(double *g_idata, double *g_odata,
                                            unsigned int rows,
                                            unsigned int cols) {
  reduce_row_max(g_idata, g_odata, rows, cols);
}

extern "C" __global__ void reduce_row_max_f(float *g_idata, float *g_odata,
                                            unsigned int rows,
                                            unsigned int cols) {
  reduce_row_max(g_idata, g_odata, rows, cols);
}

/**
 * Do a max over all columns of a matrix
 * @param g_idata   input matrix stored in device memory (of size rows * cols)
 * @param g_odata   output vector stored in device memory (of size cols)
 * @param rows      number of rows in input matrix
 * @param cols      number of columns in input matrix
 */
template <typename T>
__device__ void reduce_col_max(T *g_idata, T *g_odata, unsigned int rows,
                               unsigned int cols) {
  MaxOp<T> op;
  IdentityOp<T> aop;
  reduce_col<MaxOp<T>, IdentityOp<T>, T>(g_idata, g_odata, rows, cols, op, aop,
                                         -MAX<T>());
}

extern "C" __global__ void reduce_col_max_d(double *g_idata, double *g_odata,
                                            unsigned int rows,
                                            unsigned int cols) {
  reduce_col_max(g_idata, g_odata, rows, cols);
}

extern "C" __global__ void reduce_col_max_f(float *g_idata, float *g_odata,
                                            unsigned int rows,
                                            unsigned int cols) {
  reduce_col_max(g_idata, g_odata, rows, cols);
}

/**
 * Functor op for min operation
 */
template <typename T>
struct MinOp {
  __device__ __forceinline__ T operator()(T a, T b) const { return fmin(a, b); }
};

/**
 * Do a min over all elements of an array/matrix
 * @param g_idata   input data stored in device memory (of size n)
 * @param g_odata   output/temporary array stode in device memory (of size n)
 * @param n         size of the input and temporary/output arrays
 */
template <typename T>
__device__ void reduce_min(T *g_idata, T *g_odata, unsigned int n) {
  MinOp<T> op;
  reduce<MinOp<T>, T>(g_idata, g_odata, n, op, MAX<T>());
}

extern "C" __global__ void reduce_min_d(double *g_idata, double *g_odata,
                                        unsigned int n) {
  reduce_min(g_idata, g_odata, n);
}

extern "C" __global__ void reduce_min_f(float *g_idata, float *g_odata,
                                        unsigned int n) {
  reduce_min(g_idata, g_odata, n);
}

/**
 * Do a min over all rows of a matrix
 * @param g_idata   input matrix stored in device memory (of size rows * cols)
 * @param g_odata   output vector stored in device memory (of size rows)
 * @param rows      number of rows in input matrix
 * @param cols      number of columns in input matrix
 */
template <typename T>
__device__ void reduce_row_min(T *g_idata, T *g_odata, unsigned int rows,
                               unsigned int cols) {
  MinOp<T> op;
  IdentityOp<T> aop;
  reduce_row<MinOp<T>, IdentityOp<T>, T>(g_idata, g_odata, rows, cols, op, aop,
                                         MAX<T>());
}

extern "C" __global__ void reduce_row_min_d(double *g_idata, double *g_odata,
                                            unsigned int rows,
                                            unsigned int cols) {
  reduce_row_min(g_idata, g_odata, rows, cols);
}

extern "C" __global__ void reduce_row_min_f(float *g_idata, float *g_odata,
                                            unsigned int rows,
                                            unsigned int cols) {
  reduce_row_min(g_idata, g_odata, rows, cols);
}

/**
 * Do a min over all columns of a matrix
 * @param g_idata   input matrix stored in device memory (of size rows * cols)
 * @param g_odata   output vector stored in device memory (of size cols)
 * @param rows      number of rows in input matrix
 * @param cols      number of columns in input matrix
 */
template <typename T>
__device__ void reduce_col_min(T *g_idata, T *g_odata, unsigned int rows,
                               unsigned int cols) {
  MinOp<T> op;
  IdentityOp<T> aop;
  reduce_col<MinOp<T>, IdentityOp<T>, T>(g_idata, g_odata, rows, cols, op, aop,
                                         MAX<T>());
}

extern "C" __global__ void reduce_col_min_d(double *g_idata, double *g_odata,
                                            unsigned int rows,
                                            unsigned int cols) {
  reduce_col_min(g_idata, g_odata, rows, cols);
}

extern "C" __global__ void reduce_col_min_f(float *g_idata, float *g_odata,
                                            unsigned int rows,
                                            unsigned int cols) {
  reduce_col_min(g_idata, g_odata, rows, cols);
}

/**
 * Functor op for product operation
 */
template <typename T>
struct ProductOp {
  __device__ __forceinline__ T operator()(T a, T b) const { return a * b; }
};

/**
 * Do a product over all elements of an array/matrix
 * @param g_idata   input data stored in device memory (of size n)
 * @param g_odata   output/temporary array stode in device memory (of size n)
 * @param n         size of the input and temporary/output arrays
 */
template <typename T>
__device__ void reduce_prod(T *g_idata, T *g_odata, unsigned int n) {
  ProductOp<T> op;
  reduce<ProductOp<T>, T>(g_idata, g_odata, n, op, (T)1.0);
}

extern "C" __global__ void reduce_prod_d(double *g_idata, double *g_odata,
                                         unsigned int n) {
  reduce_prod(g_idata, g_odata, n);
}

extern "C" __global__ void reduce_prod_f(float *g_idata, float *g_odata,
                                         unsigned int n) {
  reduce_prod(g_idata, g_odata, n);
}

/**
 * Functor op for mean operation
 */
template <typename T>
struct MeanOp {
  const long
      _size;  ///< Number of elements by which to divide to calculate mean
  __device__ __forceinline__ MeanOp(long size) : _size(size) {}
  __device__ __forceinline__ T operator()(T total) const {
    return total / _size;
  }
};

/**
 * Do a mean over all rows of a matrix
 * @param g_idata   input matrix stored in device memory (of size rows * cols)
 * @param g_odata   output vector stored in device memory (of size rows)
 * @param rows      number of rows in input matrix
 * @param cols      number of columns in input matrix
 */
template <typename T>
__device__ void reduce_row_mean(T *g_idata, T *g_odata, unsigned int rows,
                                unsigned int cols) {
  SumOp<T> op;
  MeanOp<T> aop(cols);
  reduce_row<SumOp<T>, MeanOp<T>, T>(g_idata, g_odata, rows, cols, op, aop,
                                     (T)0.0);
}

extern "C" __global__ void reduce_row_mean_d(double *g_idata, double *g_odata,
                                             unsigned int rows,
                                             unsigned int cols) {
  reduce_row_mean(g_idata, g_odata, rows, cols);
}

extern "C" __global__ void reduce_row_mean_f(float *g_idata, float *g_odata,
                                             unsigned int rows,
                                             unsigned int cols) {
  reduce_row_mean(g_idata, g_odata, rows, cols);
}

/**
 * Do a mean over all columns of a matrix
 * @param g_idata   input matrix stored in device memory (of size rows * cols)
 * @param g_odata   output vector stored in device memory (of size cols)
 * @param rows      number of rows in input matrix
 * @param cols      number of columns in input matrix
 */
template <typename T>
__device__ void reduce_col_mean(T *g_idata, T *g_odata, unsigned int rows,
                                unsigned int cols) {
  SumOp<T> op;
  MeanOp<T> aop(rows);
  reduce_col<SumOp<T>, MeanOp<T>, T>(g_idata, g_odata, rows, cols, op, aop,
                                     0.0);
}

extern "C" __global__ void reduce_col_mean_d(double *g_idata, double *g_odata,
                                             unsigned int rows,
                                             unsigned int cols) {
  reduce_col_mean(g_idata, g_odata, rows, cols);
}

extern "C" __global__ void reduce_col_mean_f(float *g_idata, float *g_odata,
                                             unsigned int rows,
                                             unsigned int cols) {
  reduce_col_mean(g_idata, g_odata, rows, cols);
}

/**
 * Do an exp over all the elements of a matrix
 * @param A the input matrix (of length = size)
 * @param C the pre-allocated output matrix (of length = size)
 * @param siz the length of the input and output matrices
 */
template <typename T>
__device__ void matrix_exp(T *A, T *C, unsigned int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    C[index] = exp(A[index]);
  }
}

extern "C" __global__ void matrix_exp_d(double *A, double *C,
                                        unsigned int size) {
  matrix_exp(A, C, size);
}

extern "C" __global__ void matrix_exp_f(float *A, float *C, unsigned int size) {
  matrix_exp(A, C, size);
}

/**
 * Do an sqrt over all the elements of a matrix
 * @param A the input matrix (of length = size)
 * @param C the pre-allocated output matrix (of length = size)
 * @param siz the length of the input and output matrices
 */
template <typename T>
__device__ void matrix_sqrt(T *A, T *C, unsigned int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    C[index] = sqrt(A[index]);
  }
}

extern "C" __global__ void matrix_sqrt_d(double *A, double *C,
                                         unsigned int size) {
  matrix_sqrt(A, C, size);
}

extern "C" __global__ void matrix_sqrt_f(float *A, float *C,
                                         unsigned int size) {
  matrix_sqrt(A, C, size);
}

/**
 * Do an round over all the elements of a matrix
 * @param A the input matrix (of length = size)
 * @param C the pre-allocated output matrix (of length = size)
 * @param siz the length of the input and output matrices
 */
template <typename T>
__device__ void matrix_round(T *A, T *C, unsigned int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    C[index] = (T)llround(A[index]);
  }
}

extern "C" __global__ void matrix_round_d(double *A, double *C,
                                          unsigned int size) {
  matrix_round(A, C, size);
}

extern "C" __global__ void matrix_round_f(float *A, float *C,
                                          unsigned int size) {
  matrix_round(A, C, size);
}

/**
 * Do an abs over all the elements of a matrix
 * @param A the input matrix (of length = size)
 * @param C the pre-allocated output matrix (of length = size)
 * @param siz the length of the input and output matrices
 */
template <typename T>
__device__ void matrix_abs(T *A, T *C, unsigned int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    C[index] = (T)fabs(A[index]);
  }
}

extern "C" __global__ void matrix_abs_d(double *A, double *C,
                                        unsigned int size) {
  matrix_abs(A, C, size);
}

extern "C" __global__ void matrix_abs_f(float *A, float *C, unsigned int size) {
  matrix_abs(A, C, size);
}

/**
 * Do an log over all the elements of a matrix
 * @param A the input matrix (of length = size)
 * @param C the pre-allocated output matrix (of length = size)
 * @param siz the length of the input and output matrices
 */
template <typename T>
__device__ void matrix_log(T *A, T *C, unsigned int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    C[index] = log(A[index]);
  }
}

extern "C" __global__ void matrix_log_d(double *A, double *C,
                                        unsigned int size) {
  matrix_log(A, C, size);
}

extern "C" __global__ void matrix_log_f(float *A, float *C, unsigned int size) {
  matrix_log(A, C, size);
}

/**
 * Do an floor over all the elements of a matrix
 * @param A the input matrix (of length = size)
 * @param C the pre-allocated output matrix (of length = size)
 * @param siz the length of the input and output matrices
 */
template <typename T>
__device__ void matrix_floor(T *A, T *C, unsigned int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    C[index] = floor(A[index]);
  }
}

extern "C" __global__ void matrix_floor_d(double *A, double *C,
                                          unsigned int size) {
  matrix_floor(A, C, size);
}

extern "C" __global__ void matrix_floor_f(float *A, float *C,
                                          unsigned int size) {
  matrix_floor(A, C, size);
}

/**
 * Do an ceil over all the elements of a matrix
 * @param A the input matrix (of length = size)
 * @param C the pre-allocated output matrix (of length = size)
 * @param siz the length of the input and output matrices
 */
template <typename T>
__device__ void matrix_ceil(T *A, T *C, unsigned int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    C[index] = ceil(A[index]);
  }
}

extern "C" __global__ void matrix_ceil_d(double *A, double *C,
                                         unsigned int size) {
  matrix_ceil(A, C, size);
}

extern "C" __global__ void matrix_ceil_f(float *A, float *C,
                                         unsigned int size) {
  matrix_ceil(A, C, size);
}

/**
 * Do an sin over all the elements of a matrix
 * @param A the input matrix (of length = size)
 * @param C the pre-allocated output matrix (of length = size)
 * @param siz the length of the input and output matrices
 */
template <typename T>
__device__ void matrix_sin(T *A, T *C, unsigned int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    C[index] = sin(A[index]);
  }
}

extern "C" __global__ void matrix_sin_d(double *A, double *C,
                                        unsigned int size) {
  matrix_sin(A, C, size);
}

extern "C" __global__ void matrix_sin_f(float *A, float *C, unsigned int size) {
  matrix_sin(A, C, size);
}

/**
 * Do an sinh over all the elements of a matrix
 * @param A the input matrix (of length = size)
 * @param C the pre-allocated output matrix (of length = size)
 * @param siz the length of the input and output matrices
 */
template <typename T>
__device__ void matrix_sinh(T *A, T *C, unsigned int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    C[index] = sinh(A[index]);
  }
}

extern "C" __global__ void matrix_sinh_d(double *A, double *C,
                                         unsigned int size) {
  matrix_sinh(A, C, size);
}

extern "C" __global__ void matrix_sinh_f(float *A, float *C,
                                         unsigned int size) {
  matrix_sinh(A, C, size);
}

/**
 * Do an cos over all the elements of a matrix
 * @param A the input matrix (of length = size)
 * @param C the pre-allocated output matrix (of length = size)
 * @param siz the length of the input and output matrices
 */
template <typename T>
__device__ void matrix_cos(T *A, T *C, unsigned int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    C[index] = cos(A[index]);
  }
}

extern "C" __global__ void matrix_cos_d(double *A, double *C,
                                        unsigned int size) {
  matrix_cos(A, C, size);
}

extern "C" __global__ void matrix_cos_f(float *A, float *C, unsigned int size) {
  matrix_cos(A, C, size);
}

/**
 * Do an cosh over all the elements of a matrix
 * @param A the input matrix (of length = size)
 * @param C the pre-allocated output matrix (of length = size)
 * @param siz the length of the input and output matrices
 */
template <typename T>
__device__ void matrix_cosh(T *A, T *C, unsigned int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    C[index] = cosh(A[index]);
  }
}

extern "C" __global__ void matrix_cosh_d(double *A, double *C,
                                         unsigned int size) {
  matrix_cosh(A, C, size);
}

extern "C" __global__ void matrix_cosh_f(float *A, float *C,
                                         unsigned int size) {
  matrix_cosh(A, C, size);
}

/**
 * Do an tan over all the elements of a matrix
 * @param A the input matrix (of length = size)
 * @param C the pre-allocated output matrix (of length = size)
 * @param siz the length of the input and output matrices
 */
template <typename T>
__device__ void matrix_tan(T *A, T *C, unsigned int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    C[index] = tan(A[index]);
  }
}

extern "C" __global__ void matrix_tan_d(double *A, double *C,
                                        unsigned int size) {
  matrix_tan(A, C, size);
}

extern "C" __global__ void matrix_tan_f(float *A, float *C, unsigned int size) {
  matrix_tan(A, C, size);
}

/**
 * Do an tanh over all the elements of a matrix
 * @param A the input matrix (of length = size)
 * @param C the pre-allocated output matrix (of length = size)
 * @param siz the length of the input and output matrices
 */
template <typename T>
__device__ void matrix_tanh(T *A, T *C, unsigned int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    C[index] = tanh(A[index]);
  }
}

extern "C" __global__ void matrix_tanh_d(double *A, double *C,
                                         unsigned int size) {
  matrix_tanh(A, C, size);
}

extern "C" __global__ void matrix_tanh_f(float *A, float *C,
                                         unsigned int size) {
  matrix_tanh(A, C, size);
}

/**
 * Do an asin over all the elements of a matrix
 * @param A the input matrix (of length = size)
 * @param C the pre-allocated output matrix (of length = size)
 * @param siz the length of the input and output matrices
 */
template <typename T>
__device__ void matrix_asin(T *A, T *C, unsigned int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    C[index] = asin(A[index]);
  }
}

extern "C" __global__ void matrix_asin_d(double *A, double *C,
                                         unsigned int size) {
  matrix_asin(A, C, size);
}

extern "C" __global__ void matrix_asin_f(float *A, float *C,
                                         unsigned int size) {
  matrix_asin(A, C, size);
}

/**
 * Do an acos over all the elements of a matrix
 * @param A the input matrix (of length = size)
 * @param C the pre-allocated output matrix (of length = size)
 * @param siz the length of the input and output matrices
 */
template <typename T>
__device__ void matrix_acos(T *A, T *C, unsigned int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    C[index] = acos(A[index]);
  }
}

extern "C" __global__ void matrix_acos_d(double *A, double *C,
                                         unsigned int size) {
  matrix_acos(A, C, size);
}

extern "C" __global__ void matrix_acos_f(float *A, float *C,
                                         unsigned int size) {
  matrix_acos(A, C, size);
}

/**
 * Do an atan over all the elements of a matrix
 * @param A the input matrix (of length = size)
 * @param C the pre-allocated output matrix (of length = size)
 * @param siz the length of the input and output matrices
 */
template <typename T>
__device__ void matrix_atan(T *A, T *C, unsigned int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    C[index] = atan(A[index]);
  }
}

extern "C" __global__ void matrix_atan_d(double *A, double *C,
                                         unsigned int size) {
  matrix_atan(A, C, size);
}

extern "C" __global__ void matrix_atan_f(float *A, float *C,
                                         unsigned int size) {
  matrix_atan(A, C, size);
}

/**
 * Do an sign over all the elements of a matrix
 * Assign -1, 0 or 1 depending on the element being negative, 0 or positive
 * @param A the input matrix (of length = size)
 * @param C the pre-allocated output matrix (of length = size)
 * @param siz the length of the input and output matrices
 */
template <typename T>
__device__ void matrix_sign(T *A, T *C, unsigned int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    if (A[index] == 0.0) {
      C[index] = 0.0;
    } else {
      C[index] = copysign(1.0, A[index]);
    }
  }
}

extern "C" __global__ void matrix_sign_d(double *A, double *C,
                                         unsigned int size) {
  matrix_sign(A, C, size);
}

extern "C" __global__ void matrix_sign_f(float *A, float *C,
                                         unsigned int size) {
  matrix_sign(A, C, size);
}

/**
 * Do an sigmoid over all the elements of a matrix
 * @param A the input matrix (of length = size)
 * @param C the pre-allocated output matrix (of length = size)
 * @param siz the length of the input and output matrices
 */
template <typename T>
__device__ void matrix_sigmoid(T *A, T *C, unsigned int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    C[index] = 0.5 * tanh(0.5 * A[index]) + 0.5;
  }
}

extern "C" __global__ void matrix_sigmoid_d(double *A, double *C,
                                         unsigned int size) {
  matrix_sigmoid(A, C, size);
}

extern "C" __global__ void matrix_sigmoid_f(float *A, float *C,
                                         unsigned int size) {
  matrix_sigmoid(A, C, size);
}

template <typename T>
__device__ void prepare_lstm_input(T* smlInput, T* cudnnInput, int N, int D, int TD, int size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index < size) {
		int n = index / TD;
		int td = index % TD;
		int t = td / D;
		int d = td % D;
		cudnnInput[t*N*D + n*D + d] = smlInput[index];
	}
}


extern "C" __global__ void prepare_lstm_input_d(double* smlInput, double* cudnnInput, int N, int D, int TD, int size) {
  prepare_lstm_input(smlInput, cudnnInput, N, D, TD, size);
}

extern "C" __global__ void prepare_lstm_input_f(float* smlInput, float* cudnnInput, int N, int D, int TD, int size) {
  prepare_lstm_input(smlInput, cudnnInput, N, D, TD, size);
}

__device__ int swap_co(int offset) {
  return (offset < 2) ? offset : (offset == 2 ? 3 : 2);
}

template <typename T>
__device__ void prepare_lstm_weight(T* smlWeight, T* smlBias, T* cudnnWeight, int D, int M) {
  int DM = D*M; int MM = M*M; int DM4 = DM*4; 
  int M4 = M*4;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  // input: cbind(X_t, out_prev) => [N, D+M], weight: [D+M, 4M]
  // https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnGetRNNLinLayerMatrixParams states that 
  // Elements in each weight matrix are arranged in the row-major order, but the column-major format works !!
  // CuDNN gate order: i, f, c, o
  // CuDNN weight order: w_i, w_f, w_c, w_o, r_i, r_f, r_c, r_o
  // SystemML weight order: i, f, o, c; TF weight order: i, c, f, o
  // SystemML performs (X_t %*% W + out_prev %*% R) => [N, 4*M]
  
  // bias layout: bi bf bc bo 0 0 0 0
  // where W: [DxM], R: [MxM] and b: [1x1]
  
  // Maximum (D+M+2)*M4 threads
  int srcIndex = -1; int destIndex;
  if(index < DM4) {
    // Fill w_i, w_f, w_c and w_o
    int localIndex = index%DM;
    int smlRowIndex = localIndex/M; 
    int smlColIndex = swap_co(index/(DM))*M + localIndex%M;
    // Convert index to column-major where index = (index/(DM))*DM + (localIndex/M)*M + localIndex%M
    destIndex = (index/(DM))*DM + (localIndex%M)*D + localIndex/M;
    srcIndex = smlRowIndex*M4+smlColIndex;
  }
  else if(index < (D+M)*M4) {
    // Fill r_i, r_f, r_c and r_o
    int tmpIndex = index-DM4;
    int localIndex = tmpIndex % MM;
    int smlRowIndex = D + (localIndex / M);
    int smlColIndex = swap_co(tmpIndex/(MM))*M + localIndex%M;
    // Convert index to column-major where index = DM4 + (tmpIndex/(MM))*MM + (localIndex/M)*M + localIndex%M
    destIndex = DM4 + (tmpIndex/(MM))*MM + (localIndex%M)*M + localIndex/M;
    srcIndex = smlRowIndex*M4+smlColIndex;
  }
  else if(index < (D+M+1)*M4) {
  	// Fill bias
	int tmpIndex = index - (D+M)*M4;
	int smlColIndex = swap_co(tmpIndex/(M))*M + tmpIndex%M;
	cudnnWeight[index] = smlBias[smlColIndex];
  }
  // __syncthreads();
  if(srcIndex != -1)
  	cudnnWeight[destIndex] = smlWeight[srcIndex];
}

extern "C" __global__ void prepare_lstm_weight_d(double* smlWeight, double* smlBias, double* cudnnWeight, int D, int M) {
  prepare_lstm_weight(smlWeight, smlBias, cudnnWeight, D, M);
}

extern "C" __global__ void prepare_lstm_weight_f(float* smlWeight, float* smlBias, float* cudnnWeight, int D, int M) {
  prepare_lstm_weight(smlWeight, smlBias, cudnnWeight, D, M);
}

// We can later fold it in our reduce method
template <typename T>
__device__ void compute_nnz(
    T *g_idata,  ///< input data stored in device memory (of size n)
    T *g_odata,  ///< output/temporary array stored in device memory (of size n)
    unsigned int n)  ///< size of the input and temporary/output arrays
{
  // extern __shared__ T sdata[];
  extern __shared__ __align__(sizeof(T)) unsigned char my_sdata[];
  T *sdata = reinterpret_cast<T *>(my_sdata);

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
  unsigned int gridSize = blockDim.x * 2 * gridDim.x;

  T v = 0;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridDim).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while (i < n) {
    v += g_idata[i] != 0 ? 1 : 0;
    // ensure we don't read out of bounds
    if (i + blockDim.x < n) v += g_idata[i + blockDim.x] != 0 ? 1 : 0;
    i += gridSize;
  }

  // each thread puts its local sum into shared memory
  sdata[tid] = v;
  __syncthreads();

  // do reduction in shared mem
  if (blockDim.x >= 1024) {
    if (tid < 512) {
      sdata[tid] = v = v + sdata[tid + 512];
    }
    __syncthreads();
  }
  if (blockDim.x >= 512) {
    if (tid < 256) {
      sdata[tid] = v = v + sdata[tid + 256];
    }
    __syncthreads();
  }
  if (blockDim.x >= 256) {
    if (tid < 128) {
      sdata[tid] = v = v + sdata[tid + 128];
    }
    __syncthreads();
  }
  if (blockDim.x >= 128) {
    if (tid < 64) {
      sdata[tid] = v = v + sdata[tid + 64];
    }
    __syncthreads();
  }

  if (tid < 32) {
    // now that we are using warp-synchronous programming (below)
    // we need to declare our shared memory volatile so that the compiler
    // doesn't reorder stores to it and induce incorrect behavior.
    volatile T *smem = sdata;
    if (blockDim.x >= 64) {
      smem[tid] = v = v + smem[tid + 32];
    }
    if (blockDim.x >= 32) {
      smem[tid] = v = v + smem[tid + 16];
    }
    if (blockDim.x >= 16) {
      smem[tid] = v = v + smem[tid + 8];
    }
    if (blockDim.x >= 8) {
      smem[tid] = v = v + smem[tid + 4];
    }
    if (blockDim.x >= 4) {
      smem[tid] = v = v + smem[tid + 2];
    }
    if (blockDim.x >= 2) {
      smem[tid] = v = v + smem[tid + 1];
    }
  }

  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


extern "C" __global__ void compute_nnz_d(double *g_idata, double *g_odata, unsigned int n) {
	compute_nnz(g_idata, g_odata, n);
}

extern "C" __global__ void compute_nnz_f(float *g_idata, float *g_odata, unsigned int n) {
	compute_nnz(g_idata, g_odata, n);
}

template <typename T>
__device__ void prepare_lstm_output(T* smlInput, T* cudnnInput, int N, int T1, int M, int size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index < size) {
		int TM = T1*M;
		int NT = T1*N;
		int n = index / TM;
		int tm = index % TM;
		int t = tm / M;
		int m = tm % M;
		smlInput[index] = cudnnInput[t*N*M + n*M + m];
	}
}


extern "C" __global__ void prepare_lstm_output_d(double* smlInput, double* cudnnInput, int N, int T, int M, int size) {
  prepare_lstm_output(smlInput, cudnnInput, N, T, M, size);
}

extern "C" __global__ void prepare_lstm_output_f(float* smlInput, float* cudnnInput, int N, int T, int M, int size) {
  prepare_lstm_output(smlInput, cudnnInput, N, T, M, size);
}
