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
nvcc -ptx -arch=sm_30 SystemML.cu
***********************************/

#include <cfloat>


/**
 * Does a copy of upper to lower triangle of the given matrix
 * @param ret the input and output array allocated on the GPU
 * @param dim the number of rows of the square matrix ret
 * @param N total number of elements of the matrix
 */
extern "C"
__global__ void copy_u2l_dense(double* ret, int dim, int N) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int id_dest = iy * dim + ix;
	if(iy > ix && id_dest < N) {
		// TODO: Potential to reduce the number of threads by half
		int id_src = ix * dim + iy;
		ret[id_dest] = ret[id_src];
	}
}

extern "C"
__forceinline__ __device__ double getBoolean(int val) {
	if(val == 0)
		return 0.0;
	else
		return 1.0;
}

// op = {0=plus, 1=minus, 2=multiply, 3=divide, 4=power,
// 5=less, 6=lessequal, 7=greater, 8=greaterequal, 9=equal, 10=notequal,
// 11=min, 12=max, 13=and, 14=or, 15=log}
extern "C"
__forceinline__ __device__ double binaryOp(double x, double y, int op) {
	switch(op) {
        case 0 : return x + y;
        case 1 : return x - y;
        case 2 : return x * y;
        case 3 : return x / y;
        case 4 : return pow(x, y);
        case 5 : return getBoolean(x < y);
        case 6 : return getBoolean(x <= y);
        case 7 : return getBoolean(x > y);
        case 8 : return getBoolean(x >= y);
        case 9 : return getBoolean(x == y);
        case 10 : return getBoolean(x != y);
        case 11 : return min(x, y);
        case 12 : return max(x, y);
        default : return DBL_MAX;
    }
}

extern "C"
__global__ void relu(double* A,  double* ret, int rlen, int clen) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	if(ix < rlen && iy < clen) {
		int index = ix * clen + iy;
		ret[index] = max(0.0, A[index]);
	}
}

// This method computes the backpropagation errors for previous layer of relu operation
extern "C"
__global__ void relu_backward(double* X,  double* dout, double* ret, int rlen, int clen) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	if(ix < rlen && iy < clen) {
		int index = ix * clen + iy;
		ret[index] = X[index] > 0 ?  dout[index] : 0;
	}
}

// Performs the operation corresponding to the DML script:
// ones = matrix(1, rows=1, cols=Hout*Wout)
// output = input + matrix(bias %*% ones, rows=1, cols=F*Hout*Wout)
// This operation is often followed by conv2d and hence we have introduced bias_add(input, bias) built-in function
extern "C"
__global__ void bias_add(double* input,  double* bias, double* ret, int rlen, int clen, int PQ) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	if(ix < rlen && iy < clen) {
		int index = ix * clen + iy;
		int biasIndex = iy / PQ;
		ret[index] = input[index] + bias[biasIndex];
	}
}

// Performs the operation "ret <- A + alpha*B", where B is a vector
extern "C"
__global__ void daxpy_matrix_vector(double* A,  double* B, double alpha, double* ret, int rlenA, int clenA, int rlenB, int clenB) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	if(ix < rlenA && iy < clenA) {
		int index = ix * clenA + iy;
		if(rlenB == 1) {
			ret[index] = A[index] + alpha*B[iy];
		}
		else {
			ret[index] = A[index] + alpha*B[ix];
		}
	}
}

// Performs similar operation as bias_add except elementwise multiplication instead of add
extern "C"
__global__ void bias_multiply(double* input,  double* bias, double* ret, int rlen, int clen, int PQ) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	if(ix < rlen && iy < clen) {
		int index = ix * clen + iy;
		int biasIndex = iy / PQ;
		ret[index] = input[index] * bias[biasIndex];
	}
}

// Compares the value and set
extern "C"
__global__ void compare_and_set(double* A,  double* ret, int rlen, int clen, double compareVal, double tol, double ifEqualsVal, double ifLessThanVal, double ifGreaterThanVal) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int index = ix * clen + iy;
	if(ix < rlen && iy < clen) {
		if(abs(A[index]-compareVal) < tol)
			ret[index] = ifEqualsVal;
		else if(A[index] < compareVal)
			ret[index] = ifLessThanVal;
		else
			ret[index] = ifGreaterThanVal;
	}
}


/**
 * Performs a binary cellwise arithmetic operation on 2 matrices.
 * Either both matrices are of equal size or one of them is a vector or both are.
 * @param A                 first input matrix allocated on GPU
 * @param B                 second input matrix allocated on GPU
 * @param C                 output allocated on GPU
 * @param maxRlen           maximum of the row lengths of A and B
 * @param maxClen           maximum of the column lengths of A and B
 * @param vectorAStatus     if A is a row vector, column vector or neither
 * @param vectorBStatus     if B is a row vector, column vector or neither
 * @param op                the numeric code of the arithmetic operation to perform
 *
 */
extern "C"
__global__ void matrix_matrix_cellwise_op(double* A, double* B, double* C,
	int maxRlen, int maxClen, int vectorAStatus, int vectorBStatus, int op) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;

	if(ix < maxRlen && iy < maxClen) {
		int outIndex = ix * maxClen + iy;
		int aIndex = outIndex;
		int bIndex = outIndex;
		if(vectorAStatus == 1)
			aIndex = ix; // clen == 1
		else if(vectorAStatus == 2)
			aIndex = iy; // rlen == 1
		if(vectorBStatus == 1)
			bIndex = ix; // clen == 1
		else if(vectorBStatus == 2)
			bIndex = iy; // rlen == 1
		C[outIndex] = binaryOp(A[aIndex], B[bIndex], op);
		//printf("C[%d] = A[%d](%f) B[%d](%f) (%d %d)\n", outIndex, aIndex, A[aIndex], bIndex,  B[bIndex], (ix+1), (iy+1));
    __syncthreads();
	}
}

/**
 * Performs an arithmetic operation between a matrix and a scalar.
 * C = s op A or C = A op s (where A is the matrix, s is the scalar and op is the operation)
 * @param A             input matrix allocated on GPU
 * @param scalar        scalar input
 * @param C             output matrix allocated on GPU
 * @param size          number of elements in matrix A
 * @param op            number code of the arithmetic operation to perform
 * @param isLeftScalar  whether the scalar is on the left side
 */
extern "C"
__global__ void matrix_scalar_op(double* A, double scalar, double* C, int size, int op, int isLeftScalar) {
	int index = blockIdx.x *blockDim.x + threadIdx.x;
	if(index < size) {
		if(isLeftScalar) {
			C[index] = binaryOp(scalar, A[index], op);
		} else {
			C[index] = binaryOp(A[index], scalar, op);
    }
	}
  __syncthreads();
}


/**
 * Sets all elements (fills) of a double array of given length with a given scalar value
 * @param A         array to be filled
 * @param scalar    value to fill array with
 * @param lenA      length of array A
 */
extern "C"
__global__ void fill(double* A, double scalar, int lenA) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < lenA){
	    A[index] = scalar;
	}
}

/**
 * Does a reduce operation over all elements of the array.
 * This method has been adapted from the Reduction sample in the NVIDIA CUDA Samples (v8.0)
 * and the Reduction example available through jcuda.org
 * When invoked initially, all blocks partly compute the reduction operation over the entire array
 * and writes it to the output/temporary array. A second invokation needs to happen to get the
 * reduced value.
 * The number of threads, blocks and amount of shared memory is calculated in a specific way.
 * Please refer to the NVIDIA CUDA Sample or the SystemML code that invokes this method to see
 * how its done.
 * The template-ized version of this function is similar to what is found in NVIDIA CUB
 *
 * @param ReductionOp       Type of the functor object that implements the reduction operation
 */
template <typename ReductionOp>
__device__ void reduce(
    double *g_idata,            ///< input data stored in device memory (of size n)
    double *g_odata,            ///< output/temporary array stored in device memory (of size n)
    unsigned int n,             ///< size of the input and temporary/output arrays
    ReductionOp reduction_op,	///< Reduction operation to perform (functor object)
	double initialValue)  		///< initial value for the reduction variable
{
    extern __shared__ double sdata[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x*2 + threadIdx.x;
    unsigned int gridSize = blockDim.x*2*gridDim.x;

    double v = initialValue;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        v = reduction_op(v, g_idata[i]);
        // ensure we don't read out of bounds
        if (i + blockDim.x < n)
            v = reduction_op(v, g_idata[i+blockDim.x]);
        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = v;
    __syncthreads();


    // do reduction in shared mem
		if (blockDim.x >= 1024){ if (tid < 512) { sdata[tid] = v = reduction_op(v, sdata[tid + 512]); } __syncthreads(); }
    if (blockDim.x >= 512) { if (tid < 256) { sdata[tid] = v = reduction_op(v, sdata[tid + 256]); } __syncthreads(); }
    if (blockDim.x >= 256) { if (tid < 128) { sdata[tid] = v = reduction_op(v, sdata[tid + 128]); } __syncthreads(); }
    if (blockDim.x >= 128) { if (tid <  64) { sdata[tid] = v = reduction_op(v, sdata[tid +  64]); } __syncthreads(); }

    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile double* smem = sdata;
        if (blockDim.x >=  64) { smem[tid] = v = reduction_op(v, smem[tid + 32]); }
        if (blockDim.x >=  32) { smem[tid] = v = reduction_op(v, smem[tid + 16]); }
        if (blockDim.x >=  16) { smem[tid] = v = reduction_op(v, smem[tid +  8]); }
        if (blockDim.x >=   8) { smem[tid] = v = reduction_op(v, smem[tid +  4]); }
        if (blockDim.x >=   4) { smem[tid] = v = reduction_op(v, smem[tid +  2]); }
        if (blockDim.x >=   2) { smem[tid] = v = reduction_op(v, smem[tid +  1]); }
    }

    // write result for this block to global mem
    if (tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}



/**
 * Does a reduce (sum) over each row of the array.
 * This kernel must be launched with as many blocks as there are rows.
 * The intuition for this kernel is that each block does a reduction over a single row.
 * The maximum number of blocks that can launched (as of compute capability 3.0) is 2^31 - 1
 * This works out fine for SystemML, since the maximum elements in a Java array can be 2^31 - c (some small constant)
 * If the matrix is "fat" and "short", i.e. there are small number of rows and a large number of columns,
 * there could be under-utilization of the hardware.
 * The template-ized version of this function is similar to what is found in NVIDIA CUB
 * @param ReductionOp       Type of the functor object that implements the reduction operation
 * @param AssignmentOp      Type of the functor object that is used to modify the value before writing it to its final location in global memory for each row
 */
template <typename ReductionOp,
          typename AssignmentOp>
__device__ void reduce_row(
    double *g_idata,            ///< input data stored in device memory (of size rows*cols)
    double *g_odata,            ///< output/temporary array store in device memory (of size rows*cols)
    unsigned int rows,          ///< rows in input and temporary/output arrays
    unsigned int cols,          ///< columns in input and temporary/output arrays
    ReductionOp reduction_op,		///< Reduction operation to perform (functor object)
    AssignmentOp assignment_op, ///< Operation to perform before assigning this to its final location in global memory for each row
    double initialValue){  			///< initial value for the reduction variable
    extern __shared__ double sdata[];

    // one block per row
    if (blockIdx.x >= rows) {
        return;
    }

    unsigned int block = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int i = tid;
    unsigned int block_offset = block * cols;

    double v = initialValue;
    while (i < cols){
        v = reduction_op(v, g_idata[block_offset + i]);
        i += blockDim.x;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = v;
    __syncthreads();

 		// do reduction in shared mem
  	if (blockDim.x >= 1024){ if (tid < 512) { sdata[tid] = v = reduction_op(v, sdata[tid + 512]); } __syncthreads(); }
    if (blockDim.x >= 512) { if (tid < 256) { sdata[tid] = v = reduction_op(v, sdata[tid + 256]); } __syncthreads(); }
    if (blockDim.x >= 256) { if (tid < 128) { sdata[tid] = v = reduction_op(v, sdata[tid + 128]); } __syncthreads(); }
    if (blockDim.x >= 128) { if (tid <  64) { sdata[tid] = v = reduction_op(v, sdata[tid +  64]); } __syncthreads(); }

    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile double* smem = sdata;
        if (blockDim.x >=  64) { smem[tid] = v = reduction_op(v, smem[tid + 32]); }
        if (blockDim.x >=  32) { smem[tid] = v = reduction_op(v, smem[tid + 16]); }
        if (blockDim.x >=  16) { smem[tid] = v = reduction_op(v, smem[tid +  8]); }
        if (blockDim.x >=   8) { smem[tid] = v = reduction_op(v, smem[tid +  4]); }
        if (blockDim.x >=   4) { smem[tid] = v = reduction_op(v, smem[tid +  2]); }
        if (blockDim.x >=   2) { smem[tid] = v = reduction_op(v, smem[tid +  1]); }
    }

    // write result for this block to global mem, modify it with assignment op
    if (tid == 0)
        g_odata[block] = assignment_op(sdata[0]);
}


/**
 * Does a column wise reduction.
 * The intuition is that there are as many global threads as there are columns
 * Each global thread is responsible for a single element in the output vector
 * This of course leads to a under-utilization of the GPU resources.
 * For cases, where the number of columns is small, there can be unused SMs
 *
 * The template-ized version of this function is similar to what is found in NVIDIA CUB
 * @param ReductionOp       Type of the functor object that implements the reduction operation
 * @param AssignmentOp      Type of the functor object that is used to modify the value before writing it to its final location in global memory for each column
 */
template <typename ReductionOp,
          typename AssignmentOp>
__device__ void reduce_col(
    double *g_idata,            ///< input data stored in device memory (of size rows*cols)
    double *g_odata,            ///< output/temporary array store in device memory (of size rows*cols)
    unsigned int rows,          ///< rows in input and temporary/output arrays
    unsigned int cols,          ///< columns in input and temporary/output arrays
    ReductionOp reduction_op,	///< Reduction operation to perform (functor object)
    AssignmentOp assignment_op, ///< Operation to perform before assigning this to its final location in global memory for each column
    double initialValue)  		///< initial value for the reduction variable
{
    unsigned int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_tid >= cols) {
        return;
    }

    unsigned int i = global_tid;
    unsigned int grid_size = cols;
    double val = initialValue;

    while (i < rows * cols) {
      val = reduction_op(val, g_idata[i]);
      i += grid_size;
    }
    g_odata[global_tid] = assignment_op(val);
}

/**
 * Functor op for assignment op. This is a dummy/identity op.
 */
typedef struct {
    __device__ __forceinline__
    double operator()(double a) const {
        return a;
    }
} IdentityOp;

/**
 * Functor op for summation operation
 */
typedef struct {
    __device__ __forceinline__
    double operator()(double a, double b) const {
        return a + b;
    }
} SumOp;


/**
 * Do a summation over all elements of an array/matrix
 * @param g_idata   input data stored in device memory (of size n)
 * @param g_odata   output/temporary array stored in device memory (of size n)
 * @param n         size of the input and temporary/output arrays
 */
extern "C"
__global__ void reduce_sum(double *g_idata, double *g_odata, unsigned int n){
	SumOp op;
  reduce<SumOp>(g_idata, g_odata, n, op, 0.0);
}

/**
 * Do a summation over all rows of a matrix
 * @param g_idata   input matrix stored in device memory (of size rows * cols)
 * @param g_odata   output vector stored in device memory (of size rows)
 * @param rows      number of rows in input matrix
 * @param cols      number of columns in input matrix
 */
extern "C"
__global__ void reduce_row_sum(double *g_idata, double *g_odata, unsigned int rows, unsigned int cols){
    SumOp op;
    IdentityOp aop;
    reduce_row<SumOp, IdentityOp>(g_idata, g_odata, rows, cols, op, aop, 0.0);
}

/**
 * Do a summation over all columns of a matrix
 * @param g_idata   input matrix stored in device memory (of size rows * cols)
 * @param g_odata   output vector stored in device memory (of size cols)
 * @param rows      number of rows in input matrix
 * @param cols      number of columns in input matrix
 */
extern "C"
__global__ void reduce_col_sum(double *g_idata, double *g_odata, unsigned int rows, unsigned int cols){
    SumOp op;
    IdentityOp aop;
    reduce_col<SumOp, IdentityOp>(g_idata, g_odata, rows, cols, op, aop, 0.0);
}


/**
 * Functor op for max operation
 */
typedef struct {
    __device__ __forceinline__
    double operator()(double a, double b) const {
        return fmax(a, b);
    }
} MaxOp;


/**
 * Do a max over all elements of an array/matrix
 * @param g_idata   input data stored in device memory (of size n)
 * @param g_odata   output/temporary array stode in device memory (of size n)
 * @param n         size of the input and temporary/output arrays
 */
extern "C"
__global__ void reduce_max(double *g_idata, double *g_odata, unsigned int n){
    MaxOp op;
    reduce<MaxOp>(g_idata, g_odata, n, op, -DBL_MAX);
}

/**
 * Do a max over all rows of a matrix
 * @param g_idata   input matrix stored in device memory (of size rows * cols)
 * @param g_odata   output vector stored in device memory (of size rows)
 * @param rows      number of rows in input matrix
 * @param cols      number of columns in input matrix
 */
extern "C"
__global__ void reduce_row_max(double *g_idata, double *g_odata, unsigned int rows, unsigned int cols){
    MaxOp op;
    IdentityOp aop;
    reduce_row<MaxOp, IdentityOp>(g_idata, g_odata, rows, cols, op, aop, -DBL_MAX);
}

/**
 * Do a max over all columns of a matrix
 * @param g_idata   input matrix stored in device memory (of size rows * cols)
 * @param g_odata   output vector stored in device memory (of size cols)
 * @param rows      number of rows in input matrix
 * @param cols      number of columns in input matrix
 */
extern "C"
__global__ void reduce_col_max(double *g_idata, double *g_odata, unsigned int rows, unsigned int cols){
    MaxOp op;
    IdentityOp aop;
    reduce_col<MaxOp, IdentityOp>(g_idata, g_odata, rows, cols, op, aop, -DBL_MAX);
}

/**
 * Functor op for min operation
 */
typedef struct {
    __device__ __forceinline__
    double operator()(double a, double b) const {
        return fmin(a, b);
    }
} MinOp;

/**
 * Do a min over all elements of an array/matrix
 * @param g_idata   input data stored in device memory (of size n)
 * @param g_odata   output/temporary array stode in device memory (of size n)
 * @param n         size of the input and temporary/output arrays
 */
extern "C"
__global__ void reduce_min(double *g_idata, double *g_odata, unsigned int n){
	MinOp op;
    reduce<MinOp>(g_idata, g_odata, n, op, DBL_MAX);
}

/**
 * Do a min over all rows of a matrix
 * @param g_idata   input matrix stored in device memory (of size rows * cols)
 * @param g_odata   output vector stored in device memory (of size rows)
 * @param rows      number of rows in input matrix
 * @param cols      number of columns in input matrix
 */
extern "C"
__global__ void reduce_row_min(double *g_idata, double *g_odata, unsigned int rows, unsigned int cols){
    MinOp op;
    IdentityOp aop;
    reduce_row<MinOp, IdentityOp>(g_idata, g_odata, rows, cols, op, aop, DBL_MAX);
}

/**
 * Do a min over all columns of a matrix
 * @param g_idata   input matrix stored in device memory (of size rows * cols)
 * @param g_odata   output vector stored in device memory (of size cols)
 * @param rows      number of rows in input matrix
 * @param cols      number of columns in input matrix
 */
extern "C"
__global__ void reduce_col_min(double *g_idata, double *g_odata, unsigned int rows, unsigned int cols){
    MinOp op;
    IdentityOp aop;
    reduce_col<MinOp>(g_idata, g_odata, rows, cols, op, aop, DBL_MAX);
}

/**
 * Functor op for product operation
 */
typedef struct {
    __device__ __forceinline__
    double operator()(double a, double b) const {
        return a * b;
    }
} ProductOp;

/**
 * Do a product over all elements of an array/matrix
 * @param g_idata   input data stored in device memory (of size n)
 * @param g_odata   output/temporary array stode in device memory (of size n)
 * @param n         size of the input and temporary/output arrays
 */
extern "C"
__global__ void reduce_prod(double *g_idata, double *g_odata, unsigned int n){
	ProductOp op;
    reduce<ProductOp>(g_idata, g_odata, n, op, 1.0);
}

/**
 * Functor op for mean operation
 */
struct MeanOp {
    const long _size;   ///< Number of elements by which to divide to calculate mean
		__device__ __forceinline__
    MeanOp(long size): _size(size) {}
    __device__ __forceinline__
    double operator()(double total) const {
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
extern "C"
__global__ void reduce_row_mean(double *g_idata, double *g_odata, unsigned int rows, unsigned int cols){
    SumOp op;
    MeanOp aop(cols);
    reduce_row<SumOp, MeanOp>(g_idata, g_odata, rows, cols, op, aop, 0.0);
}

/**
 * Do a mean over all columns of a matrix
 * @param g_idata   input matrix stored in device memory (of size rows * cols)
 * @param g_odata   output vector stored in device memory (of size cols)
 * @param rows      number of rows in input matrix
 * @param cols      number of columns in input matrix
 */
extern "C"
__global__ void reduce_col_mean(double *g_idata, double *g_odata, unsigned int rows, unsigned int cols){
    SumOp op;
    MeanOp aop(rows);
    reduce_col<SumOp, MeanOp>(g_idata, g_odata, rows, cols, op, aop, 0.0);
}


/**
 * Do an exp over all the elements of a matrix
 * @param A the input matrix (of length = size)
 * @param C the pre-allocated output matrix (of length = size)
 * @param siz the length of the input and output matrices
 */
extern "C"
__global__ void matrix_exp(double *A, double *C, unsigned int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size){
        C[index] = exp(A[index]);
    }
}

/**
 * Do an sqrt over all the elements of a matrix
 * @param A the input matrix (of length = size)
 * @param C the pre-allocated output matrix (of length = size)
 * @param siz the length of the input and output matrices
 */
extern "C"
__global__ void matrix_sqrt(double *A, double *C, unsigned int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size){
        C[index] = sqrt(A[index]);
    }
}

/**
 * Do an round over all the elements of a matrix
 * @param A the input matrix (of length = size)
 * @param C the pre-allocated output matrix (of length = size)
 * @param siz the length of the input and output matrices
 */
extern "C"
__global__ void matrix_round(double *A, double *C, unsigned int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size){
        C[index] = (double)llround(A[index]);
    }
}

/**
 * Do an abs over all the elements of a matrix
 * @param A the input matrix (of length = size)
 * @param C the pre-allocated output matrix (of length = size)
 * @param siz the length of the input and output matrices
 */
extern "C"
__global__ void matrix_abs(double *A, double *C, unsigned int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size){
        C[index] = (double)fabs(A[index]);
    }
}

/**
 * Do an log over all the elements of a matrix
 * @param A the input matrix (of length = size)
 * @param C the pre-allocated output matrix (of length = size)
 * @param siz the length of the input and output matrices
 */
extern "C"
__global__ void matrix_log(double *A, double *C, unsigned int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size){
        C[index] = log(A[index]);
    }
}

/**
 * Do an floor over all the elements of a matrix
 * @param A the input matrix (of length = size)
 * @param C the pre-allocated output matrix (of length = size)
 * @param siz the length of the input and output matrices
 */
extern "C"
__global__ void matrix_floor(double *A, double *C, unsigned int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size){
        C[index] = floor(A[index]);
    }
}

/**
 * Do an ceil over all the elements of a matrix
 * @param A the input matrix (of length = size)
 * @param C the pre-allocated output matrix (of length = size)
 * @param siz the length of the input and output matrices
 */
extern "C"
__global__ void matrix_ceil(double *A, double *C, unsigned int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size){
        C[index] = ceil(A[index]);
    }
}

/**
 * Do an sin over all the elements of a matrix
 * @param A the input matrix (of length = size)
 * @param C the pre-allocated output matrix (of length = size)
 * @param siz the length of the input and output matrices
 */
extern "C"
__global__ void matrix_sin(double *A, double *C, unsigned int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size){
        C[index] = sin(A[index]);
    }
}

/**
 * Do an cos over all the elements of a matrix
 * @param A the input matrix (of length = size)
 * @param C the pre-allocated output matrix (of length = size)
 * @param siz the length of the input and output matrices
 */
extern "C"
__global__ void matrix_cos(double *A, double *C, unsigned int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size){
        C[index] = cos(A[index]);
    }
}

/**
 * Do an tan over all the elements of a matrix
 * @param A the input matrix (of length = size)
 * @param C the pre-allocated output matrix (of length = size)
 * @param siz the length of the input and output matrices
 */
extern "C"
__global__ void matrix_tan(double *A, double *C, unsigned int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size){
        C[index] = tan(A[index]);
    }
}

/**
 * Do an asin over all the elements of a matrix
 * @param A the input matrix (of length = size)
 * @param C the pre-allocated output matrix (of length = size)
 * @param siz the length of the input and output matrices
 */
extern "C"
__global__ void matrix_asin(double *A, double *C, unsigned int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size){
        C[index] = asin(A[index]);
    }
}

/**
 * Do an acos over all the elements of a matrix
 * @param A the input matrix (of length = size)
 * @param C the pre-allocated output matrix (of length = size)
 * @param siz the length of the input and output matrices
 */
extern "C"
__global__ void matrix_acos(double *A, double *C, unsigned int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size){
        C[index] = acos(A[index]);
    }
}

/**
 * Do an atan over all the elements of a matrix
 * @param A the input matrix (of length = size)
 * @param C the pre-allocated output matrix (of length = size)
 * @param siz the length of the input and output matrices
 */
extern "C"
__global__ void matrix_atan(double *A, double *C, unsigned int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size){
        C[index] = atan(A[index]);
    }
}

/**
 * Do an sign over all the elements of a matrix
 * Assign -1, 0 or 1 depending on the element being negative, 0 or positive
 * @param A the input matrix (of length = size)
 * @param C the pre-allocated output matrix (of length = size)
 * @param siz the length of the input and output matrices
 */
extern "C"
__global__ void matrix_sign(double *A, double *C, unsigned int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size){
        if (A[index] == 0.0) {
            C[index] = 0.0;
        } else {
            C[index] = copysign(1.0, A[index]);
        }
    }
}