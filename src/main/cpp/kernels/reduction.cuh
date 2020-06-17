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
 * Please refer to the NVIDIA CUDA Sample or the SystemDS code that invokes this
 * method to see
 * how its done.
 * The template-ized version of this function is similar to what is found in
 * NVIDIA CUB
 *
 * @param n		size of the input and temporary/output arrays		
 * @param ReductionOp		Type of the functor object that implements the
 *		reduction operation
 * @param SpoofCellwiseOp		initial value for the reduction variable
 */
template<typename T, typename ReductionOp, typename SpoofCellwiseOp>
__device__ void reduce(
		T *g_idata, ///< input data stored in device memory (of size n)
		T *g_odata, ///< output/temporary array stored in device memory (of size n)
		unsigned int n,
		T initialValue, 
		ReductionOp reduction_op, 
	    SpoofCellwiseOp spoof_op)
{
	auto sdata = shared_memory_proxy<T>();

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
		v = reduction_op(v, spoof_op(g_idata[i]));

		if (i + blockDim.x < n)
			v = reduction_op(v, spoof_op(g_idata[i + blockDim.x]));
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
	if (tid == 0) {
		if(gridDim.x < 10)
			printf("blockIdx.x=%d reduction result: %3.1f\n", blockIdx.x, sdata[0]);
		g_odata[blockIdx.x] = sdata[0];
	}
}

/**
 * Does a reduce (sum) over each row of the array.
 * This kernel must be launched with as many blocks as there are rows.
 * The intuition for this kernel is that each block does a reduction over a
 * single row.
 * The maximum number of blocks that can launched (as of compute capability 3.0)
 * is 2^31 - 1
 * This works out fine for SystemDS, since the maximum elements in a Java array
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
template<typename ReductionOp, typename AssignmentOp, typename T>
__device__ void reduce_row(T *g_idata, ///< input data stored in device memory (of size rows*cols)
		T *g_odata,  ///< output/temporary array store in device memory (of size
		/// rows*cols)
		unsigned int rows,  ///< rows in input and temporary/output arrays
		unsigned int cols,  ///< columns in input and temporary/output arrays
		ReductionOp reduction_op, ///< Reduction operation to perform (functor object)
		AssignmentOp assignment_op, ///< Operation to perform before assigning this
		/// to its final location in global memory for
		/// each row
		T initialValue)  ///< initial value for the reduction variable
{
	auto sdata = shared_memory_proxy<T>();

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
 * The template-ized version of this function is similar to what is found in
 * NVIDIA CUB
 * @param ReductionOp       Type of the functor object that implements the
 * reduction operation
 * @param AssignmentOp      Type of the functor object that is used to modify
 * the value before writing it to its final location in global memory for each
 * column
 */
template<typename ReductionOp, typename AssignmentOp, typename T>
__device__ void reduce_col(T *g_idata, ///< input data stored in device memory (of size rows*cols)
		T *g_odata,  ///< output/temporary array store in device memory (of size rows*cols)
		unsigned int rows,  ///< rows in input and temporary/output arrays
		unsigned int cols,  ///< columns in input and temporary/output arrays
		ReductionOp reduction_op, ///< Reduction operation to perform (functor object)
		AssignmentOp assignment_op, ///< Operation to perform before assigning this
		/// to its final location in global memory for each column
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