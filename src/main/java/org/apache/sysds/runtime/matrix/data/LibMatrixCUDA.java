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
 
package org.apache.sysds.runtime.matrix.data;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.functionobjects.And;
import org.apache.sysds.runtime.functionobjects.Builtin;
import org.apache.sysds.runtime.functionobjects.CM;
import org.apache.sysds.runtime.functionobjects.Divide;
import org.apache.sysds.runtime.functionobjects.Equals;
import org.apache.sysds.runtime.functionobjects.GreaterThan;
import org.apache.sysds.runtime.functionobjects.GreaterThanEquals;
import org.apache.sysds.runtime.functionobjects.IndexFunction;
import org.apache.sysds.runtime.functionobjects.IntegerDivide;
import org.apache.sysds.runtime.functionobjects.KahanPlus;
import org.apache.sysds.runtime.functionobjects.KahanPlusSq;
import org.apache.sysds.runtime.functionobjects.LessThan;
import org.apache.sysds.runtime.functionobjects.LessThanEquals;
import org.apache.sysds.runtime.functionobjects.Mean;
import org.apache.sysds.runtime.functionobjects.Minus;
import org.apache.sysds.runtime.functionobjects.Minus1Multiply;
import org.apache.sysds.runtime.functionobjects.MinusNz;
import org.apache.sysds.runtime.functionobjects.Modulus;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.functionobjects.Multiply2;
import org.apache.sysds.runtime.functionobjects.NotEquals;
import org.apache.sysds.runtime.functionobjects.Or;
import org.apache.sysds.runtime.functionobjects.Plus;
import org.apache.sysds.runtime.functionobjects.Power;
import org.apache.sysds.runtime.functionobjects.Power2;
import org.apache.sysds.runtime.functionobjects.ReduceAll;
import org.apache.sysds.runtime.functionobjects.ReduceCol;
import org.apache.sysds.runtime.functionobjects.ReduceDiag;
import org.apache.sysds.runtime.functionobjects.ReduceRow;
import org.apache.sysds.runtime.functionobjects.ValueFunction;
import org.apache.sysds.runtime.functionobjects.Builtin.BuiltinCode;
import org.apache.sysds.runtime.instructions.cp.DoubleObject;
import org.apache.sysds.runtime.instructions.gpu.GPUInstruction;
import org.apache.sysds.runtime.instructions.gpu.context.CSRPointer;
import org.apache.sysds.runtime.instructions.gpu.context.ExecutionConfig;
import org.apache.sysds.runtime.instructions.gpu.context.GPUContext;
import org.apache.sysds.runtime.instructions.gpu.context.GPUObject;
import org.apache.sysds.runtime.instructions.gpu.context.JCudaKernels;
import org.apache.sysds.runtime.matrix.operators.AggregateOperator;
import org.apache.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.apache.sysds.runtime.matrix.operators.BinaryOperator;
import org.apache.sysds.runtime.matrix.operators.CMOperator;
import org.apache.sysds.runtime.matrix.operators.LeftScalarOperator;
import org.apache.sysds.runtime.matrix.operators.RightScalarOperator;
import org.apache.sysds.runtime.matrix.operators.ScalarOperator;
import org.apache.sysds.runtime.util.IndexRange;
import org.apache.sysds.utils.Statistics;
import org.apache.sysds.utils.stats.Timing;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.cublasDiagType;
import jcuda.jcublas.cublasFillMode;
import jcuda.jcublas.cublasHandle;
import jcuda.jcublas.cublasOperation;
import jcuda.jcublas.cublasSideMode;
import jcuda.jcusparse.cusparseAction;
import jcuda.jcusparse.cusparseHandle;
import jcuda.jcusparse.cusparseIndexBase;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;
import static jcuda.runtime.JCuda.cudaDeviceSynchronize;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost;

import java.lang.Math;
import java.util.ArrayList;

/**
 * All CUDA kernels and library calls are redirected through this class
 * @see GPUContext
 * @see GPUObject
 */
public class LibMatrixCUDA {
	private static final Log LOG = LogFactory.getLog(LibMatrixCUDA.class.getName());

	protected static int CUDNN_DATA_TYPE = jcuda.jcudnn.cudnnDataType.CUDNN_DATA_DOUBLE;
	// The below variables are used in CSRPointer, GPUObjects, etc.
	public static CudaSupportFunctions cudaSupportFunctions = new DoublePrecisionCudaSupportFunctions();
	public static int sizeOfDataType = jcuda.Sizeof.DOUBLE;
	public static String customKernelSuffix = "_d";
	
	/**
	 * Sets the internal state based on the DMLScript.DATA_TYPE
	 */
	public static void resetFloatingPointPrecision() {
		if(DMLScript.FLOATING_POINT_PRECISION.equalsIgnoreCase("double")) {
			LibMatrixCUDA.CUDNN_DATA_TYPE = jcuda.jcudnn.cudnnDataType.CUDNN_DATA_DOUBLE;
			LibMatrixCUDA.cudaSupportFunctions = new DoublePrecisionCudaSupportFunctions();
			LibMatrixCUDA.sizeOfDataType = jcuda.Sizeof.DOUBLE;
			LibMatrixCUDA.customKernelSuffix = "_d";
		}
		else if(DMLScript.FLOATING_POINT_PRECISION.equalsIgnoreCase("single")) {
			LibMatrixCUDA.CUDNN_DATA_TYPE = jcuda.jcudnn.cudnnDataType.CUDNN_DATA_FLOAT;
			LibMatrixCUDA.cudaSupportFunctions = new SinglePrecisionCudaSupportFunctions();
			LibMatrixCUDA.sizeOfDataType = jcuda.Sizeof.FLOAT;
			LibMatrixCUDA.customKernelSuffix = "_f";
		}
		else {
			throw new DMLRuntimeException("Unsupported floating point precision: " + DMLScript.FLOATING_POINT_PRECISION);
		}
	}

	// Assume Compute Capability 3.0
	// MAX BLOCKS is 2^31 - 1 For compute capability > 3.0
	// MAX_THREADS is 1024 For compute capability > 3.0
	// WarpSize is usually 32
	// Shared mem is mostly at max 48 KB for current GPU
	private static int _MAX_THREADS = -1;
	private static int _MAX_BLOCKS  = -1;
	private static int _WARP_SIZE 	= -1;
	private static long _SHMEM_SIZE = -1;
	
	// From CuDNN 5.1 documentation:
	// The total size of a tensor including the potential padding between dimensions is limited to 2 Giga-elements of type datatype.
	protected static long maxNumElementsOfCuDNNTensor = 2000000000;

	//********************************************************************/
	//***************************** UTILS ********************************/
	//********************************************************************/

	/**
	 * Utility function to get maximum number of threads supported by the active CUDA device.
	 * @param gCtx a valid {@link GPUContext}
	 * @return max threads
	 */
	static int getMaxThreads(GPUContext gCtx){
		if (_MAX_THREADS == -1){
			_MAX_THREADS = gCtx.getMaxThreadsPerBlock();
		}
		return _MAX_THREADS;
	}

	/**
	 * Utility function to get maximum number of blocks supported by the active CUDA device.
	 * @param gCtx a valid {@link GPUContext}
	 * @return max blocks
	 */
	static int getMaxBlocks(GPUContext gCtx) {
		if (_MAX_BLOCKS == -1){
			_MAX_BLOCKS = gCtx.getMaxBlocks();
		}
		return _MAX_BLOCKS;
	}

	/**
	 * Utility function to get the maximum amount of shared memory/block size supported by the active CUDA device.
	 * @param gCtx a valid {@link GPUContext}
	 * @return warp size
	 */
	static long getMaxSharedMemory(GPUContext gCtx) {
		if (_SHMEM_SIZE == -1) {
			_SHMEM_SIZE = gCtx.getMaxSharedMemory();
		}
		return _SHMEM_SIZE;
	}

	/**
	 * Utility function to get the warp size supported by the active CUDA device.
	 * @param gCtx a valid {@link GPUContext}
	 * @return shared mem
	 */
	static int getWarpSize(GPUContext gCtx) {
		if (_WARP_SIZE == -1) {
			_WARP_SIZE = gCtx.getWarpSize();
		}
		return _WARP_SIZE;
	}

	public static boolean isInSparseFormat(GPUContext gCtx, MatrixObject mo) {
		if(mo.getGPUObject(gCtx) != null && mo.getGPUObject(gCtx).isAllocated())
			return mo.getGPUObject(gCtx).isSparse();
		return MatrixBlock.evalSparseFormatInMemory(mo.getNumRows(), mo.getNumColumns(), mo.getNnz());
	}
	
	/**
	 * Note: if the matrix is in dense format, it explicitly re-computes the number of nonzeros.
	 * 
	 * @param gCtx a valid GPU context
	 * @param instName instruction name
	 * @param mo matrix object
	 * @param recomputeDenseNNZ recompute NNZ if dense
	 * @return number of non-zeroes
	 */
	public static long getNnz(GPUContext gCtx, String instName, MatrixObject mo, boolean recomputeDenseNNZ) {
		if(mo.getGPUObject(gCtx) != null && mo.getGPUObject(gCtx).isAllocated()) {
			return mo.getGPUObject(gCtx).getNnz(instName, recomputeDenseNNZ);
		}
		else {
			return mo.getNnz();
		}
	}


	protected static cusparseHandle getCusparseHandle(GPUContext gCtx) {
		return gCtx.getCusparseHandle();
	}

	protected static cublasHandle getCublasHandle(GPUContext gCtx) {
		return gCtx.getCublasHandle();
	}

	public static JCudaKernels getCudaKernels(GPUContext gCtx) throws DMLRuntimeException {
		return gCtx.getKernels();
	}
	
	public static Pointer double2float(GPUContext gCtx, Pointer A, Pointer ret, int numElems) {
		getCudaKernels(gCtx).launchKernel("double2float", ExecutionConfig.getConfigForSimpleVectorOperations(numElems),
				A, ret, numElems);
		return ret;
	}
	
	public static Pointer float2double(GPUContext gCtx, Pointer A, Pointer ret, int numElems) {
		getCudaKernels(gCtx).launchKernel("float2double",
			ExecutionConfig.getConfigForSimpleVectorOperations(numElems), A, ret, numElems);
		return ret;
	}

	//********************************************************************/
	//************************ End of UTILS ******************************/
	//********************************************************************/

	//********************************************************************/
	//***************** DEEP LEARNING Operators **************************/
	//********************************************************************/

	private static Pointer _one;
	private static Pointer _zero;
	private static int oldDataTypeSize;
	/**
	 * Convenience method to get a pointer to value '1.0' on device. Instead of allocating and deallocating it for every kernel invocation.
	 * @return jcuda pointer
	 */
	public static Pointer one() {
		if(_one == null || oldDataTypeSize != sizeOfDataType) {
			_one = _dataTypePointerTo(1.0);
			oldDataTypeSize = sizeOfDataType;
		}
		return _one;
	}
	/**
	 * Convenience method to get a pointer to value '0.0f' on device. Instead of allocating and deallocating it for every kernel invocation.
	 * @return jcuda pointer
	 */
	public static Pointer zero() {
		if(_zero == null  || oldDataTypeSize != sizeOfDataType) {
			_zero = _dataTypePointerTo(0.0);
			oldDataTypeSize = sizeOfDataType;
		}
		return _zero;
	}

	/**
	 * Convenience method to get jcudaDenseMatrixPtr. This method explicitly converts sparse to dense format, so use it judiciously.
	 * @param gCtx a valid {@link GPUContext}
	 * @param input input matrix object
	 * @param instName  the invoking instruction's name for record {@link Statistics}.
	 * @return jcuda pointer
	 */
	public static Pointer getDensePointer(GPUContext gCtx, MatrixObject input, String instName) throws DMLRuntimeException {
		if(isInSparseFormat(gCtx, input)) {
			input.getGPUObject(gCtx).sparseToDense(instName);
		}
		return input.getGPUObject(gCtx).getDensePointer();
	}

	/**
	 * Convenience method to get the sparse matrix pointer from a {@link MatrixObject}. Converts dense to sparse if necessary.
	 * @param gCtx a valid {@link GPUContext}
	 * @param input input matrix
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @return a sparse matrix pointer
	 */
	protected static CSRPointer getSparsePointer(GPUContext gCtx, MatrixObject input, String instName) {
		if(!isInSparseFormat(gCtx, input)) {
			input.getGPUObject(gCtx).denseToSparse();
		}
		return input.getGPUObject(gCtx).getJcudaSparseMatrixPtr();
	}
	
	private static Pointer _dataTypePointerTo(double value) {
		if(sizeOfDataType == Sizeof.DOUBLE) {
			return Pointer.to(new double[] { value });
		}
		else if(sizeOfDataType == Sizeof.FLOAT) {
			return Pointer.to(new float[] { (float) value });
		}
		else {
			throw new RuntimeException("Unsupported datatype with size " + sizeOfDataType);
		}
	}
	
	protected static Pointer dataTypePointerTo(double value) {
		if(value == 1) {
			return one();
		}
		else if(value == 0) {
			return zero();
		}
		else {
			return _dataTypePointerTo(value);
		}
	}
	

	/**
	 * This method computes the backpropagation errors for previous layer of relu operation
	 * @param gCtx a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param input input image
	 * @param dout  next layer error propogation
	 * @param outputBlock output
	 */
	public static void reluBackward(GPUContext gCtx, String instName, MatrixObject input, MatrixObject dout, MatrixObject outputBlock) {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : reluBackward" + ", GPUContext=" + gCtx);
		}
		long rows = input.getNumRows();
		long cols = input.getNumColumns();
		Pointer imagePointer = getDensePointer(gCtx, input, instName);
		Pointer doutPointer = getDensePointer(gCtx, dout, instName);
		Pointer outputPointer = getDensePointer(gCtx, outputBlock, instName);

		getCudaKernels(gCtx).launchKernel("relu_backward",
			ExecutionConfig.getConfigForSimpleMatrixOperations(toInt(rows), toInt(cols)),
			imagePointer, doutPointer, outputPointer, toInt(rows), toInt(cols));
	}
	
	/**
	 * Perform channel_sums operations: out = rowSums(matrix(colSums(A), rows=C, cols=HW))
	 * 
	 * @param gCtx a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param input input image
	 * @param outputBlock output
	 * @param C number of channels
	 * @param HW height*width
	 */
	public static void channelSums(GPUContext gCtx, String instName, MatrixObject input, MatrixObject outputBlock, long C, long HW) {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : channelSums" + ", GPUContext=" + gCtx);
		}
		int N = toInt(input.getNumRows());
		int cols = toInt(input.getNumColumns());
		if(cols != C*HW) {
			throw new DMLRuntimeException("Incorrect parameters, number of columns " + cols + " != " + C + "*" + HW);
		}
		Pointer imagePointer = getDensePointer(gCtx, input, instName);
		Pointer outputPointer = getDensePointer(gCtx, outputBlock, instName);
		
		// We can replace this with CuDNN tensor reduce
		Pointer tmp = gCtx.allocate(instName, (long) cols * sizeOfDataType, false);
		reduceCol(gCtx, instName, "reduce_col_sum", imagePointer, tmp, N, cols);
		reduceRow(gCtx, instName, "reduce_row_sum", tmp, outputPointer, toInt(C), toInt(HW));
		gCtx.cudaFreeHelper(instName, tmp, DMLScript.EAGER_CUDA_FREE);
	}

	/**
	 * Performs the operation corresponding to the DML script:
	 * ones = matrix(1, rows=1, cols=Hout*Wout)
	 * output = input * matrix(bias %*% ones, rows=1, cols=F*Hout*Wout)
	 * This operation is often followed by conv2d and hence we have introduced bias_add(input, bias) built-in function
	 * @param gCtx   a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param input input image
	 * @param bias bias
	 * @param outputBlock output
	 */
	public static void biasMultiply(GPUContext gCtx, String instName, MatrixObject input, MatrixObject bias, MatrixObject outputBlock) {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : biasMultiply" + ", GPUContext=" + gCtx);
		}
		if(isInSparseFormat(gCtx, input)) {
			input.getGPUObject(gCtx).sparseToDense(instName);
		}
		if(isInSparseFormat(gCtx, bias)) {
			bias.getGPUObject(gCtx).sparseToDense(instName);
		}
		long rows = input.getNumRows();
		long cols = input.getNumColumns();
		long K = bias.getNumRows();
		long PQ = cols / K;
		if(bias.getNumColumns() != 1 || cols % K != 0) {
			throw new DMLRuntimeException("Incorrect inputs for bias_multiply: input[" + rows + " X " + cols + "] and bias[" + K + " X " + bias.getNumColumns() + "]");
		}
		Pointer imagePointer = input.getGPUObject(gCtx).getDensePointer();
		Pointer biasPointer = bias.getGPUObject(gCtx).getDensePointer();
		Pointer outputPointer = outputBlock.getGPUObject(gCtx).getDensePointer();
		getCudaKernels(gCtx).launchKernel("bias_multiply",
			ExecutionConfig.getConfigForSimpleMatrixOperations(toInt(rows), toInt(cols)),
			imagePointer, biasPointer, outputPointer, toInt(rows), toInt(cols), toInt(PQ));
	}

	/**
	 * Performs the operation corresponding to the DML script:
	 * ones = matrix(1, rows=1, cols=Hout*Wout)
	 * output = input + matrix(bias %*% ones, rows=1, cols=F*Hout*Wout)
	 * This operation is often followed by conv2d and hence we have introduced bias_add(input, bias) built-in function
	 * @param gCtx   a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param input input image
	 * @param bias bias
	 * @param outputBlock output
	 */
	public static void biasAdd(GPUContext gCtx, String instName, MatrixObject input, MatrixObject bias, MatrixObject outputBlock) {
		Pointer imagePointer = getDensePointer(gCtx, input, instName);
		Pointer biasPointer = getDensePointer(gCtx, bias, instName);
		Pointer outputPointer = getDensePointer(gCtx, outputBlock, instName);
		int rows = toInt(input.getNumRows());
		int cols = toInt(input.getNumColumns());
		int K = toInt(bias.getNumRows());
		if(bias.getNumColumns() != 1 || cols % K != 0) {
			throw new DMLRuntimeException("Incorrect inputs for bias_add: input[" + rows + " X " + cols + "] and bias[" + K + " X " + bias.getNumColumns() + "]");
		}
		biasAdd(gCtx, instName, imagePointer, biasPointer, outputPointer, rows, cols, K);
	}

	/**
	 * Performs the operation corresponding to the DML script:
	 * ones = matrix(1, rows=1, cols=Hout*Wout)
	 * output = input + matrix(bias %*% ones, rows=1, cols=F*Hout*Wout)
	 * This operation is often followed by conv2d and hence we have introduced bias_add(input, bias) built-in function
	 * @param gCtx   a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param image input image
	 * @param bias bias
	 * @param output output
	 * @param rows rows in input image
	 * @param cols cols in input image
	 * @param k rows in bias
	 */
	private static void biasAdd(GPUContext gCtx, String instName, Pointer image, Pointer bias, Pointer output, int rows, int cols, int k) {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : biasAdd" + ", GPUContext=" + gCtx);
		}
		int PQ = cols / k;
		getCudaKernels(gCtx).launchKernel("bias_add",
			ExecutionConfig.getConfigForSimpleMatrixOperations(rows, cols),
			image, bias, output, rows, cols, PQ);
	}
	

	//********************************************************************/
	//************* End of DEEP LEARNING Operators ***********************/
	//********************************************************************/



	//********************************************************************/
	//********** TRANSPOSE SELF MATRIX MULTIPLY Functions ****************/
	//********************************************************************/

	/**
	 * Performs tsmm, A %*% A' or A' %*% A, on GPU by exploiting cublasDsyrk(...)
	 * <p>
	 * Memory Usage - If dense, input space - rows * cols, no intermediate memory, output - Max(rows*rows, cols*cols)
	 * If sparse, calls matmult
	 *
	 * @param ec               execution context
	 * @param gCtx             a valid {@link GPUContext}
	 * @param instName         the invoking instruction's name for record {@link Statistics}.
	 * @param left             input matrix, as in a tsmm expression like A %*% A' or A' %*% A, we just need to check whether the left one is transposed or not, I named it 'left'
	 * @param outputName       output matrix name
	 * @param isLeftTransposed if true, left transposed
	 */
	public static void matmultTSMM(ExecutionContext ec, GPUContext gCtx, String instName, MatrixObject left, String outputName,
			boolean isLeftTransposed) {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : matmultTSMM" + ", GPUContext=" + gCtx);
		}
		if (ec.getGPUContext(0) != gCtx)
			throw new DMLRuntimeException("GPU : Invalid internal state, the GPUContext set with the ExecutionContext is not the same used to run this LibMatrixCUDA function");
		
		if(isInSparseFormat(gCtx, left)) {
			// For sparse TSMM, invoke matmult (TODO: possible performance improvement)
			LibMatrixCuMatMult.matmult(ec, gCtx, instName, left, left, outputName, isLeftTransposed, !isLeftTransposed);
			return;
		}

		// Since CuBLAS expects inputs in column-major format,
		// reverse the order of matrix-multiplication and take care of dimension mismatch.
		int transa = isLeftTransposed ? cublasOperation.CUBLAS_OP_N : cublasOperation.CUBLAS_OP_T;
		// Note: the dimensions are swapped
		int m = toInt(isLeftTransposed ? left.getNumColumns() : left.getNumRows());
		int k = toInt(isLeftTransposed ? left.getNumRows() : left.getNumColumns());

		// For dense TSMM, exploit cublasDsyrk(...) and call custom kernel to flip the matrix
		MatrixObject output = getDenseMatrixOutputForGPUInstruction(ec, instName, outputName, m, m);	// Allocated the dense output matrix

		if(m == -1)
			throw new DMLRuntimeException("Incorrect dimensions");

		int lda = toInt(isLeftTransposed ? m : k);
		int ldc = m;

		if(!left.getGPUObject(gCtx).isAllocated())
			throw new DMLRuntimeException("Input is not allocated:" + left.getGPUObject(gCtx).isAllocated());
		if(!output.getGPUObject(gCtx).isAllocated())
			throw new DMLRuntimeException("Output is not allocated:" + output.getGPUObject(gCtx).isAllocated());

		Pointer A = getDensePointer(gCtx, left, instName);
		Pointer C = getDensePointer(gCtx, output, instName);

		cudaSupportFunctions.cublassyrk(getCublasHandle(gCtx), cublasFillMode.CUBLAS_FILL_MODE_LOWER,transa, m, k, one(), A, lda, zero(), C, ldc);
		copyUpperToLowerTriangle(gCtx, instName, output);
	}

	/**
	 * Used for all version of TSMM where the result is known to be symmetric.
	 * Hence, we compute only the upper triangular matrix and copy this partial
	 * result down to lower triangular matrix once.
	 *
	 * @param gCtx     a valid {@link GPUContext}
	 * @param instName instruction name
	 * @param ret      upper triangular matrix
	 */
	private static void copyUpperToLowerTriangle(GPUContext gCtx, String instName, MatrixObject ret) {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : copyUpperToLowerTriangle" + ", GPUContext=" + gCtx);
		}
		if(isInSparseFormat(gCtx, ret)) {
			throw new DMLRuntimeException("Sparse GPU copyUpperToLowerTriangle is not implemented");
		}
		if(ret.getNumRows() != ret.getNumColumns()) {
			throw new DMLRuntimeException("Only square matrix kernel is implemented for copyUpperToLowerTriangle");
		}
		int dim = toInt(ret.getNumRows());
		getCudaKernels(gCtx).launchKernel("copy_u2l_dense",
			ExecutionConfig.getConfigForSimpleMatrixOperations(dim, dim),
			getDensePointer(gCtx, ret, instName), dim, dim*dim);
	}



	//********************************************************************/
	//******** End of TRANSPOSE SELF MATRIX MULTIPLY Functions ***********/
	//********************************************************************/


	//********************************************************************/
	//****************  UNARY AGGREGATE Functions ************************/
	//********************************************************************/

	/**
	 * Entry point to perform Unary aggregate operations on the GPU.
	 * The execution context object is used to allocate memory for the GPU.
	 *
	 * @param ec       Instance of {@link ExecutionContext}, from which the output variable will be allocated
	 * @param gCtx     a valid {@link GPUContext}
	 * @param instName name of the invoking instruction to record{@link Statistics}.
	 * @param in1      input matrix
	 * @param output   output matrix/scalar name
	 * @param op       Instance of {@link AggregateUnaryOperator} which encapsulates the direction of reduction/aggregation and the reduction operation.
	 */
	public static void unaryAggregate(ExecutionContext ec, GPUContext gCtx, String instName, MatrixObject in1, String output, AggregateUnaryOperator op) {
		if (ec.getGPUContext(0) != gCtx)
			throw new DMLRuntimeException("GPU : Invalid internal state, the GPUContext set with the ExecutionContext is not the same used to run this LibMatrixCUDA function");
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : unaryAggregate" + ", GPUContext=" + gCtx);
		}
		final int REDUCTION_ALL = 1;
		final int REDUCTION_ROW = 2;
		final int REDUCTION_COL = 3;
		final int REDUCTION_DIAG = 4;

		// A kahan sum implemention is not provided. is a "uak+" or other kahan operator is encountered,
		// it just does regular summation reduction.
		final int OP_PLUS = 1;
		final int OP_PLUS_SQ = 2;
		final int OP_MEAN = 3;
		final int OP_VARIANCE = 4;
		final int OP_MULTIPLY = 5;
		final int OP_MAX = 6;
		final int OP_MIN = 7;
		final int OP_MAXINDEX = 8;
		final int OP_MININDEX = 9;


		// Sanity Checks
		if(!in1.getGPUObject(gCtx).isAllocated())
			throw new DMLRuntimeException("Internal Error - The input is not allocated for a GPU Aggregate Unary:" + in1.getGPUObject(gCtx).isAllocated());

		boolean isSparse = in1.getGPUObject(gCtx).isSparse();
		IndexFunction indexFn = op.indexFn;
		AggregateOperator aggOp = op.aggOp;

		// Convert Reduction direction to a number
		int reductionDirection = -1;
		if (indexFn instanceof ReduceAll){
			reductionDirection = REDUCTION_ALL;
		} else if (indexFn instanceof ReduceRow){
			reductionDirection = REDUCTION_ROW;
		} else if (indexFn instanceof ReduceCol){
			reductionDirection = REDUCTION_COL;
		} else if (indexFn instanceof ReduceDiag){
			reductionDirection = REDUCTION_DIAG;
		} else {
			throw new DMLRuntimeException("Internal Error - Invalid index function type, only reducing along rows, columns, diagonals or all elements is supported in Aggregate Unary operations");
		}
		if(reductionDirection == -1)
			throw new DMLRuntimeException("Internal Error - Incorrect type of reduction direction set for aggregate unary GPU instruction");

		// Convert function type to a number
		int opIndex = -1;
		if (aggOp.increOp.fn instanceof KahanPlus) {
			opIndex = OP_PLUS;
		} else if (aggOp.increOp.fn instanceof KahanPlusSq) {
			opIndex = OP_PLUS_SQ;
		} else if (aggOp.increOp.fn instanceof Mean) {
			opIndex = OP_MEAN;
		} else if (aggOp.increOp.fn instanceof CM) {
			if(((CM)aggOp.increOp.fn).getAggOpType() != CMOperator.AggregateOperationTypes.VARIANCE)
				throw new DMLRuntimeException("Internal Error - Invalid Type of CM operator for Aggregate Unary operation on GPU");
			opIndex = OP_VARIANCE;
		} else if (aggOp.increOp.fn instanceof Plus) {
			opIndex = OP_PLUS;
		} else if (aggOp.increOp.fn instanceof Multiply) {
			opIndex = OP_MULTIPLY;
		} else if (aggOp.increOp.fn instanceof Builtin) {
			Builtin b = (Builtin)aggOp.increOp.fn;
			switch(b.bFunc) {
			case MAX: opIndex = OP_MAX; break;
			case MIN: opIndex = OP_MIN; break;
			case MAXINDEX: opIndex = OP_MAXINDEX; break;
			case MININDEX: opIndex = OP_MININDEX;break;
			default:
				throw new DMLRuntimeException("Internal Error - Unsupported Builtin Function for Aggregate unary being done on GPU");
			}
		} else {
			throw new DMLRuntimeException("Internal Error - Aggregate operator has invalid Value function");
		}
		if(opIndex == -1)
			throw new DMLRuntimeException("Internal Error - Incorrect type of operation set for aggregate unary GPU instruction");

		int rlen = (int)in1.getNumRows();
		int clen = (int)in1.getNumColumns();
		if (isSparse){
			// The strategy for the time being is to convert sparse to dense
			// until a sparse specific kernel is written.
			in1.getGPUObject(gCtx).sparseToDense(instName);
			// long nnz = in1.getNnz();
			// assert nnz > 0 : "Internal Error - number of non zeroes set to " + nnz + " in Aggregate Binary for GPU";
			// MatrixObject out = ec.getSparseMatrixOutputForGPUInstruction(output, nnz);
			// throw new DMLRuntimeException("Internal Error - Not implemented");

		}

		long outRLen = -1;
		long outCLen = -1;
		if (indexFn instanceof ReduceRow) { // COL{SUM, MAX...}
			outRLen = 1;
			outCLen = clen;
		}
		else if (indexFn instanceof ReduceCol) { // ROW{SUM, MAX,...}
			outRLen = rlen;
			outCLen = 1;
		}

		Pointer out = null;
		if (reductionDirection == REDUCTION_COL || reductionDirection == REDUCTION_ROW) {
			// Matrix output
			MatrixObject out1 = getDenseMatrixOutputForGPUInstruction(ec, instName, output, outRLen, outCLen);
			out = getDensePointer(gCtx, out1, instName);
		}

		Pointer in = getDensePointer(gCtx, in1, instName);
		int size = rlen * clen;

		// For scalars, set the scalar output in the Execution Context object
		switch (opIndex){
		case OP_PLUS: {
			switch(reductionDirection) {
			case REDUCTION_ALL : {
				double result = reduceAll(gCtx, instName, "reduce_sum", in, size);
				ec.setScalarOutput(output, new DoubleObject(result));
				break;
			}
			case REDUCTION_COL : {	// The names are a bit misleading, REDUCTION_COL refers to the direction (reduce all elements in a column)
				reduceRow(gCtx, instName, "reduce_row_sum", in, out, rlen, clen);
				break;
			}
			case REDUCTION_ROW : {
				reduceCol(gCtx, instName, "reduce_col_sum", in, out, rlen, clen);
				break;
			}
			case REDUCTION_DIAG :
				throw new DMLRuntimeException("Internal Error - Row, Column and Diag summation not implemented yet");
			}
			break;
		}
		case OP_PLUS_SQ : {
			// Calculate the squares in a temporary object tmp
			Pointer tmp = gCtx.allocate(instName, (long) size * sizeOfDataType, false);

			squareMatrix(gCtx, instName, in, tmp, rlen, clen);
			// Then do the sum on the temporary object and free it
			switch(reductionDirection) {
			case REDUCTION_ALL : {
				double result = reduceAll(gCtx, instName, "reduce_sum", tmp, size);
				ec.setScalarOutput(output, new DoubleObject(result));
				break;
			}
			case REDUCTION_COL : {	// The names are a bit misleading, REDUCTION_COL refers to the direction (reduce all elements in a column)
				reduceRow(gCtx, instName, "reduce_row_sum", tmp, out, rlen, clen);
				break;
			}
			case REDUCTION_ROW : {
				reduceCol(gCtx, instName, "reduce_col_sum", tmp, out, rlen, clen);
				break;
			}
			default:
				throw new DMLRuntimeException("Internal Error - Unsupported reduction direction for summation squared");
			}
			gCtx.cudaFreeHelper(instName, tmp, DMLScript.EAGER_CUDA_FREE);
			break;
		}
		case OP_MEAN:{
			switch(reductionDirection) {
			case REDUCTION_ALL: {
				double result = reduceAll(gCtx, instName, "reduce_sum", in, size);
				double mean = result / size;
				ec.setScalarOutput(output, new DoubleObject(mean));
				break;
			}
			case REDUCTION_COL: {
				reduceRow(gCtx, instName, "reduce_row_mean", in, out, rlen, clen);
				break;
			}
			case REDUCTION_ROW: {
				reduceCol(gCtx, instName, "reduce_col_mean", in, out, rlen, clen);
				break;
			}
			default:
				throw new DMLRuntimeException("Internal Error - Unsupported reduction direction for mean");
			}
			break;
		}
		case OP_MULTIPLY : {
			switch (reductionDirection) {
			case REDUCTION_ALL: {
				double result = reduceAll(gCtx, instName, "reduce_prod", in, size);
				ec.setScalarOutput(output, new DoubleObject(result));
				break;
			}
			default:
				throw new DMLRuntimeException("Internal Error - Unsupported reduction direction for multiplication");
			}
			break;
		}
		case OP_MAX :{
			switch(reductionDirection) {
			case REDUCTION_ALL: {
				double result = reduceAll(gCtx, instName, "reduce_max", in, size);
				ec.setScalarOutput(output, new DoubleObject(result));
				break;
			}
			case REDUCTION_COL: {
				reduceRow(gCtx, instName, "reduce_row_max", in, out, rlen, clen);
				break;
			}
			case REDUCTION_ROW: {
				reduceCol(gCtx, instName, "reduce_col_max", in, out, rlen, clen);
				break;
			}
			default:
				throw new DMLRuntimeException("Internal Error - Unsupported reduction direction for max");
			}
			break;
		}
		case OP_MIN :{
			switch(reductionDirection) {
			case REDUCTION_ALL: {
				double result = reduceAll(gCtx, instName, "reduce_min", in, size);
				ec.setScalarOutput(output, new DoubleObject(result));
				break;
			}
			case REDUCTION_COL: {
				reduceRow(gCtx, instName, "reduce_row_min", in, out, rlen, clen);
				break;
			}
			case REDUCTION_ROW: {
				reduceCol(gCtx, instName, "reduce_col_min", in, out, rlen, clen);
				break;
			}
			default:
				throw new DMLRuntimeException("Internal Error - Unsupported reduction direction for min");
			}
			break;
		}
		case OP_VARIANCE : {
			// Temporary GPU array for
			Pointer tmp = gCtx.allocate(instName, (long) size * sizeOfDataType, false);
			Pointer tmp2 = gCtx.allocate(instName, (long) size * sizeOfDataType, false);

			switch(reductionDirection) {

			case REDUCTION_ALL: {
				double result = reduceAll(gCtx, instName, "reduce_sum", in, size);
				double mean = result / size;

				// Subtract mean from every element in the matrix
				ScalarOperator minusOp = new RightScalarOperator(Minus.getMinusFnObject(), mean);
				matrixScalarOp(gCtx, instName, in, mean, rlen, clen, tmp, minusOp);

				squareMatrix(gCtx, instName, tmp, tmp2, rlen, clen);

				double result2 = reduceAll(gCtx, instName, "reduce_sum", tmp2, size);
				double variance = result2 / (size - 1);
				ec.setScalarOutput(output, new DoubleObject(variance));

				break;
			}
			case REDUCTION_COL: {
				reduceRow(gCtx, instName, "reduce_row_mean", in, out, rlen, clen);
				// Subtract the row-wise mean from every element in the matrix
				BinaryOperator minusOp = new BinaryOperator(Minus.getMinusFnObject());
				matrixMatrixOp(gCtx, instName, in, out, rlen, clen, VectorShape.NONE.code(), VectorShape.COLUMN.code(), tmp, minusOp);

				squareMatrix(gCtx, instName, tmp, tmp2, rlen, clen);

				Pointer tmpRow = gCtx.allocate(instName, (long) rlen * sizeOfDataType, false);
				reduceRow(gCtx, instName, "reduce_row_sum", tmp2, tmpRow, rlen, clen);

				ScalarOperator divideOp = new RightScalarOperator(Divide.getDivideFnObject(), clen - 1);
				matrixScalarOp(gCtx, instName, tmpRow, clen - 1, rlen, 1, out, divideOp);

				gCtx.cudaFreeHelper(instName, tmpRow, DMLScript.EAGER_CUDA_FREE);

				break;
			}
			case REDUCTION_ROW: {
				reduceCol(gCtx, instName, "reduce_col_mean", in, out, rlen, clen);
				// Subtract the columns-wise mean from every element in the matrix
				BinaryOperator minusOp = new BinaryOperator(Minus.getMinusFnObject());
				matrixMatrixOp(gCtx, instName, in, out, rlen, clen, VectorShape.NONE.code(), VectorShape.ROW.code(), tmp, minusOp);

				squareMatrix(gCtx, instName, tmp, tmp2, rlen, clen);

				Pointer tmpCol = gCtx.allocate(instName, (long) clen * sizeOfDataType, false);
				reduceCol(gCtx, instName, "reduce_col_sum", tmp2, tmpCol, rlen, clen);

				ScalarOperator divideOp = new RightScalarOperator(Divide.getDivideFnObject(), rlen - 1);
				matrixScalarOp(gCtx, instName, tmpCol, rlen - 1, 1, clen, out, divideOp);

				gCtx.cudaFreeHelper(instName, tmpCol, DMLScript.EAGER_CUDA_FREE);

				break;
			}
			default:
				throw new DMLRuntimeException("Internal Error - Unsupported reduction direction for variance");
			}
			gCtx.cudaFreeHelper(instName, tmp, DMLScript.EAGER_CUDA_FREE);
			gCtx.cudaFreeHelper(instName, tmp2, DMLScript.EAGER_CUDA_FREE);
			break;
		}
		case OP_MAXINDEX : {
			switch(reductionDirection) {
			case REDUCTION_COL:
				throw new DMLRuntimeException("Internal Error - Column maxindex of matrix not implemented yet for GPU ");
			default:
				throw new DMLRuntimeException("Internal Error - Unsupported reduction direction for maxindex");
			}
			// break;
		}
		case OP_MININDEX : {
			switch(reductionDirection) {
			case REDUCTION_COL:
				throw new DMLRuntimeException("Internal Error - Column minindex of matrix not implemented yet for GPU ");
			default:
				throw new DMLRuntimeException("Internal Error - Unsupported reduction direction for minindex");
			}
			// break;
		}
		default : throw new DMLRuntimeException("Internal Error - Invalid GPU Unary aggregate function!");
		}
	}

	/**
	 * Helper method to square a matrix in GPU memory
	 * @param gCtx   a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param in		input matrix on GPU
	 * @param out		output matrix on GPU
	 * @param rlen	row length
	 * @param clen	column length
	 */
	private static void squareMatrix(GPUContext gCtx, String instName, Pointer in, Pointer out, int rlen, int clen) {
		ScalarOperator power2op = new RightScalarOperator(Power.getPowerFnObject(), 2);
		matrixScalarOp(gCtx, instName, in, 2, rlen, clen, out, power2op);
	}

	/**
	 * Do a simple reduction, the output of which is a single value
	 * @param gCtx   a valid {@link GPUContext}
	 * @param kernelFunction 	name of the kernel function to invoke
	 * @param in							{@link Pointer} to matrix in device memory
	 * @param n								size of array
	 * @return	the reduced value
	 */
	private static double reduceAll(GPUContext gCtx, String instName, String kernelFunction, Pointer in, int n) {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : reduceAll for " + kernelFunction + ", GPUContext=" + gCtx);
		}

		int[] tmp = getKernelParamsForReduceAll(gCtx, n);
		int blocks = tmp[0], threads = tmp[1], sharedMem = tmp[2];

		Pointer tempOut = gCtx.allocate(instName, (long) blocks * sizeOfDataType, false);

		getCudaKernels(gCtx).launchKernel(kernelFunction, new ExecutionConfig(blocks, threads, sharedMem), in, tempOut, n);
		
		int s = blocks;
		while (s > 1) {
			tmp = getKernelParamsForReduceAll(gCtx, s);
			blocks = tmp[0]; threads = tmp[1]; sharedMem = tmp[2];
			getCudaKernels(gCtx).launchKernel(kernelFunction,
				new ExecutionConfig(blocks, threads, sharedMem), tempOut, tempOut, s);
			s = (s + (threads*2-1)) / (threads*2);
		}
		double[] result = {-1f};
		cudaSupportFunctions.deviceToHost(gCtx, tempOut, result, instName, false);
		gCtx.cudaFreeHelper(instName, tempOut, DMLScript.EAGER_CUDA_FREE);
		return result[0];
	}

	/**
	 * Do a reduction by row. Data is reduced per row and the
	 * resulting vector is calculated.
	 * @param gCtx            a valid {@link GPUContext}
	 * @param kernelFunction 	name of the kernel function to invoke
	 * @param in							{@link Pointer} to input matrix in device memory (size - rows * columns)
	 * @param out							{@link Pointer} to output matrix in device memory (size - rows * 1)
	 * @param rows						number of rows in input matrix
	 * @param cols						number of columns in input matrix
	 */
	private static void reduceRow(GPUContext gCtx, String instName, String kernelFunction, Pointer in, Pointer out, int rows, int cols) {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : reduceRow for " + kernelFunction + ", GPUContext=" + gCtx);
		}

		int[] tmp = getKernelParamsForReduceByRow(gCtx, rows, cols);
		int blocks = tmp[0], threads = tmp[1], sharedMem = tmp[2];

		Timing time = new Timing(false);
		double duration = 0;
		if(LOG.isTraceEnabled()) { time.start(); }

		getCudaKernels(gCtx).launchKernel(kernelFunction,
			new ExecutionConfig(blocks, threads, sharedMem), in, out, rows, cols);

		if(LOG.isTraceEnabled()) {
			cudaDeviceSynchronize();
			duration = time.stop();
			LOG.trace("uop kernel function " + kernelFunction + " executed in " + duration + "ms.");
		}
	}

	/**
	 * Do a reduction by column. Data is reduced per column and the
	 * resulting vector is calculated.
	 * @param gCtx            a valid {@link GPUContext}
	 * @param kernelFunction 	name of the kernel function to invoke
	 * @param in							{@link Pointer} to input matrix in device memory (size - rows * columns)
	 * @param out							{@link Pointer} to output matrix in device memory (size - 1 * cols)
	 * @param rows						number of rows in input matrix
	 * @param cols						number of columns in input matrix
	 */
	private static void reduceCol(GPUContext gCtx, String instName, String kernelFunction, Pointer in, Pointer out, int rows, int cols) {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : reduceCol for " + kernelFunction + ", GPUContext=" + gCtx);
		}

		int[] tmp = getKernelParamsForReduceByCol(gCtx, rows, cols);
		int blocks = tmp[0], threads = tmp[1], sharedMem = tmp[2];

		Timing time = new Timing(false);
		double duration = 0;
		if(LOG.isTraceEnabled()) { time.start(); }

		getCudaKernels(gCtx).launchKernel(kernelFunction,
			new ExecutionConfig(blocks, threads, sharedMem), in, out, rows, cols);

		if(LOG.isTraceEnabled()) {
			cudaDeviceSynchronize();
			duration = time.stop();
			LOG.trace("uop kernel function " + kernelFunction + " executed in " + duration + "ms.");
		}
	}

	/**
	 * Get threads, blocks and shared memory for a reduce all operation
	 * @param gCtx a valid {@link GPUContext}
	 * @param n size of input array
	 * @return integer array containing {blocks, threads, shared memory}
	 */
	private static int[] getKernelParamsForReduceAll(GPUContext gCtx, int n) {
		final int MAX_THREADS = getMaxThreads(gCtx);
		final int MAX_BLOCKS = getMaxBlocks(gCtx);
		final int WARP_SIZE = getWarpSize(gCtx);
		int threads = (n < MAX_THREADS *2) ? nextPow2((n + 1)/ 2) : MAX_THREADS;

		int blocks = (n + (threads * 2 - 1)) / (threads * 2);
		blocks = Math.min(MAX_BLOCKS, blocks);

		int sharedMemSize = threads * sizeOfDataType;
		if (threads <= WARP_SIZE) {
			sharedMemSize *= 2;
		}
		return new int[] {blocks, threads, sharedMemSize};
	}

	/**
	 * Get threads, blocks and shared memory for a reduce by row operation
	 * @param gCtx a valid {@link GPUContext}
	 * @param rows number of rows in input matrix
	 * @param cols number of columns in input matrix
	 * @return integer array containing {blocks, threads, shared memory}
	 */
	private static int[] getKernelParamsForReduceByRow(GPUContext gCtx, int rows, int cols) {
		final int WARP_SIZE = getWarpSize(gCtx);
		final int MAX_THREADS = getMaxThreads(gCtx);
		int threads = (cols < MAX_THREADS *2) ? nextPow2((cols + 1)/ 2) : MAX_THREADS;
		int blocks = rows;
		int sharedMemSize = threads * sizeOfDataType;
		if (threads <= WARP_SIZE){
			sharedMemSize *=2;
		}
		return new int[] {blocks, threads, sharedMemSize};
	}

	private static int[] getKernelParamsForReduceByCol(GPUContext gCtx, int rows, int cols) {
		final int MAX_THREADS = getMaxThreads(gCtx);
		final int MAX_BLOCKS = getMaxBlocks(gCtx);
		final int WARP_SIZE = getWarpSize(gCtx);
		int threads = Math.min(cols, MAX_THREADS);
		int blocks = Math.min(cols/MAX_THREADS, MAX_BLOCKS);
		if (cols % MAX_THREADS != 0) blocks++;
		int sharedMemSize = threads * sizeOfDataType;
		if (threads <= WARP_SIZE) {
			sharedMemSize *=2;
		}
		return new int[] {blocks, threads, sharedMemSize};
	}

	private static int nextPow2(int x)
	{
		--x;
		x |= x >> 1;
		x |= x >> 2;
		x |= x >> 4;
		x |= x >> 8;
		x |= x >> 16;
		return ++x;
	}

	//********************************************************************/
	//****************  END OF UNARY AGGREGATE Functions *****************/
	//********************************************************************/


	//********************************************************************/
	//************ Matrix-Matrix & Matrix-Scalar Functions ***************/
	//********************************************************************/

	/**
	 * Entry point to perform elementwise matrix-scalar relational operation specified by op
	 *
	 * @param ec         execution context
	 * @param gCtx       a valid {@link GPUContext}
	 * @param instName   the invoking instruction's name for record {@link Statistics}.
	 * @param in         input matrix
	 * @param outputName output matrix name
	 * @param op         scalar operator
	 */
	public static void matrixScalarRelational(ExecutionContext ec, GPUContext gCtx, String instName, MatrixObject in, String outputName, ScalarOperator op) {
		if (ec.getGPUContext(0) != gCtx)
			throw new DMLRuntimeException("GPU : Invalid internal state, the GPUContext set with the ExecutionContext is not the same used to run this LibMatrixCUDA function");
		double constant = op.getConstant();
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : matrixScalarRelational, scalar: " + constant + ", GPUContext=" + gCtx);
		}

		Pointer A, C;
		if (isSparseAndEmpty(gCtx, in)) {
			setOutputToConstant(ec, gCtx, instName, op.executeScalar(0.0), outputName, in.getNumRows(),
					in.getNumColumns());
			return;
		} else {
			A = getDensePointer(gCtx, in, instName);
			MatrixObject out = getDenseMatrixOutputForGPUInstruction(ec, instName, outputName, in.getNumRows(), in.getNumColumns());	// Allocated the dense output matrix
			C = getDensePointer(gCtx, out, instName);
		}

		int rlenA = toInt(in.getNumRows());
		int clenA = toInt(in.getNumColumns());

		matrixScalarOp(gCtx, instName, A, constant, rlenA, clenA, C, op);
	}

	/**
	 * Entry point to perform elementwise matrix-scalar arithmetic operation specified by op
	 *
	 * @param ec                execution context
	 * @param gCtx              a valid {@link GPUContext}
	 * @param instName          the invoking instruction's name for record {@link Statistics}.
	 * @param in                input matrix
	 * @param outputName        output matrix name
	 * @param isInputTransposed true if input transposed
	 * @param op                scalar operator
	 */
	public static void matrixScalarArithmetic(ExecutionContext ec, GPUContext gCtx, String instName, MatrixObject in, String outputName, boolean isInputTransposed, ScalarOperator op) {
		if (ec.getGPUContext(0) != gCtx)
			throw new DMLRuntimeException("GPU : Invalid internal state, the GPUContext set with the ExecutionContext is not the same used to run this LibMatrixCUDA function");
		double constant = op.getConstant();
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : matrixScalarArithmetic, scalar: " + constant + ", GPUContext=" + gCtx);
		}

		int outRLen = isInputTransposed ? (int) in.getNumColumns() : (int) in.getNumRows();
		int outCLen = isInputTransposed ? (int) in.getNumRows() : (int) in.getNumColumns();

		//boolean isCUDALibAvailable = (op.fn instanceof Multiply
		//		|| (op.fn instanceof Divide && op instanceof RightScalarOperator && constant != 0)) && !isSparseAndEmpty(gCtx, in);
		//if(!isCUDALibAvailable) {
		if(constant == 0) {
			if(op.fn instanceof Plus || (op.fn instanceof Minus && op instanceof RightScalarOperator) || op.fn instanceof Or) {
				deviceCopy(ec, gCtx, instName, in, outputName, isInputTransposed);
			}
			else if(op.fn instanceof Multiply || op.fn instanceof And) {
				setOutputToConstant(ec, gCtx, instName, 0.0, outputName, outRLen, outCLen);
			}
			else if(op.fn instanceof Power) {
				setOutputToConstant(ec, gCtx, instName, 1.0, outputName, outRLen, outCLen);
			}
			// TODO:
				// x/0.0 is either +Infinity or -Infinity according to Java.
			// In the context of a matrix, different elements of the matrix
			// could have different values.
			// If the IEEE 754 standard defines otherwise, this logic needs
			// to be re-enabled and the Java computation logic for divide by zero
			// needs to be revisited
			//else if(op.fn instanceof Divide && isSparseAndEmpty(gCtx, in)) {
			//	setOutputToConstant(ec, gCtx, instName, Double.NaN, outputName);
			//}
			//else if(op.fn instanceof Divide) {
			//	//For division, IEEE 754 defines x/0.0 as INFINITY and 0.0/0.0 as NaN.
			//	compareAndSet(ec, gCtx, instName, in, outputName, 0.0, 1e-6, Double.NaN, Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY);
			//}
			else {
				// TODO: Potential to optimize
				matrixScalarOp(ec, gCtx, instName, in, outputName, isInputTransposed, op);
			}
		}
		else if(constant == 1.0 && op.fn instanceof Or) {
			setOutputToConstant(ec, gCtx, instName, 1.0, outputName, outRLen, outCLen);
		}
		else if(constant == 1.0 && (op.fn instanceof And || op.fn instanceof Power)) {
			deviceCopy(ec, gCtx, instName, in, outputName, isInputTransposed);
		}
		else {
			matrixScalarOp(ec, gCtx, instName, in, outputName, isInputTransposed, op);
		}
		// }
		//else {
		//	double alpha = 0;
		//	if(op.fn instanceof Multiply) {
		//		alpha = op.getConstant();
		//	}
		//	else if(op.fn instanceof Divide && op instanceof RightScalarOperator) {
		//		alpha = Math.pow(op.getConstant(), -1.0);
		//	}
		//	else {
		//		throw new DMLRuntimeException("Unsupported op");
		//	}

		// TODO: Performance optimization: Call cublasDaxpy if(in.getNumRows() == 1 || in.getNumColumns() == 1)
		// C = alpha* op( A ) + beta* op ( B )
		//	dgeam(ec, gCtx, instName, in, in, outputName, isInputTransposed, isInputTransposed, alpha, 0.0);
		//}
	}


	/**
	 * Performs elementwise operation relational specified by op of two input matrices in1 and in2
	 *
	 * @param ec execution context
	 * @param gCtx a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param in1 input matrix 1
	 * @param in2 input matrix 2
	 * @param outputName output matrix name
	 * @param op binary operator
	 */
	public static void matrixMatrixRelational(ExecutionContext ec, GPUContext gCtx, String instName, MatrixObject in1, MatrixObject in2,
			String outputName, BinaryOperator op) {

		if (ec.getGPUContext(0) != gCtx)
			throw new DMLRuntimeException("GPU : Invalid internal state, the GPUContext set with the ExecutionContext is not the same used to run this LibMatrixCUDA function");

		boolean in1SparseAndEmpty = isSparseAndEmpty(gCtx, in1);
		boolean in2SparseAndEmpty = isSparseAndEmpty(gCtx, in2);
		if (in1SparseAndEmpty && in2SparseAndEmpty) {
			if (op.fn instanceof LessThan || op.fn instanceof GreaterThan || op.fn instanceof NotEquals) {
				setOutputToConstant(ec, gCtx, instName, 0.0, outputName, in1.getNumRows(), in1.getNumColumns());
			} else if (op.fn instanceof LessThanEquals || op.fn instanceof GreaterThanEquals || op.fn instanceof Equals) {
				setOutputToConstant(ec, gCtx, instName, 1.0, outputName, in1.getNumRows(), in1.getNumColumns());
			}
		} else if (in1SparseAndEmpty) {
			matrixScalarRelational(ec, gCtx, instName, in2, outputName, new LeftScalarOperator(op.fn, 0.0));
		} else if (in2SparseAndEmpty) {
			matrixScalarRelational(ec, gCtx, instName, in1, outputName, new RightScalarOperator(op.fn, 0.0));
		} else {
			matrixMatrixOp(ec, gCtx, instName, in1, in2, outputName, false, false, op);
		}
	}

	/**
	 * Performs elementwise arithmetic operation specified by op of two input matrices in1 and in2
	 *
	 * @param ec                execution context
	 * @param gCtx              a valid {@link GPUContext}
	 * @param instName          the invoking instruction's name for record {@link Statistics}.
	 * @param in1               input matrix 1
	 * @param in2               input matrix 2
	 * @param outputName        output matrix name
	 * @param isLeftTransposed  true if left-transposed
	 * @param isRightTransposed true if right-transposed
	 * @param op                binary operator
	 */
	public static void matrixMatrixArithmetic(ExecutionContext ec, GPUContext gCtx, String instName, MatrixObject in1, MatrixObject in2,
			String outputName, boolean isLeftTransposed, boolean isRightTransposed, BinaryOperator op) {
		if (ec.getGPUContext(0) != gCtx)
			throw new DMLRuntimeException("GPU : Invalid internal state, the GPUContext set with the ExecutionContext is not the same used to run this LibMatrixCUDA function");
		boolean isCUDALibAvailable = (op.fn instanceof Plus || op.fn instanceof Minus) && !isSparseAndEmpty(gCtx, in1) && !isSparseAndEmpty(gCtx, in2) && !isVector(in1) && !isVector(in2);
		if(!isCUDALibAvailable) {
			matrixMatrixOp(ec, gCtx, instName, in1, in2, outputName, isLeftTransposed, isRightTransposed, op);
		}
		else {
			double alpha;
			double beta;
			if(op.fn instanceof Plus) {
				alpha = 1.0;
				beta = 1.0;
			}
			else if(op.fn instanceof Minus) {
				alpha = 1.0;
				beta = -1.0;
			}
			else {
				throw new DMLRuntimeException("Unsupported op");
			}
			// C = alpha* op( A ) + beta* op ( B )
			dgeam(ec, gCtx, instName, in1, in2, outputName, isLeftTransposed, isRightTransposed, alpha, beta);
		}
	}

	/**
	 * Utility to do matrix-scalar operation kernel
	 *
	 * @param gCtx              a valid {@link GPUContext}
	 * @param instName          the invoking instruction's name for record {@link Statistics}.
	 * @param ec                execution context
	 * @param in                input matrix
	 * @param outputName        output variable name
	 * @param isInputTransposed true if input is transposed
	 * @param op                operator
	 */
	public static void matrixScalarOp(ExecutionContext ec, GPUContext gCtx, String instName, MatrixObject in, String outputName, boolean isInputTransposed,
			ScalarOperator op) {
		if (ec.getGPUContext(0) != gCtx)
			throw new DMLRuntimeException("GPU : Invalid internal state, the GPUContext set with the ExecutionContext is not the same used to run this LibMatrixCUDA function");
		if(isInputTransposed)
			throw new DMLRuntimeException("Transposing the input is not supported");

		int rlenA = toInt(in.getNumRows());
		int clenA = toInt(in.getNumColumns());
		Pointer A = getDensePointer(gCtx, in, instName); // TODO: FIXME: Implement sparse binCellSparseScalarOp kernel
		double scalar = op.getConstant();
		// MatrixObject out = ec.getMatrixObject(outputName);
		MatrixObject out = getDenseMatrixOutputForGPUInstruction(ec, instName, outputName, rlenA, clenA);	// Allocated the dense output matrix
		Pointer C = getDensePointer(gCtx, out, instName);
		matrixScalarOp(gCtx, instName, A, scalar, rlenA, clenA, C, op);
	}

	/**
	 * Helper method to launch binary scalar-matrix arithmetic/relational operations CUDA kernel.
	 * This method is isolated to be taken advantage of from other operations
	 * as it accepts JCuda {@link Pointer} instances instead of {@link MatrixObject} instances.
	 *
	 * @param gCtx     a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param a        the dense input matrix (allocated on GPU)
	 * @param scalar   the scalar value to do the op
	 * @param rlenA    row length of matrix a
	 * @param clenA    column lenght of matrix a
	 * @param c        the dense output matrix
	 * @param op       operation to perform
	 */
	private static void matrixScalarOp(GPUContext gCtx, String instName, Pointer a, double scalar, int rlenA, int clenA, Pointer c, ScalarOperator op) {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : matrix_scalar_op" + ", GPUContext=" + gCtx);
		}
		int isLeftScalar = (op instanceof LeftScalarOperator) ? 1 : 0;
		int size = rlenA * clenA;
		getCudaKernels(gCtx).launchKernel("matrix_scalar_op",
			ExecutionConfig.getConfigForSimpleVectorOperations(size),
			a, scalar, c, size, getBinaryOp(op.fn), isLeftScalar);
	}

	/**
	 * Utility to launch binary cellwise matrix-matrix operations CUDA kernel
	 *
	 * @param gCtx              a valid {@link GPUContext}
	 * @param ec                execution context
	 * @param instName          the invoking instruction's name for record {@link Statistics}.
	 * @param in1               left input matrix
	 * @param in2               right input matrix
	 * @param outputName        output variable name
	 * @param isLeftTransposed  true if left matrix is transposed
	 * @param isRightTransposed true if right matrix is transposed
	 * @param op                operator
	 */
	private static void matrixMatrixOp(ExecutionContext ec, GPUContext gCtx, String instName, MatrixObject in1, MatrixObject in2,
			String outputName, boolean isLeftTransposed, boolean isRightTransposed, BinaryOperator op) {
		if (ec.getGPUContext(0) != gCtx)
			throw new DMLRuntimeException("GPU : Invalid internal state, the GPUContext set with the ExecutionContext is not the same used to run this LibMatrixCUDA function");
		boolean isEmpty1 = isSparseAndEmpty(gCtx, in1);
		boolean isEmpty2 = isSparseAndEmpty(gCtx, in2);
		int rlenA = toInt(in1.getNumRows());
		int rlenB = toInt(in2.getNumRows());
		int clenA = toInt(in1.getNumColumns());
		int clenB = toInt(in2.getNumColumns());
		int vecStatusA = getVectorStatus(rlenA, clenA).code();
		int vecStatusB = getVectorStatus(rlenB, clenB).code();

		if(isLeftTransposed || isRightTransposed) {
			throw new DMLRuntimeException("Unsupported operator: GPU transposed binary op " + isLeftTransposed + " " + isRightTransposed);
		}
		long outRLen = Math.max(rlenA, rlenB);
		long outCLen = Math.max(clenA, clenB);

		if (isEmpty1 && isEmpty2){
			MatrixObject out = ec.allocateGPUMatrixObject(outputName, outRLen, outCLen);
			// When both inputs are empty, the output is empty too (except in the case of division)
			if (op.fn instanceof Divide || op.fn instanceof IntegerDivide || op.fn instanceof Modulus) {
				out.getGPUObject(gCtx).allocateAndFillDense(Double.NaN);
			} else if (op.fn instanceof Minus1Multiply) {
				out.getGPUObject(gCtx).allocateAndFillDense(1.0);
			} else {
				out.getGPUObject(gCtx).allocateSparseAndEmpty();
			}
		}
		// Check for M1 * M2 when M1 is empty; if M2 is a vector then fallback to general case
		else if(isEmpty1 && clenB != 1 && rlenB != 1) {
			// C = empty_in1 op in2 ==> becomes ==> C = 0.0 op in2
			matrixScalarArithmetic(ec, gCtx, instName, in2, outputName, isRightTransposed, new LeftScalarOperator(op.fn, 0.0));
		}
		// Check for M1 * M2 when M2 is empty; if M1 is a vector then fallback to general case
		else if(isEmpty2 && clenA != 1 && rlenA != 1) {
			// C = in1 op empty_in2 ==> becomes ==> C = in1 op 0.0
			matrixScalarArithmetic(ec, gCtx, instName, in1, outputName, isLeftTransposed, new RightScalarOperator(op.fn, 0.0));
		}
		else {
			Pointer A = getDensePointer(gCtx, in1, instName); // TODO: FIXME: Implement sparse binCellSparseOp kernel
			Pointer B = getDensePointer(gCtx, in2, instName); // TODO: FIXME: Implement sparse binCellSparseOp kernel

			// Allocated the dense output matrix
			MatrixObject out = null;
			try {
				out = getDenseMatrixOutputForGPUInstruction(ec, instName, outputName, outRLen, outCLen);
			} catch(DMLRuntimeException e) {
				throw new DMLRuntimeException("Incorrect dimensions: dimA:[" + rlenA + "," + clenA + "]"
						+ " dimB:[" + rlenB + "," + clenB + "] out:[" + outRLen + "," + outCLen + "]", e);
			}
			Pointer C = getDensePointer(gCtx, out, instName);

			int maxRlen = Math.max(rlenA, rlenB);
			int maxClen = Math.max(clenA, clenB);

			matrixMatrixOp(gCtx, instName, A, B, maxRlen, maxClen, vecStatusA, vecStatusB, C, op);
		}
	}

	/**
	 * Do an elementwise matrix-matrix arithmetic operation on the GPU
	 * c = a op b
	 * Either rows and cols in A are the same as in B or
	 * one of them is a vector or both are.
	 * @param gCtx        a valid {@link GPUContext}
	 * @param instName    the invoking instruction's name for record {@link Statistics}.
	 * @param a						The input matrix a allocated on the GPU
	 * @param b						The input matrix b allocated on the GPU
	 * @param maxRlen			the maximum of the row lengths between a & b
	 * @param maxClen			the maximum of the column lengths between a & b
	 * @param vecStatusA	if matrix A is a vector
	 * @param vecStatusB	if matrix B is a vector
	 * @param c						output matrix of size (maxRlen, maxClen) allocated on GPU
	 * @param op					the operation to perform
	 */
	private static void matrixMatrixOp(GPUContext gCtx, String instName, Pointer a, Pointer b, int maxRlen, int maxClen, int vecStatusA, int vecStatusB, Pointer c, BinaryOperator op) {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : matrix_matrix_cellwise_op" + ", GPUContext=" + gCtx);
		}
		getCudaKernels(gCtx).launchKernel("matrix_matrix_cellwise_op",
			ExecutionConfig.getConfigForSimpleMatrixOperations(maxRlen, maxClen),
			a, b, c, maxRlen, maxClen, vecStatusA, vecStatusB, getBinaryOp(op.fn));
	}

	/**
	 * This enum declares the different vector shapes
	 * as they recognized in the invoked CUDA kernel(s).
	 */
	enum VectorShape {
		COLUMN 	(1),
		ROW 		(2),
		NONE 		(0);
		private final int code;
		VectorShape(int code) {
			this.code = code;
		}
		int code() { return code; }
	}

	/**
	 * Given the number of rows and columns, returns
	 * whether this is a row vector, column vector or neither.
	 * @param rows
	 * @param cols
	 * @return 1 for column vector, 2 for row vector, 0 for neither
	 */
	private static VectorShape getVectorStatus(long rows, long cols) {
		if(cols == 1)
			return VectorShape.COLUMN;
		else if(rows == 1)
			return VectorShape.ROW;
		else
			return VectorShape.NONE;
	}

	private static boolean isVector(MatrixObject in) {
		return in.getNumRows() == 1 || in.getNumColumns() == 1;
	}

	private static boolean isSparseAndEmpty(GPUContext gCtx, MatrixObject in1) {
		boolean isSparse1 = isInSparseFormat(gCtx, in1);
		boolean isEmpty1 = isSparse1 && in1.getGPUObject(gCtx).getJcudaSparseMatrixPtr().nnz == 0;
		return isEmpty1;
	}

	private static void deviceCopy(ExecutionContext ec, GPUContext gCtx, String instName, MatrixObject src, String outputName, boolean isInputTransposed) {
		if (ec.getGPUContext(0) != gCtx)
			throw new DMLRuntimeException("GPU : Invalid internal state, the GPUContext set with the ExecutionContext is not the same used to run this LibMatrixCUDA function");
		if(!isInputTransposed)
			deviceCopy(ec, gCtx, instName, src, outputName);
		else
			transpose(ec, gCtx, instName, src, outputName);
	}

	/**
	 * Performs a deep device copy of a matrix on the GPU
	 *
	 * @param ec         execution context
	 * @param instName   the invoking instruction's name for record {@link Statistics}.
	 * @param src        source matrix
	 * @param outputName destination variable name
	 */
	private static void deviceCopy(ExecutionContext ec, GPUContext gCtx, String instName, MatrixObject src, String outputName) {
		Pointer srcPtr = getDensePointer(gCtx, src, instName); // TODO: FIXME: Implement sparse kernel
		MatrixObject out = ec.getMatrixObject(outputName);
		getDenseMatrixOutputForGPUInstruction(ec, instName, outputName, toInt(src.getNumRows()), toInt(src.getNumColumns()));	// Allocated the dense output matrix
		Pointer destPtr = getDensePointer(gCtx, out, instName);
		deviceCopy(instName, srcPtr, destPtr, (int)src.getNumRows(), (int)src.getNumColumns());
	}

	/**
	 * Fills an an array on the GPU with a given scalar value
	 * @param ec					currently active instance of the {@link ExecutionContext}
	 * @param gCtx        a valid {@link GPUContext}
	 * @param instName    name of the invoking instruction to record{@link Statistics}.
	 * @param constant		scalar value with which to fill the matrix
	 * @param outputName	(internal) name of the matrix that is to be filled
	 * @param numRows number of rows of output matrix object
	 * @param numCols number of columns of output matrix object
	 */
	private static void setOutputToConstant(ExecutionContext ec, GPUContext gCtx, String instName, double constant, String outputName, long numRows, long numCols) {
		if (ec.getGPUContext(0) != gCtx)
			throw new DMLRuntimeException("GPU : Invalid internal state, the GPUContext set with the ExecutionContext is not the same used to run this LibMatrixCUDA function");
		if(constant == 0) {
			getSparseMatrixOutputForGPUInstruction(ec, numRows, numCols, 0, instName, outputName);
		} else {
			MatrixObject out = getDenseMatrixOutputForGPUInstruction(ec, instName, outputName, numRows, numCols);   // Allocated the dense output matrix
			Pointer A = getDensePointer(gCtx, out, instName);
			int rlen = toInt(out.getNumRows());
			int clen = toInt(out.getNumColumns());
			int size = rlen * clen;
			getCudaKernels(gCtx).launchKernel("fill", ExecutionConfig.getConfigForSimpleVectorOperations(size), A, constant, size);
		}
	}

	/**
	 * Performs a deep copy of input device double pointer corresponding to matrix
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param src source matrix
	 * @param dest destination matrix
	 * @param rlen number of rows
	 * @param clen number of columns
	 */
	public static void deviceCopy(String instName, Pointer src, Pointer dest, int rlen, int clen) {
		int size = rlen * clen * sizeOfDataType;
		cudaMemcpy(dest, src, size, cudaMemcpyDeviceToDevice);
	}

	/**
	 * Helper function to get numeric value for binary op.
	 * This number is passed down to the CUDA kernel
	 * and the appropriate binary operation is performed on the GPU.
	 * op = {0=plus, 1=minus, 2=multiply, 3=divide, 4=power,
	 * 5=less, 6=lessequal, 7=greater, 8=greaterequal, 9=equal, 10=notequal,
	 * 11=min, 12=max, 13=and, 14=or, 15=minus1multiply, 16=minusnz,
	 * 17=modulus, 18=integer division}
	 */
	private static int getBinaryOp(ValueFunction fn) {
		if(fn instanceof Plus) return 0;
		else if(fn instanceof Minus) return 1;
		else if(fn instanceof Multiply) return 2;
		else if(fn instanceof Divide) return 3;
		else if(fn instanceof Power) return 4;
		else if(fn instanceof LessThan) return 5;
		else if(fn instanceof LessThanEquals) return 6;
		else if(fn instanceof GreaterThan) return 7;
		else if(fn instanceof GreaterThanEquals) return 8;
		else if(fn instanceof Equals) return 9;
		else if(fn instanceof NotEquals) return 10;
		else if(fn instanceof And) return 13;
		else if(fn instanceof Or) return 14;
		else if(fn instanceof Multiply2) return 2;
		else if(fn instanceof Power2) return 4;
		else if(fn instanceof Minus1Multiply) return 15;
		else if(fn instanceof MinusNz) return 16;
		else if(fn instanceof Modulus) return 17;
		else if(fn instanceof IntegerDivide) return 18;
		else if(fn instanceof Builtin && ((Builtin)fn).getBuiltinCode()==BuiltinCode.MIN) return 11;
		else if(fn instanceof Builtin && ((Builtin)fn).getBuiltinCode()==BuiltinCode.MAX) return 12;

		throw new DMLRuntimeException("The given value function is not supported:" + fn.getClass().getName());
	}

	/**
	 * Performs sparse and dense dgeam given two input matrices
	 * C = alpha* op( A ) + beta* op ( B )
	 * where op = transpose or not (specified by isLeftTransposed and isRightTransposed).
	 * To indicate a transpose operation, make sure in1 == in2 and isLeftTransposed == isRightTransposed == true
	 * @param ec execution context
	 * @param gCtx a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param in1 left input matrix
	 * @param in2 right input matrix
	 * @param outputName output variable name
	 * @param isLeftTransposed true if left matrix is transposed
	 * @param isRightTransposed true if right matrix is transposed
	 * @param alpha alpha
	 * @param beta beta
	 */
	private static void dgeam(ExecutionContext ec, GPUContext gCtx, String instName, MatrixObject in1, MatrixObject in2, String outputName,
			boolean isLeftTransposed, boolean isRightTransposed, double alpha, double beta) {
		if (ec.getGPUContext(0) != gCtx)
			throw new DMLRuntimeException("GPU : Invalid internal state, the GPUContext set with the ExecutionContext is not the same used to run this LibMatrixCUDA function");
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : dgeam" + ", GPUContext=" + gCtx);
		}

		Pointer alphaPtr = dataTypePointerTo(alpha);
		Pointer betaPtr = dataTypePointerTo(beta);
		int transa = isLeftTransposed ? CUBLAS_OP_T : CUBLAS_OP_N;
		int transb = isRightTransposed ? CUBLAS_OP_T : CUBLAS_OP_N;

		long outRLen = isLeftTransposed ? in1.getNumColumns() : in1.getNumRows();
		long outCLen = isLeftTransposed ? in1.getNumRows() : in1.getNumColumns();

		MatrixObject out = ec.getMatrixObject(outputName);
		boolean isSparse1 = isInSparseFormat(gCtx, in1);
		boolean isSparse2 = isInSparseFormat(gCtx, in2);

		// TODO: Implement sparse-dense matrix cublasDgeam kernel
		if(isSparse1 || isSparse2) {
			int m = (int)in1.getNumRows();
			int n = (int)in1.getNumColumns();
			// Invoke cuSparse when either are in sparse format
			// Perform sparse-sparse dgeam
			if (!isInSparseFormat(gCtx, in1)) {
				in1.getGPUObject(gCtx).denseToSparse();
			}
			CSRPointer A = in1.getGPUObject(gCtx).getJcudaSparseMatrixPtr();
			if (!isInSparseFormat(gCtx, in2)) {
				in2.getGPUObject(gCtx).denseToSparse();
			}
			CSRPointer B = in2.getGPUObject(gCtx).getJcudaSparseMatrixPtr();

			ec.allocateGPUMatrixObject(outputName, outRLen, outCLen);

			if (in1 == in2 && isLeftTransposed == true && isLeftTransposed == isRightTransposed) {
				// Special case for transpose

				int nnz = toInt(A.nnz);
				CSRPointer C = CSRPointer.allocateEmpty(gCtx, nnz, n);
				out.getGPUObject(gCtx).setSparseMatrixCudaPointer(C);
				cudaSupportFunctions.cusparsecsr2csc(getCusparseHandle(gCtx), m, n, nnz, A.val, A.rowPtr, A.colInd, C.val, C.colInd, C.rowPtr, cusparseAction.CUSPARSE_ACTION_NUMERIC, cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO);
			} else {
				// General case (cusparse does not support accept the transpose operator for dgeam)
				// TODO: to implement the transposed + dgeam for sparse matrices, they need to be converted to csc, which is effectively a tranpose
				if (isLeftTransposed || isRightTransposed) {
					throw new DMLRuntimeException(
						"Transpose in cusparseDcsrgeam not supported for sparse matrices on GPU");
				}

				CSRPointer C = CSRPointer.allocateForDgeam(gCtx, getCusparseHandle(gCtx), A, B, m, n);
				out.getGPUObject(gCtx).setSparseMatrixCudaPointer(C);
				//long sizeOfC = CSRPointer.estimateSize(C.nnz, out.getNumRows());
				cudaSupportFunctions.cusparsecsrgeam(getCusparseHandle(gCtx), m, n, alphaPtr, A.descr, toInt(A.nnz), A.val, A.rowPtr, A.colInd, betaPtr,
						B.descr, toInt(B.nnz), B.val, B.rowPtr, B.colInd, C.descr, C.val, C.rowPtr, C.colInd);
			}
		} else {
			// Dense-Dense dgeam

			int lda = toInt(in1.getNumColumns());
			int ldb = toInt(in2.getNumColumns());
			int m = toInt(in1.getNumColumns());
			int n = toInt(in2.getNumRows());
			if (isLeftTransposed && isRightTransposed) {
				m = toInt(in1.getNumRows());
				n = toInt(in2.getNumColumns());
			}
			else if (isLeftTransposed) {
				m = toInt(in1.getNumRows());
			} else if (isRightTransposed) {
				n = toInt(in2.getNumColumns());
			}
			int ldc = m;

			Pointer A = getDensePointer(gCtx, in1, instName);
			Pointer B = getDensePointer(gCtx, in2, instName);
			getDenseMatrixOutputForGPUInstruction(ec, instName, outputName, outRLen, outCLen);	// Allocated the dense output matrix
			Pointer C = getDensePointer(gCtx, out, instName);

			cudaSupportFunctions.cublasgeam(getCublasHandle(gCtx), transa, transb, m, n, alphaPtr, A, lda, betaPtr, B, ldb, C, ldc);
		}
	}
	
	/**
	 * Computes C = t(A)
	 * @param ec execution context
	 * @param gCtx gpu context
	 * @param instName name of the instruction
	 * @param A pointer to the input matrix
	 * @param C pointer to the output matrix
	 * @param numRowsA number of rows of the input matrix
	 * @param numColsA number of columns of the output matrix
	 * @throws DMLRuntimeException if error
	 */
	public static void denseTranspose(ExecutionContext ec, GPUContext gCtx, String instName, 
			Pointer A, Pointer C, long numRowsA, long numColsA) throws DMLRuntimeException {
		if (ec.getGPUContext(0) != gCtx)
			throw new DMLRuntimeException("GPU : Invalid internal state, the GPUContext set with the ExecutionContext is not the same used to run this LibMatrixCUDA function");
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : dense transpose" + ", GPUContext=" + gCtx);
		}
		// Dense-Dense dgeam
		int lda = toInt(numColsA);
		int ldb = lda;
		int m = toInt(numRowsA);
		int n = lda;
		int ldc = m;
		cudaSupportFunctions.cublasgeam(getCublasHandle(gCtx), CUBLAS_OP_T, CUBLAS_OP_T, m, n, one(), A, lda, zero(), A, ldb, C, ldc);
	}


	//********************************************************************/
	//****** End of Matrix-Matrix & Matrix-Scalar Functions **************/
	//********************************************************************/



	//********************************************************************/
	//************************ Re-org Functions **************************/
	//********************************************************************/

	/**
	 * Transposes the input matrix using cublasDgeam
	 *
	 * @param ec execution context
	 * @param gCtx a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param in input matrix
	 * @param outputName output matrix name
	 */
	public static void transpose(ExecutionContext ec, GPUContext gCtx, String instName, MatrixObject in, String outputName) {
		// C = alpha* op( A ) + beta* op ( B )
		// = 1.0 * A^T + 0.0 * A^T
		if (ec.getGPUContext(0) != gCtx)
			throw new DMLRuntimeException("GPU : Invalid internal state, the GPUContext set with the ExecutionContext is not the same used to run this LibMatrixCUDA function");
		dgeam(ec, gCtx, instName, in, in, outputName, true, true, 1.0, 0.0);
	}

	//********************************************************************/
	//******************* End of Re-org Functions ************************/
	//********************************************************************/

	public static int toInt(long num) {
		if(num >= Integer.MAX_VALUE || num <= Integer.MIN_VALUE) {
			throw new DMLRuntimeException("GPU : Exceeded supported size " + num);
		}
		return (int)num;
	}

	//********************************************************************/
	//**************** Matrix Manipulation Functions *********************/
	//********************************************************************/

	/**
	 * Method to perform rightIndex operation for a given lower and upper bounds in row and column dimensions.
	 *  
	 * @param ec current execution context
	 * @param gCtx current gpu context
	 * @param instName name of the instruction for maintaining statistics
	 * @param in1 input matrix object
	 * @param ixrange index range (0-based)
	 * @param outputName output matrix object
	 */
	public static void sliceOperations(ExecutionContext ec, GPUContext gCtx, String instName, MatrixObject in1,
			IndexRange ixrange, String outputName) {
		if (ec.getGPUContext(0) != gCtx)
			throw new DMLRuntimeException(
					"GPU : Invalid internal state, the GPUContext set with the ExecutionContext is not the same used to run this LibMatrixCUDA function");
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : sliceOperations" + ", GPUContext=" + gCtx);
		}

		int rl = (int) ixrange.rowStart;
		int ru = (int) ixrange.rowEnd;
		int cl = (int) ixrange.colStart;
		int cu = (int) ixrange.colEnd;
		if (rl < 0 || rl >= in1.getNumRows() || ru < rl || ru >= in1.getNumRows() || cl < 0
				|| cu >= in1.getNumColumns() || cu < cl || cu >= in1.getNumColumns()) {
			throw new DMLRuntimeException("Invalid values for matrix indexing: [" + (rl + 1) + ":" + (ru + 1) + ","
					+ (cl + 1) + ":" + (cu + 1) + "] " + "must be within matrix dimensions [" + in1.getNumRows() + ","
					+ in1.getNumColumns() + "]");
		}
		int len1 = toInt(in1.getNumColumns());
		if(isInSparseFormat(gCtx, in1)) {
			// Input in1 is in sparse format and output is in dense format
			MatrixObject out = getDenseMatrixOutputForGPUInstruction(ec, instName, outputName, ru - rl + 1, cu - cl + 1);
			CSRPointer inPointer = getSparsePointer(gCtx, in1, instName);
			Pointer outPointer = getDensePointer(gCtx, out, instName);
			sliceSparseDense(gCtx, instName, inPointer, outPointer, rl, ru, cl, cu, len1);
		}
		else {
			// Input in1 is in dense format (see inPointer)
			MatrixObject out = getDenseMatrixOutputForGPUInstruction(ec, instName, outputName, ru - rl + 1, cu - cl + 1);

			Pointer inPointer = getDensePointer(gCtx, in1, instName);
			Pointer outPointer = getDensePointer(gCtx, out, instName);
			sliceDenseDense(gCtx, instName, inPointer, outPointer, rl, ru, cl, cu, len1);
		}
	}
	
	/**
	 * Perform slice operation on dense input and output it in dense format
	 * 
	 * @param gCtx gpu context
	 * @param instName instruction name
	 * @param inPointer dense input pointer
	 * @param outPointer dense output pointer (doesnot need to be zeroed out)
	 * @param rl row lower
	 * @param ru row upper
	 * @param cl column lower
	 * @param cu column upper
	 * @param inClen input number of columns
	 */
	protected static void sliceDenseDense(GPUContext gCtx, String instName, Pointer inPointer, Pointer outPointer, 
			int rl, int ru, int cl, int cu, int inClen) {
		long retClen = cu - cl + 1;
		if (inClen == retClen) {
			cudaMemcpy(outPointer, inPointer.withByteOffset(rl * inClen * sizeOfDataType), (ru - rl + 1) * inClen
					* sizeOfDataType, cudaMemcpyDeviceToDevice);
		} else {
			long retRlen = ru - rl + 1;
			getCudaKernels(gCtx).launchKernel("slice_dense_dense", ExecutionConfig.getConfigForSimpleVectorOperations(toInt(retRlen*retClen)),
					inPointer, outPointer, rl, ru, cl, cu, inClen,  retRlen, retClen);
		}
	}
	
	/**
	 * Perform slice operation on sparse input and output it in dense format
	 * 
	 * @param gCtx gpu context
	 * @param instName instruction name
	 * @param inPointer sparse CSR input pointer
	 * @param outPointer dense output pointer (expected to be zeroed out)
	 * @param rl row lower
	 * @param ru row upper
	 * @param cl column lower
	 * @param cu column upper
	 * @param inClen number of columns of input matrix
	 */
	protected static void sliceSparseDense(GPUContext gCtx, String instName, CSRPointer inPointer, Pointer outPointer, 
			int rl, int ru, int cl, int cu, int inClen) {
		int size = getNnz(inPointer, rl, ru);
		// Return since nnz of the output is 0 as outPointer is expected to be zeroed out.
		if(size == 0) return;
		
		int retRlen = ru - rl + 1;
		int retClen = cu - cl + 1;
		
		String kernel = null;
		// Note: row-wise parallelization scheme iterates over input rows in single thread 
		// whereas nnz parallelization scheme iterates over number of output rows in single thread.
		if(inClen > 10 && retClen > 2*retRlen) {
			// Perform nnz parallelization for wide and short matrices
			kernel = "slice_sparse_dense_nnz";
		}
		else {
			size = retRlen;
			kernel = "slice_sparse_dense_row";
		}
		
		// Performs a slice operation where the input matrix is sparse and the output matrix is dense.
		// This function avoids unnecessary sparse to dense conversion of the input matrix.
		// We can generalize this later to output sparse matrix.
		getCudaKernels(gCtx).launchKernel(kernel, ExecutionConfig.getConfigForSimpleVectorOperations(size),
			inPointer.val, inPointer.rowPtr, inPointer.colInd, outPointer, rl, ru, cl, cu, retClen);
	}
	
	/**
	 * Returns the number of non-zeroes in the given range of rows
	 * 
	 * @param inPointer input CSR pointer
	 * @param rl lower row index (inclusive and zero-based)
	 * @param ru upper row index (inclusive and zero-based)
	 * @return number of non-zeroes
	 */
	private static int getNnz(CSRPointer inPointer, int rl, int ru) {
		int[] rlPtr = { -1 }; int[] ruPtr = { -1 };
		cudaMemcpy(Pointer.to(rlPtr), inPointer.rowPtr.withByteOffset(rl*Sizeof.INT), Sizeof.INT, cudaMemcpyDeviceToHost);
		cudaMemcpy(Pointer.to(ruPtr), inPointer.rowPtr.withByteOffset((ru+1)*Sizeof.INT), Sizeof.INT, cudaMemcpyDeviceToHost);
		return ruPtr[0] - rlPtr[0];
	}

	public static void cbind(ExecutionContext ec, GPUContext gCtx, String instName, MatrixObject in1, MatrixObject in2, String outputName) {
		if (ec.getGPUContext(0) != gCtx)
			throw new DMLRuntimeException("GPU : Invalid internal state, the GPUContext set with the ExecutionContext is not the same used to run this LibMatrixCUDA function");
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : cbind" + ", GPUContext=" + gCtx);
		}

		long rowsA = toInt(in1.getNumRows());
		long colsA = toInt(in1.getNumColumns());
		long rowsB = toInt(in2.getNumRows());
		long colsB = toInt(in2.getNumColumns());

		if (rowsA != rowsB) {
			throw new DMLRuntimeException("GPU : Invalid internal state - the rows must match up for a cbind operation");
		}

		// only Dense supported
		MatrixObject out = getDenseMatrixOutputForGPUInstruction(ec, instName, outputName, rowsA, colsA + colsB);
		Pointer C = getDensePointer(gCtx, out, instName);
		Pointer A = getDensePointer(gCtx, in1, instName);
		Pointer B = getDensePointer(gCtx, in2, instName);

		int maxRows = toInt(Math.max(rowsA, rowsB));
		int maxCols = toInt(Math.max(colsA, colsB));

		getCudaKernels(gCtx).launchKernel("cbind",
			ExecutionConfig.getConfigForSimpleMatrixOperations(maxRows, maxCols),
			A, B, C, rowsA, colsA, rowsB, colsB);
	}

	public static void rbind(ExecutionContext ec, GPUContext gCtx, String instName, MatrixObject in1, MatrixObject in2, String outputName) {
		if (ec.getGPUContext(0) != gCtx)
			throw new DMLRuntimeException("GPU : Invalid internal state, the GPUContext set with the ExecutionContext is not the same used to run this LibMatrixCUDA function");
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : rbind" + ", GPUContext=" + gCtx);
		}

		int rowsA = toInt(in1.getNumRows());
		int colsA = toInt(in1.getNumColumns());
		int rowsB = toInt(in2.getNumRows());
		int colsB = toInt(in2.getNumColumns());

		if (colsA != colsB){
			throw new DMLRuntimeException("GPU : Invalid internal state - the columns must match up for a rbind operation");
		}

		// only Dense supported
		MatrixObject out = getDenseMatrixOutputForGPUInstruction(ec, instName, outputName, rowsA + rowsB, colsA);
		Pointer C = getDensePointer(gCtx, out, instName);
		Pointer A = getDensePointer(gCtx, in1, instName);
		Pointer B = getDensePointer(gCtx, in2, instName);

		int maxRows = Math.max(rowsA, rowsB);
		int maxCols = Math.max(colsA, colsB);

		getCudaKernels(gCtx).launchKernel("rbind",
			ExecutionConfig.getConfigForSimpleMatrixOperations(maxRows, maxCols),
			A, B, C, rowsA, colsA, rowsB, colsB);
	}


	//********************************************************************/
	//*********** End of Matrix Manipulation Functions *******************/
	//********************************************************************/


	//********************************************************************/
	//************************ Builtin Functions *************************/
	//********************************************************************/

	/**
	 * Performs an "exp" operation on a matrix on the GPU
	 * @param ec	execution context
	 * @param gCtx a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param in1	input matrix
	 * @param outputName	output matrix name
	 */
	public static void exp(ExecutionContext ec, GPUContext gCtx, String instName, MatrixObject in1, String outputName) {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : exp" + ", GPUContext=" + gCtx);
		}
		// e^0 = 1, create a dense block full of 1s
		unaryOp(ec, gCtx, in1, "matrix_exp", 1, outputName, instName, GPUInstruction.MISC_TIMER_EXP_KERNEL);
	}

	/**
	 * Performs an "sqrt" operation on a matrix on the GPU
	 * @param ec	execution context
	 * @param gCtx a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param in1	input matrix
	 * @param outputName	output matrix name
	 */
	public static void sqrt(ExecutionContext ec, GPUContext gCtx, String instName, MatrixObject in1, String outputName) {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : sqrt" + ", GPUContext=" + gCtx);
		}
		// sqrt(0) = 0, create a dense block full of 0s
		unaryOp(ec, gCtx, in1, "matrix_sqrt", 0, outputName, instName, GPUInstruction.MISC_TIMER_SQRT_KERNEL);
	}

	/**
	 * Performs an "round" operation on a matrix on the GPU
	 * @param ec	execution context
	 * @param gCtx a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param in1	input matrix
	 * @param outputName	output matrix name
	 */
	public static void round(ExecutionContext ec, GPUContext gCtx, String instName, MatrixObject in1, String outputName) {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : round" + ", GPUContext=" + gCtx);
		}
		// round(0) = 0, create a dense block full of 0s
		unaryOp(ec, gCtx, in1, "matrix_round", 0, outputName, instName, GPUInstruction.MISC_TIMER_ROUND_KERNEL);
	}

	/**
	 * Performs an "abs" operation on a matrix on the GPU
	 * @param ec	execution context
	 * @param gCtx a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param in1	input matrix
	 * @param outputName	output matrix name
	 */
	public static void abs(ExecutionContext ec, GPUContext gCtx, String instName, MatrixObject in1, String outputName) {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : abs" + ", GPUContext=" + gCtx);
		}
		// abs(0) = 0, create a dense block full of 0s
		unaryOp(ec, gCtx, in1, "matrix_abs", 0, outputName, instName, GPUInstruction.MISC_TIMER_ABS_KERNEL);
	}

	/**
	 * Performs an "log" operation on a matrix on the GPU
	 * @param ec	execution context
	 * @param gCtx a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param in1	input matrix
	 * @param outputName	output matrix name
	 */
	public static void log(ExecutionContext ec, GPUContext gCtx, String instName, MatrixObject in1, String outputName) {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : log" + ", GPUContext=" + gCtx);
		}
		// log(0) = -Inf
		unaryOp(ec, gCtx, in1, "matrix_log", Double.NEGATIVE_INFINITY, outputName, instName, GPUInstruction.MISC_TIMER_LOG_KERNEL);
	}

	/**
	 * Performs an "floor" operation on a matrix on the GPU
	 * @param ec	execution context
	 * @param gCtx a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param in1	input matrix
	 * @param outputName	output matrix name
	 */
	public static void floor(ExecutionContext ec, GPUContext gCtx, String instName, MatrixObject in1, String outputName) {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : floor" + ", GPUContext=" + gCtx);
		}
		// floor(0) = 0
		unaryOp(ec, gCtx, in1, "matrix_floor", 0, outputName, instName, GPUInstruction.MISC_TIMER_FLOOR_KERNEL);
	}

	/**
	 * Performs an "ceil" operation on a matrix on the GPU
	 * @param ec	execution context
	 * @param gCtx a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param in1	input matrix
	 * @param outputName	output matrix name
	 */
	public static void ceil(ExecutionContext ec, GPUContext gCtx, String instName, MatrixObject in1, String outputName) {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : ceil" + ", GPUContext=" + gCtx);
		}
		// ceil(0) = 0
		unaryOp(ec, gCtx, in1, "matrix_ceil", 0, outputName, instName, GPUInstruction.MISC_TIMER_CEIL_KERNEL);
	}

	/**
	 * Performs an "sin" operation on a matrix on the GPU
	 * @param ec	execution context
	 * @param gCtx a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param in1	input matrix
	 * @param outputName	output matrix name
	 */
	public static void sin(ExecutionContext ec, GPUContext gCtx, String instName, MatrixObject in1, String outputName) {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : sin" + ", GPUContext=" + gCtx);
		}
		// sin(0) = 0
		unaryOp(ec, gCtx, in1, "matrix_sin", 0, outputName, instName, GPUInstruction.MISC_TIMER_SIN_KERNEL);
	}

	/**
	 * Performs an "cos" operation on a matrix on the GPU
	 * @param ec	execution context
	 * @param gCtx a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param in1	input matrix
	 * @param outputName	output matrix name
	 */
	public static void cos(ExecutionContext ec, GPUContext gCtx, String instName, MatrixObject in1, String outputName) {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : cos" + ", GPUContext=" + gCtx);
		}
		// cos(0) = 1
		unaryOp(ec, gCtx, in1, "matrix_cos", 1, outputName, instName, GPUInstruction.MISC_TIMER_COS_KERNEL);
	}

	/**
	 * Performs an "tan" operation on a matrix on the GPU
	 * @param ec	execution context
	 * @param gCtx a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param in1	input matrix
	 * @param outputName	output matrix name
	 */
	public static void tan(ExecutionContext ec, GPUContext gCtx, String instName, MatrixObject in1, String outputName) {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : tan" + ", GPUContext=" + gCtx);
		}
		// tan(0) = 0
		unaryOp(ec, gCtx, in1, "matrix_tan", 0, outputName, instName, GPUInstruction.MISC_TIMER_TAN_KERNEL);
	}
	
	/**
	 * Performs an "sinh" operation on a matrix on the GPU
	 * @param ec	execution context
	 * @param gCtx a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param in1	input matrix
	 * @param outputName	output matrix name
	 */
	public static void sinh(ExecutionContext ec, GPUContext gCtx, String instName, MatrixObject in1, String outputName) {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : sinh" + ", GPUContext=" + gCtx);
		}
		// sin(0) = 0
		unaryOp(ec, gCtx, in1, "matrix_sinh", 0, outputName, instName, GPUInstruction.MISC_TIMER_SINH_KERNEL);
	}

	/**
	 * Performs an "cosh" operation on a matrix on the GPU
	 * @param ec	execution context
	 * @param gCtx a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param in1	input matrix
	 * @param outputName	output matrix name
	 */
	public static void cosh(ExecutionContext ec, GPUContext gCtx, String instName, MatrixObject in1, String outputName) {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : cosh" + ", GPUContext=" + gCtx);
		}
		// cos(0) = 1
		unaryOp(ec, gCtx, in1, "matrix_cosh", 1, outputName, instName, GPUInstruction.MISC_TIMER_COSH_KERNEL);
	}

	/**
	 * Performs an "tanh" operation on a matrix on the GPU
	 * @param ec	execution context
	 * @param gCtx a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param in1	input matrix
	 * @param outputName	output matrix name
	 */
	public static void tanh(ExecutionContext ec, GPUContext gCtx, String instName, MatrixObject in1, String outputName) {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : tanh" + ", GPUContext=" + gCtx);
		}
		// tan(0) = 0
		unaryOp(ec, gCtx, in1, "matrix_tanh", 0, outputName, instName, GPUInstruction.MISC_TIMER_TANH_KERNEL);
	}

	/**
	 * Performs an "asin" operation on a matrix on the GPU
	 * @param ec	execution context
	 * @param gCtx a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param in1	input matrix
	 * @param outputName	output matrix name
	 */
	public static void asin(ExecutionContext ec, GPUContext gCtx, String instName, MatrixObject in1, String outputName) {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : asin" + ", GPUContext=" + gCtx);
		}
		// asin(0) = 0
		unaryOp(ec, gCtx, in1, "matrix_asin", 0, outputName, instName, GPUInstruction.MISC_TIMER_ASIN_KERNEL);
	}

	/**
	 * Performs an "acos" operation on a matrix on the GPU
	 * @param ec	execution context
	 * @param gCtx a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param in1	input matrix
	 * @param outputName	output matrix name
	 */
	public static void acos(ExecutionContext ec, GPUContext gCtx, String instName, MatrixObject in1, String outputName) {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : acos" + ", GPUContext=" + gCtx);
		}
		// acos(0) = PI/2
		unaryOp(ec, gCtx, in1, "matrix_acos", Math.PI/2.0, outputName, instName, GPUInstruction.MISC_TIMER_ACOS_KERNEL);
	}

	/**
	 * Performs an "atan" operation on a matrix on the GPU
	 * @param ec execution context
	 * @param gCtx a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param in1 input matrix
	 * @param outputName output matrix name
	 */
	public static void atan(ExecutionContext ec, GPUContext gCtx, String instName, MatrixObject in1, String outputName) {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : atan" + ", GPUContext=" + gCtx);
		}
		// atan(0) = 0
		unaryOp(ec, gCtx, in1, "matrix_atan", 0, outputName, instName, GPUInstruction.MISC_TIMER_ATAN_KERNEL);
	}

	/**
	 * Performs an "sign" operation on a matrix on the GPU
	 * @param ec	execution context
	 * @param gCtx a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param in1	input matrix
	 * @param outputName	output matrix name
	 */
	public static void sign(ExecutionContext ec, GPUContext gCtx, String instName, MatrixObject in1, String outputName) {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : sign" + ", GPUContext=" + gCtx);
		}
		// sign(0) = 0
		unaryOp(ec, gCtx, in1, "matrix_sign", 0, outputName, instName, GPUInstruction.MISC_TIMER_SIGN_KERNEL);
	}
	
	/**
	 * Performs an "sigmoid" operation on a matrix on the GPU
	 * @param ec	execution context
	 * @param gCtx a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param in1	input matrix
	 * @param outputName	output matrix name
	 */
	public static void sigmoid(ExecutionContext ec, GPUContext gCtx, String instName, MatrixObject in1, String outputName) {
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : sigmoid" + ", GPUContext=" + gCtx);
		}
		// sigmoid(0) = 0.5
		unaryOp(ec, gCtx, in1, "matrix_sigmoid", 0.5, outputName, instName, GPUInstruction.MISC_TIMER_SIGMOID_KERNEL);
	}

	/**
	 * Calculate the greatest common denominator recursively
	 * @param a a number
	 * @param b another number
	 * @return greatest common denominator
	 */
	private static int gcd(int a, int b)
	{
		return b == 0 ? a : gcd(b, a % b);
	}

	/**
	 * Utility method to print kernel timings
	 * @param time a Timing object
	 * @param kernelFunction Name of the executed kernel function
	 * @param total_duration Sum of all execution times
	 * @param cascade_blocks Number of blocks in the timed iteration of the cascade
	 * @return accumulated time
	 */
	private static double printKernelTiming(Timing time, String kernelFunction, double total_duration, int cascade_blocks) {
		if (LOG.isTraceEnabled()) {
			cudaDeviceSynchronize();
			double duration = time.stop();
			total_duration += duration;
			if(cascade_blocks > 0)
				LOG.trace("uop kernel function " + kernelFunction + " (cascading_blocks=" + cascade_blocks + ") executed in " + duration + "ms.");
			else
				LOG.trace("uop kernel function " + kernelFunction + " executed in " + duration + "ms.");
			time.start();
			return total_duration;
		}
		else
			return 0.0;
	}

	/**
	 * Cumulative scan
  	 * @param ec valid execution context
	 * @param gCtx a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param kernelFunction The name of the cuda kernel to call
	 * @param in input matrix
	 * @param outputName output matrix name
	 */
	public static void cumulativeScan(ExecutionContext ec, GPUContext gCtx, String instName, String kernelFunction, MatrixObject in, String outputName)	{
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : cumulative scan (" + GPUInstruction.MISC_TIMER_CUMULATIVE_SCAN_KERNEL + ") for instruction " + instName +
					" , GPUContext=" + gCtx);
		}

		int rows = toInt(in.getNumRows());
		int cols = toInt(in.getNumColumns());

		MatrixObject out = getDenseMatrixOutputForGPUInstruction(ec, instName, outputName, in.getNumRows(), in.getNumColumns());
		int[] tmp = getKernelParamsForCumScan(gCtx, rows, cols);
		int blocks_x = tmp[0], blocks_y = tmp[1], threads_x = tmp[2], sharedMem = 0, block_height = tmp[3];

		if (blocks_y > 1) {
			Timing time = new Timing(false);
			double duration = 0;
			double alloc_duration = 0;
			double total_duration = 0;
			if(LOG.isTraceEnabled()) { time.start(); }

			Pointer input = getDensePointer(gCtx, in, instName);
			Pointer output = getDensePointer(gCtx, out, instName);
			// storage for last value of each block
			Pointer blk_res = gCtx.allocate(instName, (long) cols * blocks_y * sizeOfDataType, false);

			alloc_duration = printKernelTiming(time, "allocation of temporary buffer (" +
				cols * blocks_y * sizeOfDataType + " bytes)", alloc_duration, 0);

			getCudaKernels(gCtx).launchKernel(kernelFunction + "_up_sweep", new ExecutionConfig(blocks_x, blocks_y,
				threads_x, 1, 0), input, blk_res, rows, cols, block_height);

			total_duration = printKernelTiming(time, kernelFunction + "_up_sweep", total_duration, 0);

			getCudaKernels(gCtx).launchKernel(kernelFunction + "_down_sweep", new ExecutionConfig(blocks_x, 1,
				threads_x, 1, 0), blk_res, blk_res, blk_res, blocks_y, cols, blocks_y);

			total_duration = printKernelTiming(time, kernelFunction + "_down_sweep", total_duration, 0);

			getCudaKernels(gCtx).launchKernel(kernelFunction + "_down_sweep", new ExecutionConfig(blocks_x,
				blocks_y, threads_x, 1, 0), input, output, blk_res, rows, cols, block_height);

			total_duration = printKernelTiming(time, "final cumulative_scan_down_sweep", total_duration, 0);
			if(LOG.isTraceEnabled()) { LOG.trace("total kernel execution time: " + total_duration + "ms."); }

			gCtx.cudaFreeHelper(instName, blk_res, DMLScript.EAGER_CUDA_FREE);
			if(LOG.isTraceEnabled()) {
				cudaDeviceSynchronize();
				duration = time.stop();
				alloc_duration += duration;
				LOG.trace("freeing of temporary buffer " + " executed in " + duration + "ms.");
				LOG.trace("total memory mgmt execution time: " + alloc_duration + "ms.");
				LOG.trace("total execution time (kernel + mem): " + (total_duration + alloc_duration) + "ms.");
			}
		}
		else {
			Pointer input = getDensePointer(gCtx, in, instName);
			Pointer output = getDensePointer(gCtx, out, instName);

			Timing time = new Timing(true);
			double duration = 0;

			getCudaKernels(gCtx).launchKernel(kernelFunction + "_down_sweep",
				new ExecutionConfig(blocks_x, 1, threads_x, 1, sharedMem), input, output, input, rows, cols, block_height);

			if(LOG.isTraceEnabled()) {
				cudaDeviceSynchronize();
				duration = time.stop();
				LOG.trace("total kernel execution time: " + duration + "ms.");
			}
		}
	}

	/**
	 * Cumulative sum-product kernel cascade invokation
	 * @param ec valid execution context
	 * @param gCtx a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param kernelFunction The name of the cuda kernel to call
	 * @param in input matrix
	 * @param outputName output matrix name
	 */
	public static void cumulativeSumProduct(ExecutionContext ec, GPUContext gCtx,
		String instName, String kernelFunction, MatrixObject in, String outputName)
	{
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : cumulative sum product for " + GPUInstruction.MISC_TIMER_CUMULATIVE_SCAN_KERNEL + ", GPUContext=" + gCtx);
		}

		Timing time = new Timing(false);

		double total_duration = 0, alloc_duration = 0;
		int rows = toInt(in.getNumRows());
		if(LOG.isTraceEnabled()) { time.start(); }

		if(rows > 128) {
			final int MAX_BLOCKS = getMaxBlocks(gCtx);
			ArrayList<Pointer> intermediate_buffers = new ArrayList<>();
			ArrayList<Integer> cb_list = new ArrayList<>();

			int block_height = 64;
			int blocks = (rows + block_height - 1) / block_height;
			if(blocks > MAX_BLOCKS) {
				blocks = MAX_BLOCKS;
				block_height = nextPow2((rows + (blocks - 1)) / blocks);
				blocks = (rows + block_height - 1) / block_height;
			}

			int threads = 1;
			int cascade_blocks = (blocks + block_height - 1) / block_height;
			cb_list.add(cascade_blocks);

			long total_mem_size = 0;
			while( cascade_blocks > 0) {
				long buf_size = 2L * block_height * cascade_blocks * sizeOfDataType;
				total_mem_size += buf_size;
				intermediate_buffers.add(gCtx.allocate(instName, buf_size, false));
				cascade_blocks = (cascade_blocks + block_height - 2) / block_height;
				if(cascade_blocks > 0)
					cb_list.add(cascade_blocks);
			}
			alloc_duration = printKernelTiming(time, "allocation of temporary buffer (" 
				+ total_mem_size + " bytes)", alloc_duration, 0);
			cascade_blocks = blocks;

			MatrixObject out = getDenseMatrixOutputForGPUInstruction(ec, instName, outputName, in.getNumRows(), 1);
			Pointer input = getDensePointer(gCtx, in, instName);
			Pointer output = getDensePointer(gCtx, out, instName);

			if(LOG.isTraceEnabled()) {
				LOG.trace("Launch configuration for cumulative aggregate: blocks=" + blocks +
						" block_height=" + block_height + " threads=" + threads);
			}

			getCudaKernels(gCtx).launchKernel(kernelFunction, new ExecutionConfig(blocks, threads), input,
				0, 0, intermediate_buffers.get(0), rows, block_height, 0);

			int cascade_level = 0;

			while(cascade_level < intermediate_buffers.size() - 1) {
				total_duration = printKernelTiming(time, kernelFunction, total_duration, cascade_blocks);
				cascade_blocks = cb_list.get(cascade_level);
				getCudaKernels(gCtx).launchKernel(kernelFunction, new ExecutionConfig(cascade_blocks, threads),
					intermediate_buffers.get(cascade_level), intermediate_buffers.get(cascade_level), 0,
					intermediate_buffers.get(cascade_level+1), rows / ((cascade_level+1) * block_height), block_height, 1);
				cascade_level++;
			}

			while (cascade_level > 0) {
				total_duration = printKernelTiming(time, kernelFunction, total_duration, cascade_blocks);
				cascade_level--;
				cascade_blocks = cb_list.get(cascade_level);

				getCudaKernels(gCtx).launchKernel(kernelFunction, new ExecutionConfig(cascade_blocks, threads),
					intermediate_buffers.get(cascade_level), intermediate_buffers.get(cascade_level),
					intermediate_buffers.get(cascade_level+1), 0, rows / ((cascade_level+1) * block_height), block_height, 2);
			}

			total_duration = printKernelTiming(time, kernelFunction, total_duration, cascade_blocks);
			getCudaKernels(gCtx).launchKernel(kernelFunction, new ExecutionConfig(blocks, threads), input,
				output, intermediate_buffers.get(0), 0, rows, block_height, 3);
			if(LOG.isTraceEnabled()) {
				cudaDeviceSynchronize();
				double duration = time.stop();
				total_duration += duration;
				LOG.trace("final cascade ("+ kernelFunction + ", cascading_blocks=" + blocks + ") executed in " + duration + "ms.");
				LOG.trace("total kernel execution time: " + total_duration + "ms.");
				time.start();
			}

			for(int j = 0; j < intermediate_buffers.size(); ++j)
				gCtx.cudaFreeHelper(instName, intermediate_buffers.get(j), DMLScript.EAGER_CUDA_FREE);

			if(LOG.isTraceEnabled()) {
				cudaDeviceSynchronize();
				double duration = time.stop();
				alloc_duration += duration;
				LOG.trace("freeing of temporary buffer " + " executed in " + duration + "ms.");
				LOG.trace("total memory mgmt execution time: " + alloc_duration + "ms.");
				LOG.trace("total execution time (kernel + mem): " + (total_duration + alloc_duration) + "ms.");
			}
		}
		else {
			MatrixObject out = getDenseMatrixOutputForGPUInstruction(ec, instName, outputName, in.getNumRows(), 1);
			Pointer input = getDensePointer(gCtx, in, instName);
			Pointer output = getDensePointer(gCtx, out, instName);
			getCudaKernels(gCtx).launchKernel(kernelFunction, new ExecutionConfig(1, 1), input,
				output, 0, 0, rows, rows);

			if(LOG.isTraceEnabled()) {
				cudaDeviceSynchronize();
				double duration = time.stop();
				total_duration += duration;
				LOG.trace("uop kernel function " + kernelFunction + " executed in " + duration + "ms.");
				LOG.trace("total kernel execution time: " + total_duration + "ms.");
			}
		}
	}

	/**
	 * Get threads, blocks and shared memory for cumulative scan along columns
	 * @param gCtx a valid {@link GPUContext}
	 * @param rows number of rows in input matrix
	 * @param cols number of columns in input matrix
	 * @return integer array containing {blocks, threads, shared memory}
	 */
	private static int[] getKernelParamsForCumScan(GPUContext gCtx, int rows, int cols) {
		final int MAX_THREADS = getMaxThreads(gCtx);
		final int WARP_SIZE = getWarpSize(gCtx);
		final int MAX_BLOCKS_Y = gCtx.getGPUProperties().maxGridSize[1];

		int t1 = cols % MAX_THREADS;
		int t2 = (t1 + WARP_SIZE - 1) / WARP_SIZE;
		int t3 = t2 * WARP_SIZE;
		int threads_x = gcd(MAX_THREADS, t3);
		int blocks_x = Math.max(1, (cols + (threads_x - 1)) / (threads_x));

		int block_height = Math.max(8, MAX_THREADS / threads_x);
		int blocks_y = (rows + block_height - 1) / block_height;
		int min_loop_length = 128;
		if(rows <= min_loop_length) {
			block_height = rows;
			blocks_y = 1;
		}

		if(blocks_y > MAX_BLOCKS_Y) {
			block_height = Math.max(2 ,2 * rows / MAX_BLOCKS_Y);
			blocks_y = (rows + block_height - 1) / block_height;
		}

		if(LOG.isTraceEnabled()) {
			LOG.trace("Launch configuration for cumulative aggregate: blocks_x=" + blocks_x + " blocks_y=" +
				blocks_y + " block_height=" + block_height + " threads_x=" + threads_x);
		}

		return new int[] {blocks_x, blocks_y, threads_x, block_height};
	}

	/**
	 * A helper function for all Unary ops (sqrt, abs, sin.. etc)
	 * @param ec valid execution context
	 * @param gCtx a valid {@link GPUContext}
	 * @param in1 input matrix
	 * @param kernel name of CUDA kernel for the unary op to execute
	 * @param sparseAndEmptyFillValue the result of the unary op on a completely empty input matrix block
	 * @param outputName output matrix name
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param kernelTimer the name of the timer to measure the kernel invocation
	 */
	private static void unaryOp(ExecutionContext ec, GPUContext gCtx, MatrixObject in1, String kernel, double sparseAndEmptyFillValue, String outputName, String instName, String kernelTimer) {
		if (ec.getGPUContext(0) != gCtx)
			throw new DMLRuntimeException("GPU : Invalid internal state, the GPUContext set with the ExecutionContext is not the same used to run this LibMatrixCUDA function");
		GPUObject in = in1.getGPUObject(gCtx);
		boolean isSparseAndEmpty = in.isSparseAndEmpty();
		if (isSparseAndEmpty) {
			MatrixObject out = ec.getMatrixObject(outputName);
			ec.allocateGPUMatrixObject(outputName, in1.getNumRows(), in1.getNumColumns());
			out.getGPUObject(gCtx).allocateAndFillDense(sparseAndEmptyFillValue);
		} else {
			// Dense
			MatrixObject out = getDenseMatrixOutputForGPUInstruction(ec, instName, outputName, in1.getNumRows(), in1.getNumColumns());
			Pointer output = getDensePointer(gCtx, out, instName);
			Pointer input = getDensePointer(gCtx, in1, instName);
			int size = toInt(in1.getNumColumns() * in1.getNumRows());
			getCudaKernels(gCtx).launchKernel(kernel, ExecutionConfig.getConfigForSimpleVectorOperations(size), input, output, size);
		}
	}

	/**
	 * Performs daxpy operation
	 *
	 * @param ec execution context
	 * @param gCtx a valid {@link GPUContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param in1 input matrix 1
	 * @param in2 input matrix 2
	 * @param outputName output matrix name
	 * @param constant pointer constant
	 */
	public static void axpy(ExecutionContext ec, GPUContext gCtx, String instName, MatrixObject in1, MatrixObject in2,
			String outputName,  double constant) {
		if (ec.getGPUContext(0) != gCtx)
			throw new DMLRuntimeException("GPU : Invalid internal state, the GPUContext set with the ExecutionContext is not the same used to run this LibMatrixCUDA function");
		Pointer A = getDensePointer(gCtx, in1, instName);
		Pointer B = getDensePointer(gCtx, in2, instName);
		MatrixObject out = ec.getMatrixObject(outputName);
		getDenseMatrixOutputForGPUInstruction(ec, instName, outputName, in1.getNumRows(), in1.getNumColumns());	// Allocated the dense output matrix
		Pointer C = getDensePointer(gCtx, out, instName);

		if(in1.getNumRows() == in2.getNumRows() && in1.getNumColumns() == in2.getNumColumns()) {
			if(LOG.isTraceEnabled()) {
				LOG.trace("GPU : cublasDaxpy" + ", GPUContext=" + gCtx);
			}

			// Matrix-Matrix daxpy
			long n = in1.getNumRows()*in2.getNumColumns(); // Since A is always a matrix
			Pointer alphaPtr = dataTypePointerTo(constant);
			// C <- A + alpha*B
			// becomes
			// C <- A
			// C <- alpha*B + C
			cudaMemcpy(C, A, n*sizeOfDataType, cudaMemcpyDeviceToDevice);
			cudaSupportFunctions.cublasaxpy(getCublasHandle(gCtx), toInt(n), alphaPtr, B, 1, C, 1);
		}
		else {
			if(LOG.isTraceEnabled()) {
				LOG.trace("GPU : daxpy_matrix_vector" + ", GPUContext=" + gCtx);
			}

			// Matrix-Vector daxpy
			// Note: Vector-Matrix operation is not supported
			// daxpy_matrix_vector(double* A,  double* B, double alpha, double* ret, int rlenA, int clenA, int rlenB, int clenB)
			int rlenA = toInt(in1.getNumRows()); int clenA =  toInt(in1.getNumColumns());
			int rlenB = toInt(in2.getNumRows()); int clenB =  toInt(in2.getNumColumns());
			getCudaKernels(gCtx).launchKernel("daxpy_matrix_vector",
				ExecutionConfig.getConfigForSimpleMatrixOperations(rlenA, clenA),
				A, B, constant, C, rlenA, clenA, rlenB, clenB);
		}
	}


	/**
	 * Implements the "solve" function for systemds Ax = B (A is of size m*n, B is of size m*1, x is of size n*1)
	 *
	 * @param ec         a valid {@link ExecutionContext}
	 * @param gCtx       a valid {@link GPUContext}
	 * @param instName   the invoking instruction's name for record {@link Statistics}.
	 * @param in1        input matrix A
	 * @param in2        input matrix B
	 * @param outputName name of the output matrix
	 */
	public static void solve(ExecutionContext ec, GPUContext gCtx, String instName, MatrixObject in1, MatrixObject in2, String outputName) {
		if (ec.getGPUContext(0) != gCtx)
			throw new DMLRuntimeException("GPU : Invalid internal state, the GPUContext set with the ExecutionContext is not the same used to run this LibMatrixCUDA function");

		// x = solve(A, b)
		if(LOG.isTraceEnabled()) {
			LOG.trace("GPU : solve" + ", GPUContext=" + gCtx);
		}
		
		GPUObject Aobj = in1.getGPUObject(gCtx);
		if (isInSparseFormat(gCtx, in1))
			Aobj.sparseToDense(instName);

		GPUObject bobj = in2.getGPUObject(gCtx);
		if (isInSparseFormat(gCtx, in2))
			bobj.sparseToDense(instName);

		int m = (int) in1.getNumRows();
		int n = (int) in1.getNumColumns();
		if ((int) in2.getNumRows() != m)
			throw new DMLRuntimeException("GPU : Incorrect input for solve(), rows in A should be the same as rows in B");
		if ((int) in2.getNumColumns() != 1)
			throw new DMLRuntimeException("GPU : Incorrect input for solve(), columns in B should be 1");


		// Copy over matrices and
		// convert dense matrices to row major
		// Operation in cuSolver and cuBlas are for column major dense matrices
		// and are destructive to the original input
		GPUObject ATobj = (GPUObject) Aobj.clone();
		ATobj.denseRowMajorToColumnMajor();
		Pointer A = ATobj.getDensePointer();

		GPUObject bTobj = (GPUObject) bobj.clone();
		bTobj.denseRowMajorToColumnMajor();
		

		Pointer b = bTobj.getDensePointer();

		// The following set of operations is done following the example in the cusolver documentation
		// http://docs.nvidia.com/cuda/cusolver/#ormqr-example1

		// step 3: query working space of geqrf and ormqr
		int[] lwork = {0};
		cudaSupportFunctions.cusolverDngeqrf_bufferSize(gCtx.getCusolverDnHandle(), m, n, A, m, lwork);
		
		// step 4: compute QR factorization
		Pointer work = gCtx.allocate(instName, (long) lwork[0] * sizeOfDataType, false);
		Pointer tau = gCtx.allocate(instName, (long) m * sizeOfDataType, false);
		Pointer devInfo = gCtx.allocate(instName, Sizeof.INT, false);
		cudaSupportFunctions.cusolverDngeqrf(gCtx.getCusolverDnHandle(), m, n, A, m, tau, work, lwork[0], devInfo);
		
		int[] qrError = {-1};
		cudaMemcpy(Pointer.to(qrError), devInfo, Sizeof.INT, cudaMemcpyDeviceToHost);
		if (qrError[0] != 0) {
			throw new DMLRuntimeException("GPU : Error in call to geqrf (QR factorization) as part of solve, argument " + qrError[0] + " was wrong");
		}

		// step 5: compute Q^T*B
		cudaSupportFunctions.cusolverDnormqr(gCtx.getCusolverDnHandle(), cublasSideMode.CUBLAS_SIDE_LEFT, cublasOperation.CUBLAS_OP_T, m, 1, n, A, m, tau, b, m, work, lwork[0], devInfo);
		cudaMemcpy(Pointer.to(qrError), devInfo, Sizeof.INT, cudaMemcpyDeviceToHost);
		if (qrError[0] != 0) {
			throw new DMLRuntimeException("GPU : Error in call to ormqr (to compuete Q^T*B after QR factorization) as part of solve, argument " + qrError[0] + " was wrong");
		}

		// step 6: compute x = R \ Q^T*B
		cudaSupportFunctions.cublastrsm(gCtx.getCublasHandle(),
			cublasSideMode.CUBLAS_SIDE_LEFT, cublasFillMode.CUBLAS_FILL_MODE_UPPER, cublasOperation.CUBLAS_OP_N, cublasDiagType.CUBLAS_DIAG_NON_UNIT,
			n, 1, dataTypePointerTo(1.0), A, m, b, m);
		
		bTobj.denseColumnMajorToRowMajor();
		
		// TODO  : Find a way to assign bTobj directly to the output and set the correct flags so as to not crash
		// There is an avoidable copy happening here
		MatrixObject out = getDenseMatrixOutputForGPUInstruction(ec, instName, outputName, in1.getNumColumns(), 1);
		cudaMemcpy(out.getGPUObject(gCtx).getDensePointer(), bTobj.getDensePointer(), n * 1 * sizeOfDataType, cudaMemcpyDeviceToDevice);

		gCtx.cudaFreeHelper(instName, work, DMLScript.EAGER_CUDA_FREE);
		gCtx.cudaFreeHelper(instName, tau, DMLScript.EAGER_CUDA_FREE);
		ATobj.clearData(instName, DMLScript.EAGER_CUDA_FREE);
		bTobj.clearData(instName, DMLScript.EAGER_CUDA_FREE);

		//debugPrintMatrix(b, n, 1);
	}

	//********************************************************************/
	//*****************  END OF Builtin Functions ************************/
	//********************************************************************/

	/**
	 * Helper method to get the output block (allocated on the GPU)
	 * Also records performance information into {@link Statistics}
	 * @param ec		active {@link ExecutionContext}
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param name	name of input matrix (that the {@link ExecutionContext} is aware of)
	 * @param numRows number of rows of output matrix object
	 * @param numCols number of columns of output matrix object
	 * @return	the matrix object
	 */
	public static MatrixObject getDenseMatrixOutputForGPUInstruction(ExecutionContext ec, String instName, String name,
		long numRows, long numCols)
	{
		return getDenseMatrixOutputForGPUInstruction(ec, instName, name, numRows, numCols, true);
	}

	public static MatrixObject getDenseMatrixOutputForGPUInstruction(ExecutionContext ec, String instName, String name,
		long numRows, long numCols, boolean initialize)
	{
		return ec.getDenseMatrixOutputForGPUInstruction(name, numRows, numCols, initialize).getKey();
	}

	/**
	 * Helper method to get the output block (allocated on the GPU)
	 * Also records performance information into {@link Statistics}
	 * @param ec		active {@link ExecutionContext}
	 * @param numRows number of rows of matrix object
	 * @param numCols number of columns of matrix object
	 * @param nnz number of non zeroes in output matrix
	 * @param instName the invoking instruction's name for record {@link Statistics}.
	 * @param name	name of input matrix (that the {@link ExecutionContext} is aware of)
	 * @param initialize memset to zero?
	 *
	 * @return	the matrix object
	 */
	private static MatrixObject getSparseMatrixOutputForGPUInstruction(ExecutionContext ec, long numRows, long numCols,
		long nnz, String instName, String name, boolean initialize)
	{
		return ec.getSparseMatrixOutputForGPUInstruction(name, numRows, numCols, nnz, initialize).getKey();
	}

	private static MatrixObject getSparseMatrixOutputForGPUInstruction(ExecutionContext ec, long numRows, long numCols,
		long nnz, String instName, String name)
	{
		return getSparseMatrixOutputForGPUInstruction(ec, numRows, numCols, nnz, instName, name, true);
	}

	/**
	 * Utility to compute number of non-zeroes on the GPU
	 * 
	 * @param gCtx the associated GPUContext
	 * @param densePtr device pointer to the dense matrix
	 * @param length length of the dense pointer
	 * @return the number of non-zeroes
	 */
	public static synchronized int computeNNZ(GPUContext gCtx, Pointer densePtr, int length) {
		return (int) reduceAll(gCtx, null, "compute_nnz", densePtr, length);
		// This is extremely slow
//		cusparseMatDescr matDescr = CSRPointer.getDefaultCuSparseMatrixDescriptor();
//		cusparseHandle cusparseHandle = gCtx.getCusparseHandle();
//		if(_TMP_NNZ_ROW_PTR == null) {
//			// As these are 4-byte pointers, using cudaMalloc directly so as not to include them in memory information.
//			_TMP_NNZ_ROW_PTR = new Pointer();
//			cudaMalloc(_TMP_NNZ_ROW_PTR, jcuda.Sizeof.INT);
//			_TMP_NNZ_PTR = new Pointer();
//			cudaMalloc(_TMP_NNZ_PTR, jcuda.Sizeof.INT);
//			// _TMP_NNZ_ROW_PTR = gCtx.allocate(jcuda.Sizeof.INT);
//			// _TMP_NNZ_PTR = gCtx.allocate(jcuda.Sizeof.INT);
//		}
//		// Output is in dense vector format, convert it to CSR
//		LibMatrixCUDA.cudaSupportFunctions.cusparsennz(cusparseHandle, cusparseDirection.CUSPARSE_DIRECTION_ROW, 1, length, matDescr, densePtr, 1,
//				_TMP_NNZ_ROW_PTR, _TMP_NNZ_PTR);
//		int[] nnzC = { -1 };
//		cudaMemcpy(Pointer.to(nnzC), _TMP_NNZ_PTR, jcuda.Sizeof.INT, cudaMemcpyDeviceToHost);
//		return nnzC[0];
	}
}
