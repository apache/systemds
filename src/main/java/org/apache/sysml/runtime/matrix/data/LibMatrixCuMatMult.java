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
package org.apache.sysml.runtime.matrix.data;

import static jcuda.jcusparse.cusparseOperation.CUSPARSE_OPERATION_NON_TRANSPOSE;
import static jcuda.jcusparse.cusparseOperation.CUSPARSE_OPERATION_TRANSPOSE;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.jcublas.cublasOperation;
import jcuda.jcusparse.JCusparse;
import jcuda.jcusparse.cusparseHandle;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.instructions.gpu.GPUInstruction;
import org.apache.sysml.runtime.instructions.gpu.context.CSRPointer;
import org.apache.sysml.runtime.instructions.gpu.context.GPUContext;
import org.apache.sysml.utils.GPUStatistics;
import org.apache.sysml.utils.Statistics;


public class LibMatrixCuMatMult extends LibMatrixCUDA {
	
	private static final Log LOG = LogFactory.getLog(LibMatrixCuMatMult.class.getName());
	
	private static class CuMatMultParameters {
		public int m; public int n; public int k;  
		public int lda;  public int ldb; public int ldc;
		public long leftNumRows; public long leftNumCols; public long rightNumRows; public long rightNumCols;
		private boolean isLeftTransposed; private  boolean isRightTransposed;
		public CuMatMultParameters(long leftNumRows1, long leftNumCols1, long rightNumRows1, long rightNumCols1, 
				boolean isLeftTransposed1, boolean isRightTransposed1) throws DMLRuntimeException {
			leftNumRows = leftNumRows1;
			leftNumCols = leftNumCols1;
			rightNumRows = rightNumRows1;
			rightNumCols = rightNumCols1;
			isLeftTransposed = isLeftTransposed1;
			isRightTransposed = isRightTransposed1;
			setDimensions();
		}
		
		public void rowToColumnMajor() throws DMLRuntimeException {
			// To compensate for the input matrices being in row-major format instead of column-major (the way cublas expects)
			isRightTransposed = swap(isLeftTransposed, isLeftTransposed = isRightTransposed);
			rightNumCols = swap(leftNumRows, leftNumRows=rightNumCols);
			rightNumRows = swap(leftNumCols, leftNumCols=rightNumRows);
			setDimensions();
		}
		
		private void validate() throws DMLRuntimeException {
			int k1 = toInt(isRightTransposed ? rightNumCols : rightNumRows);
			if(k != k1)
				throw new DMLRuntimeException("Dimension mismatch: " + k + " != " + k1 + " [" + 
						leftNumRows + "," + leftNumCols + "," + rightNumRows + "," + rightNumCols + "], " + isLeftTransposed + " " + isRightTransposed);
		}
		
		private void setDimensions() throws DMLRuntimeException {
			// Validate the dimensions
			m = toInt(isLeftTransposed ? leftNumCols : leftNumRows) ;
			n = toInt(isRightTransposed ? rightNumRows : rightNumCols);
			k = toInt(isLeftTransposed ? leftNumRows :  leftNumCols);
			lda = isLeftTransposed ?  k : m;
			ldb = isRightTransposed ? n : k;
			ldc = m;
			if(m == -1 || n == -1 || k == -1)
				throw new DMLRuntimeException("Incorrect dimensions");
		}
	}
	
	/**
	 * Matrix multiply on GPU
	 * Examines sparsity and shapes and routes call to appropriate method
	 * from cuBLAS or cuSparse
	 * C = op(A) x op(B)
	 * <p>
	 * Memory Requirements -
	 * Both dense - inputs, output, no intermediate
	 * Both sparse - inputs, output, no intermediate
	 * One sparse, one dense - inputs, output, intermediates - (input_dim1 * input_dim2) OR (input_dim1 * input_dim2 + input in sparse format)
	 *
	 * The user is expected to call ec.releaseMatrixOutputForGPUInstruction(outputName);
	 *
	 * @param ec                Current {@link ExecutionContext} instance
	 * @param gCtx              a valid {@link GPUContext}
	 * @param instName          name of the invoking instruction to record{@link Statistics}.
	 * @param left              Matrix A
	 * @param right             Matrix B
	 * @param outputName        Name of the output matrix C (in code generated after LOP layer)
	 * @param isLeftTransposed  op for A, transposed or not
	 * @param isRightTransposed op for B, tranposed or not
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 * @return output of matrix multiply
	 */
	public static MatrixObject matmult(ExecutionContext ec, GPUContext gCtx, String instName, 
			MatrixObject left, MatrixObject right, String outputName,
			boolean isLeftTransposed, boolean isRightTransposed) throws DMLRuntimeException {
		boolean isM1Sparse = isInSparseFormat(gCtx, left);
		boolean isM2Sparse = isInSparseFormat(gCtx, right);
		MatrixObject output = ec.getMatrixObject(outputName);
		long outRLen = isLeftTransposed ? left.getNumColumns() : left.getNumRows();
		long outCLen = isRightTransposed ? right.getNumRows() : right.getNumColumns();
		
		CuMatMultParameters params = new CuMatMultParameters(left.getNumRows(), left.getNumColumns(), 
				right.getNumRows(), right.getNumColumns(), isLeftTransposed, isRightTransposed);
		
		if(isM1Sparse && isM2Sparse) {
			// -------------------------------------------------------------------------------------
			// sparse-sparse matrix multiplication
			params.validate();
			int transa = cusparseOp(isLeftTransposed);
			int transb = cusparseOp(isRightTransposed);
			
			// Step 1: Allocate output => sparse format
			ec.allocateGPUMatrixObject(outputName, outRLen, outCLen);
			
			// Step 2: Get the handles to sparse/dense pointers for left, right and output
			CSRPointer A = left.getGPUObject(gCtx).getJcudaSparseMatrixPtr();
			CSRPointer B = right.getGPUObject(gCtx).getJcudaSparseMatrixPtr();
			long t0 = GPUStatistics.DISPLAY_STATISTICS ? System.nanoTime() : 0;
			CSRPointer C = CSRPointer.allocateForMatrixMultiply(gCtx, getCusparseHandle(gCtx), A, transa, B, transb, params.m, params.n, params.k);
			if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_SPARSE_ALLOCATE_LIB, System.nanoTime() - t0);
			
			// Step 3: Invoke the kernel
			long t1 = GPUStatistics.DISPLAY_STATISTICS ? System.nanoTime() : 0;
			JCusparse.cusparseDcsrgemm(getCusparseHandle(gCtx), transa, transb, params.m, params.n, params.k,
					A.descr, (int)A.nnz, A.val, A.rowPtr, A.colInd,
					B.descr, (int)B.nnz, B.val, B.rowPtr, B.colInd,
					C.descr, C.val, C.rowPtr, C.colInd);
			if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, GPUInstruction.MISC_TIMER_SPARSE_MATRIX_SPARSE_MATRIX_LIB, System.nanoTime() - t1);
			output.getGPUObject(gCtx).setSparseMatrixCudaPointer(C);
			// -------------------------------------------------------------------------------------
		}
		else if(!isM1Sparse && isM2Sparse) {
			// -------------------------------------------------------------------------------------
			// dense-sparse matrix multiplication
			// sparse matrix is very large, so it is not wise to convert it into dense format
			// C = op(A) * op(B) ... where A is dense and B is sparse
			// => t(C) = t(op(B)) * t(op(A))
			// Step 1: Allocate output => dense format
			getDenseMatrixOutputForGPUInstruction(ec, instName, outputName, outRLen, outCLen); 
			
			// Step 2: Get the handles to sparse/dense pointers for left, right and output
			Pointer A = getDensePointer(gCtx, left, instName);
			CSRPointer B = right.getGPUObject(gCtx).getJcudaSparseMatrixPtr();
			Pointer C = getDensePointer(gCtx, output, instName);
			
			// Step 3: Invoke the kernel
			denseSparseMatMult(getCusparseHandle(gCtx), instName, C, A, B, params);
			
		}
		else {
			// -------------------------------------------------------------------------------------
			// dense-dense matrix multiplication 
			if(isM1Sparse && !isM2Sparse) {
				LOG.debug("Potential OOM as conversion of sparse input to dense");
			}
			
			// Step 1: Allocate output => dense format
			getDenseMatrixOutputForGPUInstruction(ec, instName, outputName, outRLen, outCLen); 
			
			// Step 2: Get the handles to sparse/dense pointers for left, right and output
			Pointer A = getDensePointer(gCtx, left, instName);
			Pointer B = getDensePointer(gCtx, right, instName);
			Pointer C = getDensePointer(gCtx, output, instName);
			
			// Step 3: Invoke the kernel
			denseDenseMatMult(getCublasHandle(gCtx), instName, C, A, B, params);
			// -------------------------------------------------------------------------------------
		}
		return output;
	}
	
	
	/**
	 * Internal method to invoke the appropriate CuSPARSE kernel for matrix multiplication for operation: C = op(A) * op(B)
	 * This assumes A and C are allocated in dense row-major format and A is sparse.
	 * 
	 * @param handle cusparse handle
	 * @param instName name of the invoking instruction to record{@link Statistics}.
	 * @param C output matrix pointer
	 * @param A left matrix pointer
	 * @param B right matrix pointer
	 * @param param BLAS parameters
	 * @throws DMLRuntimeException if error
	 */
	private static void denseSparseMatMult(cusparseHandle handle, String instName, Pointer C, Pointer A, CSRPointer B,  
			CuMatMultParameters param) throws DMLRuntimeException {
		long t0 = GPUStatistics.DISPLAY_STATISTICS ? System.nanoTime() : 0;
		String kernel = GPUInstruction.MISC_TIMER_SPARSE_MATRIX_DENSE_MATRIX_LIB;
		// Ignoring sparse vector dense matrix multiplication and dot product
		if(param.leftNumRows == 1) {
			LOG.debug(" GPU Sparse-Dense Matrix Vector ");
			int m = toInt(param.rightNumRows);
			int n = toInt(param.rightNumCols);
			int transa = reverseCusparseOp(cusparseOp(param.isLeftTransposed));
			JCusparse.cusparseDcsrmv(handle, transa, m, n, 
					toInt(B.nnz), one(), B.descr, B.val, B.rowPtr, B.colInd, 
					A, zero(), C);
			kernel = GPUInstruction.MISC_TIMER_SPARSE_MATRIX_DENSE_VECTOR_LIB;
		}
		else {
		 	int m = toInt(param.rightNumRows);
			int k = toInt(param.rightNumCols);
			param.rowToColumnMajor(); param.validate();
			int transa = reverseCusparseOp(cusparseOp(param.isLeftTransposed));
			int transb = cusparseOp(param.isRightTransposed);
			LOG.debug(" GPU Sparse-Dense Matrix Multiply (rhs transpose) ");
			JCusparse.cusparseDcsrmm2(handle, transa, transb, m, param.n, k, 
					toInt(B.nnz), one(), B.descr, B.val, B.rowPtr, B.colInd, 
					A, param.ldb, zero(), C, param.ldc);
		}
		if(GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, kernel, System.nanoTime() - t0);
	}
	
	/**
	 * Internal method to invoke the appropriate CuBLAS kernel for matrix multiplication for operation: C = op(A) * op(B)
	 * This assumes A, B and C are allocated in dense format.
	 * The caller is expected to invoke params.rowToColumnMajor().
	 * 
	 * @param handle cublas handle
	 * @param instName name of the invoking instruction to record{@link Statistics}.
	 * @param C output matrix pointer
	 * @param A left matrix pointer
	 * @param B right matrix pointer
	 * @param param BLAS parameters
	 * @throws DMLRuntimeException if error
	 */
	private static void denseDenseMatMult(cublasHandle handle, String instName, Pointer C, Pointer A, Pointer B, 
			CuMatMultParameters param) throws DMLRuntimeException {
		long t0 = GPUStatistics.DISPLAY_STATISTICS ? System.nanoTime() : 0;
		String kernel = null;
		param.rowToColumnMajor(); param.validate();
		int transa = cublasOp(param.isLeftTransposed);
		int transb = cublasOp(param.isRightTransposed);
		B = swap(A, A=B);
		if (param.m == 1 && param.n == 1){
			// Vector product
			LOG.debug(" GPU Dense-dense Vector Product");
			double[] result = {0};
			JCublas2.cublasDdot(handle, param.k, A, 1, B, 1, Pointer.to(result));
			// By default in CuBlas V2, cublas pointer mode is set to CUBLAS_POINTER_MODE_HOST.
			// This means that scalar values passed are on host (as opposed to on device).
			// The result is copied from the host back to the device so that the rest of
			// infrastructure can treat it uniformly.
			cudaMemcpy(C, Pointer.to(result), 1 * Sizeof.DOUBLE, cudaMemcpyHostToDevice);
			kernel = GPUInstruction.MISC_TIMER_DENSE_DOT_LIB;
		} else if (param.m == 1) {
			// Vector-matrix multiply
			LOG.debug(" GPU Dense Vector-Matrix Multiply");
			transb = reverseCublasOp(transb);
			int rightNumRows = (transb == CUSPARSE_OPERATION_TRANSPOSE) ?  param.k : param.n;
			int rightNumCols = (transb == CUSPARSE_OPERATION_TRANSPOSE) ? param.n : param.k;
			JCublas2.cublasDgemv(handle, transb, rightNumRows, rightNumCols, one(), B, param.ldb, A, 1, zero(), C, 1);
			kernel = GPUInstruction.MISC_TIMER_DENSE_VECTOR_DENSE_MATRIX_LIB;
		} else if (param.n == 1){
			// Matrix-vector multiply
			LOG.debug(" GPU Dense Matrix-Vector Multiply");
			int leftNumRows = (transa == CUSPARSE_OPERATION_NON_TRANSPOSE) ?  param.m : param.k;
			int leftNumCols = (transa == CUSPARSE_OPERATION_NON_TRANSPOSE) ?  param.k : param.m;
			JCublas2.cublasDgemv(handle, transa, leftNumRows, leftNumCols, one(), A, param.lda, B, 1, zero(), C, 1);
			kernel = GPUInstruction.MISC_TIMER_DENSE_MATRIX_DENSE_VECTOR_LIB;
		} else {
			LOG.debug(" GPU Dense-Dense Matrix Multiply ");
			JCublas2.cublasDgemm(handle, transa, transb, param.m, param.n, param.k, one(), A, param.lda, B, param.ldb, zero(), C, param.ldc);
			kernel = GPUInstruction.MISC_TIMER_DENSE_MATRIX_DENSE_MATRIX_LIB;
		}
		if (GPUStatistics.DISPLAY_STATISTICS) GPUStatistics.maintainCPMiscTimes(instName, kernel, System.nanoTime() - t0);
	}
	
	// Convenient methods to swap two values
	// Usage: y = swap(x, x=y);
	private static long swap(long x, long y) {
		return x;
	}
	
	private static boolean swap(boolean x, boolean y) {
		return x;
	}
	
	private static Pointer swap(Pointer x, Pointer y) {
		return x;
	}
	
	private static int cusparseOp(boolean isTransposed) {
		return isTransposed ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
	}
	
	private static int cublasOp(boolean isTransposed) {
		return isTransposed ? cublasOperation.CUBLAS_OP_T : cublasOperation.CUBLAS_OP_N;
	}
	
	private static int reverseCublasOp(int trans) {
		return trans == cublasOperation.CUBLAS_OP_T ? cublasOperation.CUBLAS_OP_N : cublasOperation.CUBLAS_OP_T;
	}
	
	private static int reverseCusparseOp(int trans) {
		return trans == CUSPARSE_OPERATION_TRANSPOSE ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE;
	}
}
