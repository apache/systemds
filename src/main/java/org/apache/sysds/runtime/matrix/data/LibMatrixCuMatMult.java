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

import static jcuda.cudaDataType.CUDA_R_32F;
import static jcuda.cudaDataType.CUDA_R_64F;
import static jcuda.jcusparse.cusparseOperation.CUSPARSE_OPERATION_NON_TRANSPOSE;
import static jcuda.jcusparse.cusparseOperation.CUSPARSE_OPERATION_TRANSPOSE;
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;
import static jcuda.jcusparse.cusparseSpMMAlg.CUSPARSE_SPMM_ALG_DEFAULT;
import static org.apache.sysds.runtime.instructions.gpu.context.CSRPointer.getCSRMatrixInfo;
import static org.apache.sysds.runtime.instructions.gpu.context.CSRPointer.transposeCSR;

import jcuda.Pointer;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.instructions.gpu.context.CSRPointer;
import org.apache.sysds.runtime.instructions.gpu.context.GPUContext;
import org.apache.sysds.utils.Statistics;

import jcuda.jcusparse.cusparseHandle;
import jcuda.jcublas.cublasHandle;
import jcuda.jcublas.cublasOperation;
import jcuda.runtime.JCuda;

public class LibMatrixCuMatMult extends LibMatrixCUDA {

	private static final Log LOG = LogFactory.getLog(LibMatrixCuMatMult.class.getName());

	private static class CuMatMultParameters {
		/*
		 * For the operation, C = op(A) %*% op(B), the below parameters are used
		 * to invoke the corresponding kernels in CuBLAS and CuSPARSE.
		 * 
		 * All the below values have to be valid or else this class has to throw
		 * an exception. No special values like -1 for unknowns allowed.
		 */
		public int m; // number of rows of matrix op(A) and C.
		public int n; // number of columns of matrix op(B) and C.
		public int k; // number of columns of op(A) and rows of op(B).
		public int lda; // leading dimension of two-dimensional array used to
						// store the matrix A.
		public int ldb; // leading dimension of two-dimensional array used to
						// store matrix B.
		public int ldc; // leading dimension of a two-dimensional array used to
						// store the matrix C.
		public long leftNumRows; // number of rows of A
		public long leftNumCols; // number of cols of A
		public long rightNumRows; // number of rows of B
		public long rightNumCols; // number of cols of B
		private boolean isLeftTransposed; // is op(A) = t(A)
		private boolean isRightTransposed; // is op(B) = t(B)

		public CuMatMultParameters(long leftNumRows1, long leftNumCols1, long rightNumRows1, long rightNumCols1,
				boolean isLeftTransposed1, boolean isRightTransposed1) {
			leftNumRows = leftNumRows1;
			leftNumCols = leftNumCols1;
			rightNumRows = rightNumRows1;
			rightNumCols = rightNumCols1;
			isLeftTransposed = isLeftTransposed1;
			isRightTransposed = isRightTransposed1;
			setDimensions();
		}

		public void rowToColumnMajor() {
			// To compensate for the input matrices being in row-major format
			// instead of column-major (the way cublas expects)
			isRightTransposed = swap(isLeftTransposed, isLeftTransposed = isRightTransposed);
			rightNumCols = swap(leftNumRows, leftNumRows = rightNumCols);
			rightNumRows = swap(leftNumCols, leftNumCols = rightNumRows);
			setDimensions();
		}

		private void validate() {
			int k1 = toInt(isRightTransposed ? rightNumCols : rightNumRows);
			if (k != k1)
				throw new DMLRuntimeException("Dimension mismatch: " + k + " != " + k1 + " [" + leftNumRows + ","
						+ leftNumCols + "," + rightNumRows + "," + rightNumCols + "], " + isLeftTransposed + " "
						+ isRightTransposed);
		}

		private void setDimensions() {
			// Validate the dimensions
			m = toInt(isLeftTransposed ? leftNumCols : leftNumRows);
			n = toInt(isRightTransposed ? rightNumRows : rightNumCols);
			k = toInt(isLeftTransposed ? leftNumRows : leftNumCols);
			lda = isLeftTransposed ? k : m;
			ldb = isRightTransposed ? n : k;
			ldc = m;
			if (m == -1 || n == -1 || k == -1)
				throw new DMLRuntimeException("Incorrect dimensions");
		}
	}

	/**
	 * Matrix multiply on GPU Examines sparsity and shapes and routes call to
	 * appropriate method from cuBLAS or cuSparse C = op(A) x op(B)
	 *
	 * The user is expected to call
	 * ec.releaseMatrixOutputForGPUInstruction(outputName);
	 *
	 * @param ec
	 *            Current {@link ExecutionContext} instance
	 * @param gCtx
	 *            a valid {@link GPUContext}
	 * @param instName
	 *            name of the invoking instruction to record{@link Statistics}.
	 * @param left
	 *            Matrix A
	 * @param right
	 *            Matrix B
	 * @param outputName
	 *            Name of the output matrix C (in code generated after LOP
	 *            layer)
	 * @param isLeftTransposed
	 *            op for A, transposed or not
	 * @param isRightTransposed
	 *            op for B, tranposed or not
	 * @return output of matrix multiply
	 */
	public static MatrixObject matmult(ExecutionContext ec, GPUContext gCtx, String instName, MatrixObject left,
			MatrixObject right, String outputName, boolean isLeftTransposed, boolean isRightTransposed) {
		boolean isM1Sparse = isInSparseFormat(gCtx, left);
		boolean isM2Sparse = isInSparseFormat(gCtx, right);
		MatrixObject output = ec.getMatrixObject(outputName);
		long outRLen = isLeftTransposed ? left.getNumColumns() : left.getNumRows();
		long outCLen = isRightTransposed ? right.getNumRows() : right.getNumColumns();

		CuMatMultParameters params = new CuMatMultParameters(left.getNumRows(), left.getNumColumns(),
				right.getNumRows(), right.getNumColumns(), isLeftTransposed, isRightTransposed);

		if (isM1Sparse && isM2Sparse) {
			// -------------------------------------------------------------------------------------
			// sparse-sparse matrix multiplication
			params.validate();
			int transA = cusparseOp(isLeftTransposed);
			int transB = cusparseOp(isRightTransposed);
			int dataType = (sizeOfDataType == 4) ? CUDA_R_32F : CUDA_R_64F;

			// Step 1: Allocate output => sparse format
			ec.allocateGPUMatrixObject(outputName, outRLen, outCLen);
			// Step 2: Get the handles to sparse/dense pointers for left, right
			// and output
			CSRPointer A = left.getGPUObject(gCtx).getJcudaSparseMatrixPtr();
			CSRPointer B = right.getGPUObject(gCtx).getJcudaSparseMatrixPtr();
			// transpose if required
			// cusparseSpGEMM works only with CUSPARSE_OPERATION_NON_TRANSPOSE
			if(transA == CUSPARSE_OPERATION_TRANSPOSE) {
				A = transposeCSR(gCtx, getCusparseHandle(gCtx), A, params.k, params.m, dataType);
			}
			if(transB == CUSPARSE_OPERATION_TRANSPOSE) {
				B = transposeCSR(gCtx, getCusparseHandle(gCtx), B, params.n, params.k, dataType);
			}
			transA = CUSPARSE_OPERATION_NON_TRANSPOSE;
			transB = CUSPARSE_OPERATION_NON_TRANSPOSE;
			CSRPointer C = CSRPointer.allocateForMatrixMultiply(gCtx, getCusparseHandle(gCtx), A, transA, B, transB,
				params.m, params.n, params.k, dataType);
			// Step 3: Invoke the kernel
			cudaSupportFunctions.cusparsecsrgemm(getCusparseHandle(gCtx), transA, transB, CUSPARSE_SPMM_ALG_DEFAULT,
				A.spMatDescr, B.spMatDescr, C.spMatDescr, C.spgemmDesc);
			output.getGPUObject(gCtx).setSparseMatrixCudaPointer(C);
			// -------------------------------------------------------------------------------------
		} else if (!isM1Sparse && isM2Sparse) {
			// -------------------------------------------------------------------------------------
			// dense-sparse matrix multiplication
			// Step 1: Allocate output => dense format
			getDenseMatrixOutputForGPUInstruction(ec, instName, outputName, outRLen, outCLen);

			// Step 2: Get the handles to sparse/dense pointers for left, right
			// and output
			Pointer A = getDensePointer(gCtx, left, instName);
			CSRPointer B = right.getGPUObject(gCtx).getJcudaSparseMatrixPtr();
			Pointer C = getDensePointer(gCtx, output, instName);

			// Step 3: Invoke the kernel
			denseSparseMatMult(getCusparseHandle(gCtx), instName, C, A, B, params);
			// -------------------------------------------------------------------------------------
		} else if (isM1Sparse && !isM2Sparse) {
			// -------------------------------------------------------------------------------------
			// sparse-dense matrix multiplication
			// Step 1: Allocate output => dense format
			getDenseMatrixOutputForGPUInstruction(ec, instName, outputName, outRLen, outCLen);

			// Step 2: Get the handles to sparse/dense pointers for left, right
			// and output
			CSRPointer A = left.getGPUObject(gCtx).getJcudaSparseMatrixPtr();
			Pointer B = getDensePointer(gCtx, right, instName);
			Pointer C = getDensePointer(gCtx, output, instName);

			// Step 3: Invoke the kernel
			sparseDenseMatMult(gCtx, instName, C, A, B, left.getNumRows(), left.getNumColumns(), right.getNumRows(),
					right.getNumColumns(), outRLen, outCLen, isLeftTransposed, isRightTransposed);
			// -------------------------------------------------------------------------------------
		} else {
			// -------------------------------------------------------------------------------------
			// dense-dense matrix multiplication
			// Step 1: Allocate output => dense format
			getDenseMatrixOutputForGPUInstruction(ec, instName, outputName, outRLen, outCLen);

			// Step 2: Get the handles to sparse/dense pointers for left, right
			// and output
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
	 * Internal method to invoke the appropriate CuSPARSE kernel for matrix
	 * multiplication for operation: C = op(A) * op(B) This assumes B and C are
	 * allocated in dense row-major format and A is sparse.
	 * 
	 * Other than input and output, this method requires additional memory =
	 * outRLen * outCLen * sizeOfDataType
	 * 
	 * @param gCtx
	 *            a valid {@link GPUContext}
	 * @param instName
	 *            name of the invoking instruction to record{@link Statistics}.
	 * @param C
	 *            output matrix pointer
	 * @param A
	 *            left matrix pointer
	 * @param B
	 *            right matrix pointer
	 * @param leftNumRows
	 *            number of rows of A
	 * @param leftNumColumns
	 *            number of cols of A
	 * @param rightNumRows
	 *            number of rows of B
	 * @param rightNumColumns
	 *            number of cols of B
	 * @param outRLen
	 *            number of rows of C
	 * @param outCLen
	 *            number of cols of C
	 * @param isLeftTransposed
	 *            is op(A) = t(A)
	 * @param isRightTransposed
	 *            is op(B) = t(B)
	 */
	static void sparseDenseMatMult(GPUContext gCtx, String instName, Pointer C, CSRPointer A, Pointer B,
			long leftNumRows, long leftNumColumns, long rightNumRows, long rightNumColumns, long outRLen, long outCLen,
			boolean isLeftTransposed, boolean isRightTransposed) {
		// t(C) = t(B) %*% t(A)
		Pointer output = null;
		if (outRLen != 1 && outCLen != 1) {
			output = gCtx.allocate(instName, outRLen * outCLen * sizeOfDataType, false);
		} else {
			// no transpose required for vector output
			output = C;
		}
		CuMatMultParameters params = new CuMatMultParameters(rightNumRows, rightNumColumns, leftNumRows,
				leftNumColumns, !isRightTransposed, !isLeftTransposed);
		denseSparseMatMult(getCusparseHandle(gCtx), instName, output, B, A, params);
		if (outRLen != 1 && outCLen != 1) {
			// Transpose: C = t(output)
			cudaSupportFunctions.cublasgeam(gCtx.getCublasHandle(), cublasOperation.CUBLAS_OP_T, cublasOperation.CUBLAS_OP_T,
					toInt(outCLen), toInt(outRLen), one(), output, toInt(outRLen), zero(), new Pointer(),
					toInt(outRLen), C, toInt(outCLen));
			if (!DMLScript.EAGER_CUDA_FREE)
				JCuda.cudaDeviceSynchronize();
			gCtx.cudaFreeHelper(instName, output, DMLScript.EAGER_CUDA_FREE);
		}
	}

	/**
	 * Internal method to invoke the appropriate CuSPARSE kernel for matrix
	 * multiplication for operation: C = op(A) * op(B) This assumes B and C are
	 * allocated in dense row-major format and A is sparse.
	 * 
	 * @param handle
	 *            cusparse handle
	 * @param instName
	 *            name of the invoking instruction to record{@link Statistics}.
	 * @param C
	 *            output matrix pointer
	 * @param A
	 *            left matrix pointer
	 * @param B
	 *            right matrix pointer
	 * @param param
	 *            BLAS parameters
	 */
	private static void denseSparseMatMult(cusparseHandle handle, String instName, Pointer C, Pointer A, CSRPointer B,
			CuMatMultParameters param) {
		// Ignoring sparse vector dense matrix multiplication and dot product
		boolean isVector = (param.leftNumRows == 1 && !param.isLeftTransposed)
				|| (param.leftNumCols == 1 && param.isLeftTransposed);
		if (isVector) {
			LOG.debug(" GPU Sparse-Dense Matrix Vector ");
			int m = toInt(param.rightNumRows);
			int n = toInt(param.rightNumCols);
			int transa = reverseCusparseOp(cusparseOp(param.isLeftTransposed));
			cudaSupportFunctions.cusparsecsrmv(handle, transa, m, n, toInt(B.nnz), one(), B.spMatDescr, B.descr, B.val, B.rowPtr, B.colInd, A,
					zero(), C);
		} else {
			int m = toInt(param.rightNumRows);
			int k = toInt(param.rightNumCols);
			param.rowToColumnMajor();
			param.validate();
			int transa = reverseCusparseOp(cusparseOp(param.isLeftTransposed));
			int transb = cusparseOp(param.isRightTransposed);
			LOG.debug(" GPU Sparse-Dense Matrix Multiply (rhs transpose) ");
			cudaSupportFunctions.cusparsecsrmm2(handle, transa, transb, m, param.n, k, toInt(B.nnz), one(), B.descr, B.spMatDescr, B.val,
					B.rowPtr, B.colInd, A, param.ldb, zero(), C, param.ldc);
		}
	}

	/**
	 * Internal method to invoke the appropriate CuBLAS kernel for matrix
	 * multiplication for operation: C = op(A) * op(B) This assumes A, B and C
	 * are allocated in dense format. The caller is expected to invoke
	 * params.rowToColumnMajor().
	 * 
	 * @param handle
	 *            cublas handle
	 * @param instName
	 *            name of the invoking instruction to record{@link Statistics}.
	 * @param C
	 *            output matrix pointer
	 * @param A
	 *            left matrix pointer
	 * @param B
	 *            right matrix pointer
	 * @param param
	 *            BLAS parameters
	 */
	private static void denseDenseMatMult(cublasHandle handle, String instName, Pointer C, Pointer A, Pointer B,
			CuMatMultParameters param) {
		param.rowToColumnMajor();
		param.validate();
		int transa = cublasOp(param.isLeftTransposed);
		int transb = cublasOp(param.isRightTransposed);
		B = swap(A, A = B);
		if (param.m == 1 && param.n == 1) {
			// Vector product
			LOG.debug(" GPU Dense-dense Vector Product");
			double[] result = { 0 };
			cudaSupportFunctions.cublasdot(handle, param.k, A, 1, B, 1, Pointer.to(result));
			// By default in CuBlas V2, cublas pointer mode is set to
			// CUBLAS_POINTER_MODE_HOST.
			// This means that scalar values passed are on host (as opposed to
			// on device).
			// The result is copied from the host back to the device so that the
			// rest of
			// infrastructure can treat it uniformly.
			cudaMemcpy(C, Pointer.to(result), 1 * sizeOfDataType, cudaMemcpyHostToDevice);
		} else if (param.m == 1) {
			// Vector-matrix multiply
			LOG.debug(" GPU Dense Vector-Matrix Multiply");
			transb = reverseCublasOp(transb);
			int rightNumRows = (transb == CUSPARSE_OPERATION_TRANSPOSE) ? param.k : param.n;
			int rightNumCols = (transb == CUSPARSE_OPERATION_TRANSPOSE) ? param.n : param.k;
			cudaSupportFunctions.cublasgemv(handle, transb, rightNumRows, rightNumCols, one(), B, param.ldb, A, 1, zero(), C, 1);
		} else if (param.n == 1) {
			// Matrix-vector multiply
			LOG.debug(" GPU Dense Matrix-Vector Multiply");
			int leftNumRows = (transa == CUSPARSE_OPERATION_NON_TRANSPOSE) ? param.m : param.k;
			int leftNumCols = (transa == CUSPARSE_OPERATION_NON_TRANSPOSE) ? param.k : param.m;
			cudaSupportFunctions.cublasgemv(handle, transa, leftNumRows, leftNumCols, one(), A, param.lda, B, 1, zero(), C, 1);
		} else {
			LOG.debug(" GPU Dense-Dense Matrix Multiply ");
			cudaSupportFunctions.cublasgemm(handle, transa, transb, param.m, param.n, param.k, one(), A, param.lda, B, param.ldb,
					zero(), C, param.ldc);
		}
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

	/**
	 * Convenient wrapper to return appropriate cuSPARSE trans value
	 * 
	 * @param isTransposed
	 *            is op(input) = t(input)
	 * @return CUSPARSE_OPERATION_TRANSPOSE or CUSPARSE_OPERATION_NON_TRANSPOSE
	 */
	private static int cusparseOp(boolean isTransposed) {
		return isTransposed ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
	}

	/**
	 * Convenient wrapper to return appropriate cuBLAS trans value
	 * 
	 * @param isTransposed
	 *            is op(input) = t(input)
	 * @return CUBLAS_OP_T or CUBLAS_OP_N
	 */
	private static int cublasOp(boolean isTransposed) {
		return isTransposed ? cublasOperation.CUBLAS_OP_T : cublasOperation.CUBLAS_OP_N;
	}

	/**
	 * Flips the cuBLAS trans value
	 * 
	 * @param trans
	 *            can be CUBLAS_OP_T or CUBLAS_OP_N
	 * @return CUBLAS_OP_N if trans is CUBLAS_OP_T else CUBLAS_OP_T
	 */
	private static int reverseCublasOp(int trans) {
		return trans == cublasOperation.CUBLAS_OP_T ? cublasOperation.CUBLAS_OP_N : cublasOperation.CUBLAS_OP_T;
	}

	/**
	 * Flips the cuSPARSE trans value
	 * 
	 * @param trans
	 *            can be CUSPARSE_OPERATION_NON_TRANSPOSE or
	 *            CUSPARSE_OPERATION_TRANSPOSE
	 * @return CUSPARSE_OPERATION_NON_TRANSPOSE if trans is
	 *         CUSPARSE_OPERATION_TRANSPOSE else CUSPARSE_OPERATION_TRANSPOSE
	 */
	private static int reverseCusparseOp(int trans) {
		return trans == CUSPARSE_OPERATION_TRANSPOSE ? CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE;
	}
}
