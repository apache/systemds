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

package org.apache.sysds.cujava.cusparse;

import org.apache.sysds.cujava.CuJavaLibLoader;
import org.apache.sysds.cujava.CudaException;
import org.apache.sysds.cujava.Pointer;

public class CuJavaCusparse {

	private static boolean exceptionsEnabled = false;

	private static final String LIB_BASE = "cujava_cusparse";

	private CuJavaCusparse() {

	}

	static {
		CuJavaLibLoader.load(LIB_BASE);
	}

	private static int checkCusparseStatus(int result) {
		if(exceptionsEnabled && result != cusparseStatus.CUSPARSE_STATUS_SUCCESS) {
			throw new CudaException(cusparseStatus.statusString(result));
		}
		return result;
	}

	public static void setExceptionsEnabled(boolean enabled) {
		exceptionsEnabled = enabled;
	}

	public static int cusparseSpGEMM_copy(cusparseHandle handle, int opA, int opB, Pointer alpha,
		cusparseConstSpMatDescr matA, cusparseConstSpMatDescr matB, Pointer beta, cusparseSpMatDescr matC,
		int computeType, int alg, cusparseSpGEMMDescr spgemmDescr) {
		return checkCusparseStatus(
			cusparseSpGEMM_copyNative(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, spgemmDescr));
	}

	private static native int cusparseSpGEMM_copyNative(cusparseHandle handle, int opA, int opB, Pointer alpha,
		cusparseConstSpMatDescr matA, cusparseConstSpMatDescr matB, Pointer beta, cusparseSpMatDescr matC,
		int computeType, int alg, cusparseSpGEMMDescr spgemmDescr);

	public static int cusparseGetMatIndexBase(cusparseMatDescr descrA) {
		return checkCusparseStatus(cusparseGetMatIndexBaseNative(descrA));
	}

	private static native int cusparseGetMatIndexBaseNative(cusparseMatDescr descrA);

	public static int cusparseCreateCsr(cusparseSpMatDescr spMatDescr, long rows, long cols, long nnz,
		Pointer csrRowOffsets, Pointer csrColInd, Pointer csrValues, int csrRowOffsetsType, int csrColIndType,
		int idxBase, int valueType) {
		return checkCusparseStatus(
			cusparseCreateCsrNative(spMatDescr, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues, csrRowOffsetsType,
				csrColIndType, idxBase, valueType));
	}

	private static native int cusparseCreateCsrNative(cusparseSpMatDescr spMatDescr, long rows, long cols, long nnz,
		Pointer csrRowOffsets, Pointer csrColInd, Pointer csrValues, int csrRowOffsetsType, int csrColIndType,
		int idxBase, int valueType);

	public static int cusparseCreateDnVec(cusparseDnVecDescr dnVecDescr, long size, Pointer values, int valueType) {
		return checkCusparseStatus(cusparseCreateDnVecNative(dnVecDescr, size, values, valueType));
	}

	private static native int cusparseCreateDnVecNative(cusparseDnVecDescr dnVecDescr, long size, Pointer values,
		int valueType);

	public static int cusparseSpMV_bufferSize(cusparseHandle handle, int opA, Pointer alpha,
		cusparseConstSpMatDescr matA, cusparseConstDnVecDescr vecX, Pointer beta, cusparseDnVecDescr vecY,
		int computeType, int alg, long[] bufferSize) {
		return checkCusparseStatus(
			cusparseSpMV_bufferSizeNative(handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg, bufferSize));
	}

	private static native int cusparseSpMV_bufferSizeNative(cusparseHandle handle, int opA, Pointer alpha,
		cusparseConstSpMatDescr matA, cusparseConstDnVecDescr vecX, Pointer beta, cusparseDnVecDescr vecY,
		int computeType, int alg, long[] bufferSize);

	public static int cusparseSpMV(cusparseHandle handle, int opA, Pointer alpha, cusparseConstSpMatDescr matA,
		cusparseConstDnVecDescr vecX, Pointer beta, cusparseDnVecDescr vecY, int computeType, int alg,
		Pointer externalBuffer) {
		return checkCusparseStatus(
			cusparseSpMVNative(handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg, externalBuffer));
	}

	private static native int cusparseSpMVNative(cusparseHandle handle, int opA, Pointer alpha,
		cusparseConstSpMatDescr matA, cusparseConstDnVecDescr vecX, Pointer beta, cusparseDnVecDescr vecY,
		int computeType, int alg, Pointer externalBuffer);

	public static int cusparseDestroy(cusparseHandle handle) {
		return checkCusparseStatus(cusparseDestroyNative(handle));
	}

	private static native int cusparseDestroyNative(cusparseHandle handle);

	public static int cusparseDestroyDnVec(cusparseConstDnVecDescr dnVecDescr) {
		return checkCusparseStatus(cusparseDestroyDnVecNative(dnVecDescr));
	}

	private static native int cusparseDestroyDnVecNative(cusparseConstDnVecDescr dnVecDescr);

	public static int cusparseDestroyDnMat(cusparseConstDnMatDescr dnMatDescr) {
		return checkCusparseStatus(cusparseDestroyDnMatNative(dnMatDescr));
	}

	private static native int cusparseDestroyDnMatNative(cusparseConstDnMatDescr dnMatDescr);

	public static int cusparseDestroySpMat(cusparseConstSpMatDescr spMatDescr) {
		return checkCusparseStatus(cusparseDestroySpMatNative(spMatDescr));
	}

	private static native int cusparseDestroySpMatNative(cusparseConstSpMatDescr spMatDescr);

	public static int cusparseSpMM(cusparseHandle handle, int opA, int opB, Pointer alpha, cusparseConstSpMatDescr matA,
		cusparseConstDnMatDescr matB, Pointer beta, cusparseDnMatDescr matC, int computeType, int alg,
		Pointer externalBuffer) {
		return checkCusparseStatus(
			cusparseSpMMNative(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg, externalBuffer));
	}

	private static native int cusparseSpMMNative(cusparseHandle handle, int opA, int opB, Pointer alpha,
		cusparseConstSpMatDescr matA, cusparseConstDnMatDescr matB, Pointer beta, cusparseDnMatDescr matC,
		int computeType, int alg, Pointer externalBuffer);

	public static int cusparseSpMM_bufferSize(cusparseHandle handle, int opA, int opB, Pointer alpha,
		cusparseConstSpMatDescr matA, cusparseConstDnMatDescr matB, Pointer beta, cusparseDnMatDescr matC,
		int computeType, int alg, long[] bufferSize) {
		return checkCusparseStatus(
			cusparseSpMM_bufferSizeNative(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg,
				bufferSize));
	}

	private static native int cusparseSpMM_bufferSizeNative(cusparseHandle handle, int opA, int opB, Pointer alpha,
		cusparseConstSpMatDescr matA, cusparseConstDnMatDescr matB, Pointer beta, cusparseDnMatDescr matC,
		int computeType, int alg, long[] bufferSize);

	public static int cusparseCreateDnMat(cusparseDnMatDescr dnMatDescr, long rows, long cols, long ld, Pointer values,
		int valueType, int order) {
		return checkCusparseStatus(cusparseCreateDnMatNative(dnMatDescr, rows, cols, ld, values, valueType, order));
	}

	private static native int cusparseCreateDnMatNative(cusparseDnMatDescr dnMatDescr, long rows, long cols, long ld,
		Pointer values, int valueType, int order);

	public static int cusparseCsrSetPointers(cusparseSpMatDescr spMatDescr, Pointer csrRowOffsets, Pointer csrColInd,
		Pointer csrValues) {
		return checkCusparseStatus(cusparseCsrSetPointersNative(spMatDescr, csrRowOffsets, csrColInd, csrValues));
	}

	private static native int cusparseCsrSetPointersNative(cusparseSpMatDescr spMatDescr, Pointer csrRowOffsets,
		Pointer csrColInd, Pointer csrValues);

	public static int cusparseCsr2cscEx2(cusparseHandle handle, int m, int n, int nnz, Pointer csrVal,
		Pointer csrRowPtr, Pointer csrColInd, Pointer cscVal, Pointer cscColPtr, Pointer cscRowInd, int valType,
		int copyValues, int idxBase, int alg, Pointer buffer) {
		return checkCusparseStatus(
			cusparseCsr2cscEx2Native(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscColPtr, cscRowInd,
				valType, copyValues, idxBase, alg, buffer));
	}

	private static native int cusparseCsr2cscEx2Native(cusparseHandle handle, int m, int n, int nnz, Pointer csrVal,
		Pointer csrRowPtr, Pointer csrColInd, Pointer cscVal, Pointer cscColPtr, Pointer cscRowInd, int valType,
		int copyValues, int idxBase, int alg, Pointer buffer);

	public static int cusparseCsr2cscEx2_bufferSize(cusparseHandle handle, int m, int n, int nnz, Pointer csrVal,
		Pointer csrRowPtr, Pointer csrColInd, Pointer cscVal, Pointer cscColPtr, Pointer cscRowInd, int valType,
		int copyValues, int idxBase, int alg, long[] bufferSize) {
		return checkCusparseStatus(
			cusparseCsr2cscEx2_bufferSizeNative(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscColPtr,
				cscRowInd, valType, copyValues, idxBase, alg, bufferSize));
	}

	private static native int cusparseCsr2cscEx2_bufferSizeNative(cusparseHandle handle, int m, int n, int nnz,
		Pointer csrVal, Pointer csrRowPtr, Pointer csrColInd, Pointer cscVal, Pointer cscColPtr, Pointer cscRowInd,
		int valType, int copyValues, int idxBase, int alg, long[] bufferSize);

	public static int cusparseDcsrgeam2(cusparseHandle handle, int m, int n, Pointer alpha, cusparseMatDescr descrA,
		int nnzA, Pointer csrSortedValA, Pointer csrSortedRowPtrA, Pointer csrSortedColIndA, Pointer beta,
		cusparseMatDescr descrB, int nnzB, Pointer csrSortedValB, Pointer csrSortedRowPtrB, Pointer csrSortedColIndB,
		cusparseMatDescr descrC, Pointer csrSortedValC, Pointer csrSortedRowPtrC, Pointer csrSortedColIndC,
		Pointer pBuffer) {
		return checkCusparseStatus(
			cusparseDcsrgeam2Native(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA,
				csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC,
				csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer));
	}

	private static native int cusparseDcsrgeam2Native(cusparseHandle handle, int m, int n, Pointer alpha,
		cusparseMatDescr descrA, int nnzA, Pointer csrSortedValA, Pointer csrSortedRowPtrA, Pointer csrSortedColIndA,
		Pointer beta, cusparseMatDescr descrB, int nnzB, Pointer csrSortedValB, Pointer csrSortedRowPtrB,
		Pointer csrSortedColIndB, cusparseMatDescr descrC, Pointer csrSortedValC, Pointer csrSortedRowPtrC,
		Pointer csrSortedColIndC, Pointer pBuffer);

	public static int cusparseDcsrgeam2_bufferSizeExt(cusparseHandle handle, int m, int n, Pointer alpha,
		cusparseMatDescr descrA, int nnzA, Pointer csrSortedValA, Pointer csrSortedRowPtrA, Pointer csrSortedColIndA,
		Pointer beta, cusparseMatDescr descrB, int nnzB, Pointer csrSortedValB, Pointer csrSortedRowPtrB,
		Pointer csrSortedColIndB, cusparseMatDescr descrC, Pointer csrSortedValC, Pointer csrSortedRowPtrC,
		Pointer csrSortedColIndC, long[] pBufferSizeInBytes) {
		return checkCusparseStatus(
			cusparseDcsrgeam2_bufferSizeExtNative(handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA,
				csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC,
				csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBufferSizeInBytes));
	}

	private static native int cusparseDcsrgeam2_bufferSizeExtNative(cusparseHandle handle, int m, int n, Pointer alpha,
		cusparseMatDescr descrA, int nnzA, Pointer csrSortedValA, Pointer csrSortedRowPtrA, Pointer csrSortedColIndA,
		Pointer beta, cusparseMatDescr descrB, int nnzB, Pointer csrSortedValB, Pointer csrSortedRowPtrB,
		Pointer csrSortedColIndB, cusparseMatDescr descrC, Pointer csrSortedValC, Pointer csrSortedRowPtrC,
		Pointer csrSortedColIndC, long[] pBufferSizeInBytes);

	public static int cusparseSparseToDense(cusparseHandle handle, cusparseConstSpMatDescr matA,
		cusparseDnMatDescr matB, int alg, Pointer externalBuffer) {
		return checkCusparseStatus(cusparseSparseToDenseNative(handle, matA, matB, alg, externalBuffer));
	}

	private static native int cusparseSparseToDenseNative(cusparseHandle handle, cusparseConstSpMatDescr matA,
		cusparseDnMatDescr matB, int alg, Pointer externalBuffer);

	public static int cusparseSparseToDense_bufferSize(cusparseHandle handle, cusparseConstSpMatDescr matA,
		cusparseDnMatDescr matB, int alg, long[] bufferSize) {
		return checkCusparseStatus(cusparseSparseToDense_bufferSizeNative(handle, matA, matB, alg, bufferSize));
	}

	private static native int cusparseSparseToDense_bufferSizeNative(cusparseHandle handle,
		cusparseConstSpMatDescr matA, cusparseDnMatDescr matB, int alg, long[] bufferSize);

	public static int cusparseDenseToSparse_bufferSize(cusparseHandle handle, cusparseConstDnMatDescr matA,
		cusparseSpMatDescr matB, int alg, long[] bufferSize) {
		return checkCusparseStatus(cusparseDenseToSparse_bufferSizeNative(handle, matA, matB, alg, bufferSize));
	}

	private static native int cusparseDenseToSparse_bufferSizeNative(cusparseHandle handle,
		cusparseConstDnMatDescr matA, cusparseSpMatDescr matB, int alg, long[] bufferSize);

	public static int cusparseDenseToSparse_analysis(cusparseHandle handle, cusparseConstDnMatDescr matA,
		cusparseSpMatDescr matB, int alg, Pointer externalBuffer) {
		return checkCusparseStatus(cusparseDenseToSparse_analysisNative(handle, matA, matB, alg, externalBuffer));
	}

	private static native int cusparseDenseToSparse_analysisNative(cusparseHandle handle, cusparseConstDnMatDescr matA,
		cusparseSpMatDescr matB, int alg, Pointer externalBuffer);

	public static int cusparseDenseToSparse_convert(cusparseHandle handle, cusparseConstDnMatDescr matA,
		cusparseSpMatDescr matB, int alg, Pointer externalBuffer) {
		return checkCusparseStatus(cusparseDenseToSparse_convertNative(handle, matA, matB, alg, externalBuffer));
	}

	private static native int cusparseDenseToSparse_convertNative(cusparseHandle handle, cusparseConstDnMatDescr matA,
		cusparseSpMatDescr matB, int alg, Pointer externalBuffer);

	public static int cusparseDnnz(cusparseHandle handle, int dirA, int m, int n, cusparseMatDescr descrA, Pointer A,
		int lda, Pointer nnzPerRowCol, Pointer nnzTotalDevHostPtr) {
		return checkCusparseStatus(
			cusparseDnnzNative(handle, dirA, m, n, descrA, A, lda, nnzPerRowCol, nnzTotalDevHostPtr));
	}

	private static native int cusparseDnnzNative(cusparseHandle handle, int dirA, int m, int n, cusparseMatDescr descrA,
		Pointer A, int lda, Pointer nnzPerRowCol, Pointer nnzTotalDevHostPtr);

}
