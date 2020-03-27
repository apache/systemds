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

import static jcuda.runtime.JCuda.cudaMemset;

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.instructions.gpu.context.CSRPointer;
import org.apache.sysds.runtime.instructions.gpu.context.GPUContext;

import jcuda.Pointer;

/**
 * Performs a slice operation: out = in[(n+1):(n+1), 1:numColumns]
 */
public class LibMatrixCuDNNInputRowFetcher extends LibMatrixCUDA implements java.lang.AutoCloseable {
	GPUContext gCtx; String instName; int numColumns; boolean isInputInSparseFormat; 
	Object inPointer; // can be either CSRPointer or Pointer 
	Pointer outPointer;

	/**
	 * Initialize the input fetcher
	 * 
	 * @param gCtx current gpu context
	 * @param instName name of the instruction
	 * @param image input matrix object.
	 */
	public LibMatrixCuDNNInputRowFetcher(GPUContext gCtx, String instName, MatrixObject image) {
		this.gCtx = gCtx; this.instName = instName;
		numColumns = LibMatrixCUDA.toInt(image.getNumColumns());
		isInputInSparseFormat = LibMatrixCUDA.isInSparseFormat(gCtx, image);
		inPointer = isInputInSparseFormat ? LibMatrixCUDA.getSparsePointer(gCtx, image, instName) : LibMatrixCuDNN.getDensePointerForCuDNN(gCtx, image, instName);
		outPointer = gCtx.allocate(instName, numColumns*sizeOfDataType);
	}
	/**
	 * Copy the nth row and return the dense pointer
	 * @param n zero-based row index
	 * @return dense pointer containing the nth row. This row is reused in the next iteration
	 */
	public Pointer getNthRow(int n) {
		if(isInputInSparseFormat) {
			jcuda.runtime.JCuda.cudaDeviceSynchronize();
			cudaMemset(outPointer, 0, numColumns*sizeOfDataType);
			jcuda.runtime.JCuda.cudaDeviceSynchronize();
			LibMatrixCUDA.sliceSparseDense(gCtx, instName, (CSRPointer)inPointer, outPointer, n, n, 0, LibMatrixCUDA.toInt(numColumns-1), numColumns);
		}
		else {
			LibMatrixCUDA.sliceDenseDense(gCtx, instName, (Pointer)inPointer, outPointer, n, n, 0, LibMatrixCUDA.toInt(numColumns-1), numColumns);
		}
		return outPointer;
	}
	/**
	 * Deallocates temporary pointer
	 */
	@Override
	public void close() {
		try {
			gCtx.cudaFreeHelper(null, outPointer, true);
		} catch (DMLRuntimeException e) {
			throw new RuntimeException(e);
		}
	}
}
