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

package org.apache.sysds.runtime.data;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

public abstract class SparseBlockFactory{
		protected static final Log LOG = LogFactory.getLog(SparseBlockFactory.class.getName());


	public static SparseBlock createSparseBlock(int rlen) {
		return createSparseBlock(MatrixBlock.DEFAULT_SPARSEBLOCK, rlen);
	}

	public static SparseBlock createSparseBlock(SparseBlock.Type type, int rlen) {
		switch( type ) {
			case MCSR: return new SparseBlockMCSR(rlen, -1);
			case CSR: return new SparseBlockCSR(rlen);
			case COO: return new SparseBlockCOO(rlen);
			case DCSR: return new SparseBlockDCSR(rlen);
			case MCSC: return new SparseBlockMCSC(rlen);
			case CSC: return new SparseBlockCSC(rlen, 0);
			default:
				throw new RuntimeException("Unexpected sparse block type: "+type.toString());
		}
	}
	
	public static SparseBlock createSparseBlock(SparseBlock.Type type, SparseRow row) {
		SparseBlock ret = createSparseBlock(type, 1);
		ret.set(0, row, true);
		return ret;
	}

	public static SparseBlock copySparseBlock(SparseBlock.Type type, SparseBlock sblock, boolean forceCopy) {
		//Call this method in case 'type' is row format
		return copySparseBlock(type, sblock, forceCopy, 1000); // Default clen value
	}

	public static SparseBlock copySparseBlock( SparseBlock.Type type, SparseBlock sblock, boolean forceCopy , int clen)
	{
		//sanity check for empty inputs
		if( sblock == null )
			return null;
		
		//check for existing target type
		if( !forceCopy && isSparseBlockType(sblock, type) ){
			return sblock;
		}
		
		//create target sparse block
		switch( type ) {
			case MCSR: return new SparseBlockMCSR(sblock);
			case CSR: return new SparseBlockCSR(sblock);
			case COO: return new SparseBlockCOO(sblock);
			case DCSR: return new SparseBlockDCSR(sblock);
			case MCSC: return new SparseBlockMCSC(sblock, clen);
			case CSC: return new SparseBlockCSC(sblock, clen);
			default:
				throw new RuntimeException("Unexpected sparse block type: "+type.toString());
		}
	}
	
	public static boolean isSparseBlockType(SparseBlock sblock, SparseBlock.Type type) {
		return (getSparseBlockType(sblock) == type);
	}
	
	public static SparseBlock.Type getSparseBlockType(SparseBlock sblock) {
		return (sblock instanceof SparseBlockMCSR) ? SparseBlock.Type.MCSR :
			(sblock instanceof SparseBlockCSR) ? SparseBlock.Type.CSR : 
			(sblock instanceof SparseBlockCOO) ? SparseBlock.Type.COO :
			(sblock instanceof SparseBlockDCSR) ? SparseBlock.Type.DCSR :
			(sblock instanceof SparseBlockMCSC) ? SparseBlock.Type.MCSC :
			(sblock instanceof SparseBlockCSC) ? SparseBlock.Type.CSC : null;
	}

	public static long estimateSizeSparseInMemory(SparseBlock.Type type, long nrows, long ncols, double sparsity) {
		switch( type ) {
			case MCSR: return SparseBlockMCSR.estimateSizeInMemory(nrows, ncols, sparsity);
			case CSR: return SparseBlockCSR.estimateSizeInMemory(nrows, ncols, sparsity);
			case CSC: return SparseBlockCSC.estimateSizeInMemory(nrows, ncols, sparsity);
			case COO: return SparseBlockCOO.estimateSizeInMemory(nrows, ncols, sparsity);
			case DCSR: return SparseBlockDCSR.estimateSizeInMemory(nrows, ncols, sparsity);
			case MCSC: return SparseBlockMCSC.estimateSizeInMemory(nrows, ncols, sparsity);
			default:
				throw new RuntimeException("Unexpected sparse block type: "+type.toString());
		}
	}

	public static SparseBlock createIdentityMatrix(int nRowCol){
		final int[] rowPtr = new int[nRowCol+1];
		final int[] colIdx = new int[nRowCol];
		final double[] vals = new double[nRowCol];
		int nnz = nRowCol;
		
		for(int i = 0; i < nRowCol; i++){
			rowPtr[i] = i;
			colIdx[i] = i;
			vals[i] = 1;
		}
		rowPtr[nRowCol] = nRowCol; // add last index for row pointers.
		
		return new SparseBlockCSR(rowPtr, colIdx, vals, nnz);
	}

	public static SparseBlock createIdentityMatrixWithEmptyRow(int nRowCol){
		final int[] rowPtr = new int[nRowCol+2];
		final int[] colIdx = new int[nRowCol];
		final double[] vals = new double[nRowCol];
		int nnz = nRowCol;
		
		for(int i = 0; i < nRowCol; i++){
			rowPtr[i] = i;
			colIdx[i] = i;
			vals[i] = 1;
		}
		// add last index for row pointers.
		rowPtr[nRowCol] = nRowCol; 
		rowPtr[nRowCol+1] = nRowCol;
		return new SparseBlockCSR(rowPtr, colIdx, vals, nnz);
	}

	/**
	 * Create a sparse block from an array. Note that the nnz count should be absolutely correct for this call to work.
	 * 
	 * @param valsDense a double array of values linearized.
	 * @param nCol The number of columns in reach row.
	 * @param nnz  The number of non zero values.
	 * @return A sparse block.
	 */
	public static SparseBlock createFromArray(final double[] valsDense, final  int nCol, final int nnz) {
		final int nRow = valsDense.length / nCol;
		if(nnz > 0) {

			final int[] rowPtr = new int[nRow + 1];
			final int[] colIdx = new int[nnz];
			final double[] valsSparse = new double[nnz];
			int off = 0;
			for(int i = 0; i < valsDense.length; i++) {
				final int mod = i % nCol;
				if(mod == 0)
					rowPtr[i / nCol] = off;
				if(valsDense[i] != 0) {
					valsSparse[off] = valsDense[i];
					colIdx[off] = mod;
					off++;
				}
			}
			rowPtr[rowPtr.length -1] = off;

			return new SparseBlockCSR(rowPtr, colIdx, valsSparse, nnz);
		}
		else {
			return new SparseBlockMCSR(nRow); // empty MCSR block
		}
	}
}
