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


package org.tugraz.sysds.runtime.meta;

import java.io.Serializable;
import java.util.Arrays;

import org.tugraz.sysds.hops.OptimizerUtils;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.operators.AggregateBinaryOperator;
import org.tugraz.sysds.runtime.matrix.operators.AggregateUnaryOperator;
import org.tugraz.sysds.runtime.matrix.operators.ReorgOperator;


public class MatrixCharacteristics implements Serializable
{
	private static final long serialVersionUID = 8300479822915546000L;

	private long numRows = -1;
	private long numColumns = -1;
	private int numRowsPerBlock = 1;
	private int numColumnsPerBlock = 1;
	private long nonZero = -1;
	private boolean ubNnz = false;
	
	public MatrixCharacteristics() {}
	
	public MatrixCharacteristics(long nr, long nc, long nnz) {
		set(nr, nc, -1, -1, nnz);
	}
	
	public MatrixCharacteristics(long nr, long nc, int bnr, int bnc) {
		set(nr, nc, bnr, bnc);
	}

	public MatrixCharacteristics(long nr, long nc, int bnr, int bnc, long nnz) {
		set(nr, nc, bnr, bnc, nnz);
	}
	
	public MatrixCharacteristics(MatrixCharacteristics that) {
		set(that);
	}

	public MatrixCharacteristics set(long nr, long nc, int bnr, int bnc) {
		numRows = nr;
		numColumns = nc;
		numRowsPerBlock = bnr;
		numColumnsPerBlock = bnc;
		return this;
	}
	
	public MatrixCharacteristics set(long nr, long nc, int bnr, int bnc, long nnz) {
		set(nr, nc, bnr, bnc);
		nonZero = nnz;
		ubNnz = false;
		return this;
	}
	
	public MatrixCharacteristics set(MatrixCharacteristics that) {
		set(that.numRows, that.numColumns, that.numRowsPerBlock,
			that.numColumnsPerBlock, that.nonZero);
		ubNnz = that.ubNnz;
		return this;
	}
	
	public long getRows(){
		return numRows;
	}
	
	public void setRows(long rlen) {
		numRows = rlen;
	}

	public long getCols(){
		return numColumns;
	}
	
	public void setCols(long clen) {
		numColumns = clen;
	}
	
	public long getLength() {
		return numRows * numColumns;
	}
	
	public int getRowsPerBlock() {
		return numRowsPerBlock;
	}
	
	public void setRowsPerBlock( int brlen){
		numRowsPerBlock = brlen;
	} 
	
	public int getColsPerBlock() {
		return numColumnsPerBlock;
	}
	
	public void setColsPerBlock( int bclen){
		numColumnsPerBlock = bclen;
	} 
	
	public long getNumBlocks() {
		return getNumRowBlocks() * getNumColBlocks();
	}
	
	public long getNumRowBlocks() {
		//number of row blocks w/ awareness of zero rows
		return Math.max((long) Math.ceil((double)getRows() / getRowsPerBlock()), 1);
	}
	
	public long getNumColBlocks() {
		//number of column blocks w/ awareness of zero columns
		return Math.max((long) Math.ceil((double)getCols() / getColsPerBlock()), 1);
	}
	
	@Override
	public String toString() {
		return "["+numRows+" x "+numColumns+", nnz="+nonZero+" ("+ubNnz+")"
		+", blocks ("+numRowsPerBlock+" x "+numColumnsPerBlock+")]";
	}
	
	public void setDimension(long nr, long nc) {
		numRows = nr;
		numColumns = nc;
	}
	
	public MatrixCharacteristics setBlockSize(int blen) {
		return setBlockSize(blen, blen);
	}
	
	public MatrixCharacteristics setBlockSize(int bnr, int bnc) {
		numRowsPerBlock = bnr;
		numColumnsPerBlock = bnc;
		return this;
	}
	
	public void setNonZeros(long nnz) {
		ubNnz = false;
		nonZero = nnz;
	}
	
	public long getNonZeros() {
		return !ubNnz ? nonZero : -1;
	}
	
	public void setNonZerosBound(long nnz) {
		ubNnz = true;
		nonZero = nnz;
	}
	
	public long getNonZerosBound() {
		return nonZero;
	}
	
	public double getSparsity() {
		return OptimizerUtils.getSparsity(this);
	}
	
	public boolean dimsKnown() {
		return ( numRows >= 0 && numColumns >= 0 );
	}
	
	public boolean dimsKnown(boolean includeNnz) {
		return ( numRows >= 0 && numColumns >= 0
			&& (!includeNnz || nnzKnown()));
	}
	
	public boolean rowsKnown() {
		return ( numRows >= 0 );
	}

	public boolean colsKnown() {
		return ( numColumns >= 0 );
	}
	
	public boolean nnzKnown() {
		return ( !ubNnz && nonZero >= 0 );
	}
	
	public boolean isUltraSparse() {
		return dimsKnown(true) && OptimizerUtils.getSparsity(this)
			< MatrixBlock.ULTRA_SPARSITY_TURN_POINT;
	}
	
	public boolean mightHaveEmptyBlocks() {
		long singleBlk = Math.max(Math.min(numRows, numRowsPerBlock),1) 
				* Math.max(Math.min(numColumns, numColumnsPerBlock),1);
		return !nnzKnown() || numRows==0 || numColumns==0
			|| (nonZero < numRows*numColumns - singleBlk);
	}
	
	public static void reorg(MatrixCharacteristics dim, ReorgOperator op, MatrixCharacteristics dimOut) {
		op.fn.computeDimension(dim, dimOut);
	}
	
	public static void aggregateUnary(MatrixCharacteristics dim, AggregateUnaryOperator op, MatrixCharacteristics dimOut) {
		op.indexFn.computeDimension(dim, dimOut);
	}
	
	public static void aggregateBinary(MatrixCharacteristics dim1, MatrixCharacteristics dim2,
			AggregateBinaryOperator op, MatrixCharacteristics dimOut) 
	{
		//set dimension
		dimOut.set(dim1.numRows, dim2.numColumns, dim1.numRowsPerBlock, dim2.numColumnsPerBlock);
	}
	
	@Override
	public boolean equals (Object anObject) {
		if( !(anObject instanceof MatrixCharacteristics) )
			return false;
		MatrixCharacteristics mc = (MatrixCharacteristics) anObject;
		return ((numRows == mc.numRows)
			&& (numColumns == mc.numColumns)
			&& (numRowsPerBlock == mc.numRowsPerBlock)
			&& (numColumnsPerBlock == mc.numColumnsPerBlock)
			&& (nonZero == mc.nonZero));
	}
	
	@Override
	public int hashCode() {
		return Arrays.hashCode(new long[]{numRows,numColumns,
			numRowsPerBlock,numColumnsPerBlock,nonZero});
	}
}
