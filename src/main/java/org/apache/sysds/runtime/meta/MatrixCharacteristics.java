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

package org.apache.sysds.runtime.meta;

import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import java.util.Arrays;


public class MatrixCharacteristics extends DataCharacteristics
{
	private static final long serialVersionUID = 8300479822915546000L;

	/** Number of columns */
	private long numRows = -1;
	/** Number of rows */
	private long numColumns = -1;
	/** Number of non zero values if -1 then unknown */
	private long nonZero = -1;
	/** Upper bound non zero value, indicate if the non zero value is an upper bound */
	private boolean ubNnz = false;
	
	public MatrixCharacteristics() {}
	
	public MatrixCharacteristics(long nr, long nc) {
		set(nr, nc, ConfigurationManager.getBlocksize(), -1);
	}
	
	public MatrixCharacteristics(long nr, long nc, long nnz) {
		set(nr, nc, ConfigurationManager.getBlocksize(), nnz);
	}
	
	public MatrixCharacteristics(long nr, long nc, int blen) {
		set(nr, nc, blen, -1);
	}

	public MatrixCharacteristics(long nr, long nc, int blen, long nnz) {
		set(nr, nc, blen, nnz);
	}
	
	public MatrixCharacteristics(DataCharacteristics that) {
		set(that);
	}

	@Override
	public DataCharacteristics set(long nr, long nc, int blen) {
		numRows = nr;
		numColumns = nc;
		_blocksize = blen;
		return this;
	}

	@Override
	public DataCharacteristics set(long nr, long nc, int blen, long nnz) {
		set(nr, nc, blen);
		nonZero = nnz;
		ubNnz = false;
		return this;
	}

	@Override
	public DataCharacteristics set(DataCharacteristics that) {
		set(that.getRows(), that.getCols(), that.getBlocksize(), that.getNonZeros());
		ubNnz = (that instanceof MatrixCharacteristics && ((MatrixCharacteristics)that).ubNnz);
		return this;
	}

	@Override
	public long getRows(){
		return numRows;
	}

	@Override
	public DataCharacteristics setRows(long rlen) {
		numRows = rlen;
		return this;
	}

	@Override
	public long getCols(){
		return numColumns;
	}

	@Override
	public DataCharacteristics setCols(long clen) {
		numColumns = clen;
		return this;
	}

	@Override
	public long getLength() {
		return numRows * numColumns;
	}

	@Override
	public long getNumBlocks() {
		return getNumRowBlocks() * getNumColBlocks();
	}

	@Override
	public long getNumRowBlocks() {
		//number of row blocks w/ awareness of zero rows
		return Math.max((long) Math.ceil((double)getRows() / getBlocksize()), 1);
	}

	@Override
	public long getNumColBlocks() {
		//number of column blocks w/ awareness of zero columns
		return Math.max((long) Math.ceil((double)getCols() / getBlocksize()), 1);
	}
	
	@Override
	public DataCharacteristics setDimension(long nr, long nc) {
		numRows = nr;
		numColumns = nc;
		return this;
	}
	
	@Override
	public long getDim(int i) {
		if (i == 0)
			return numRows;
		else if (i == 1)
			return numColumns;
		throw new DMLRuntimeException("Matrices have only 2 dimensions");
	}
	
	@Override
	public long[] getLongDims() {
		return new long[]{numRows, numColumns};
	}
	
	@Override
	public int[] getIntDims() {
		return new int[]{(int) numRows, (int) numColumns};
	}
	
	@Override
	public int getNumDims() {
		return 2;
	}
	
	@Override
	public DataCharacteristics setNonZeros(long nnz) {
		ubNnz = false;
		nonZero = nnz;
		return this;
	}

	@Override
	public long getNonZeros() {
		return !ubNnz ? nonZero : -1;
	}

	@Override
	public DataCharacteristics setNonZerosBound(long nnz) {
		ubNnz = true;
		nonZero = nnz;
		return this;
	}

	@Override
	public long getNonZerosBound() {
		return nonZero;
	}

	@Override
	public double getSparsity() {
		return OptimizerUtils.getSparsity(this);
	}

	@Override
	public boolean dimsKnown() {
		return ( numRows >= 0 && numColumns >= 0 );
	}

	@Override
	public boolean dimsKnown(boolean includeNnz) {
		return ( numRows >= 0 && numColumns >= 0
			&& (!includeNnz || nnzKnown()));
	}

	@Override
	public boolean rowsKnown() {
		return ( numRows >= 0 );
	}

	@Override
	public boolean colsKnown() {
		return ( numColumns >= 0 );
	}

	@Override
	public boolean nnzKnown() {
		return ( !ubNnz && nonZero >= 0 );
	}

	@Override
	public boolean isUltraSparse() {
		return dimsKnown(true) && OptimizerUtils.getSparsity(this)
			< MatrixBlock.ULTRA_SPARSITY_TURN_POINT;
	}

	@Override
	public boolean mightHaveEmptyBlocks() {
		long singleBlk = Math.max(Math.min(numRows, _blocksize),1)
				* Math.max(Math.min(numColumns, _blocksize),1);
		return !nnzKnown() || numRows==0 || numColumns==0
			|| (nonZero < numRows*numColumns - singleBlk);
	}
	
	@Override
	public boolean equalDims(Object anObject) {
		if( !(anObject instanceof MatrixCharacteristics) )
			return false;
		MatrixCharacteristics mc = (MatrixCharacteristics) anObject;
		return dimsKnown() && mc.dimsKnown()
			&& numRows == mc.numRows
			&& numColumns == mc.numColumns;
	}
	
	@Override
	public boolean equals (Object anObject) {
		if( !(anObject instanceof MatrixCharacteristics) )
			return false;
		MatrixCharacteristics mc = (MatrixCharacteristics) anObject;
		return ((numRows == mc.numRows)
			&& (numColumns == mc.numColumns)
			&& (_blocksize == mc._blocksize)
			&& (nonZero == mc.nonZero));
	}
	
	@Override
	public int hashCode() {
		return Arrays.hashCode(new long[]{
			numRows, numColumns, _blocksize, nonZero});
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder(30);
		sb.append("[");
		sb.append(numRows);
		sb.append(" x ");
		sb.append(numColumns);
		sb.append(", nnz=");
		sb.append(nonZero);
		sb.append(" (");
		sb.append(ubNnz);
		sb.append("), blocks (");
		sb.append(_blocksize);
		sb.append(" x ");
		sb.append(_blocksize);
		sb.append(")]");
		return sb.toString();
	}
}
