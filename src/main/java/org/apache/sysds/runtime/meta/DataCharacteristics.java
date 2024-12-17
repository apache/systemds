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

import java.io.Serializable;

public abstract class DataCharacteristics implements Serializable {
	private static final long serialVersionUID = 3411056029517599342L;

	protected int _blocksize;                 // squared block size
	protected boolean _noEmptyBlocks = false; // does not materialize empty blocks
	
	public abstract DataCharacteristics set(long nr, long nc, int blen);

	public abstract DataCharacteristics set(long nr, long nc, int blen, long nnz);

	public abstract DataCharacteristics set(long[] dims, int blocksize);

	public abstract DataCharacteristics set(long[] dims, int blocksize, long nnz);

	public abstract DataCharacteristics set(DataCharacteristics that);

	public abstract long getRows();

	public abstract DataCharacteristics setRows(long rlen);

	public abstract long getCols();

	public abstract DataCharacteristics setCols(long clen);

	public abstract long getLength();

	public int getBlocksize() {
		return _blocksize;
	}

	public DataCharacteristics setBlocksize(int blen){
		_blocksize = blen;
		return this;
	}
	

	public DataCharacteristics setNoEmptyBlocks(boolean flag) {
		_noEmptyBlocks = flag;
		return this;
	}
	
	public boolean isNoEmptyBlocks() {
		return _noEmptyBlocks;
	}

	public long getNumBlocks() {
		return getNumRowBlocks() * getNumColBlocks();
	}

	public abstract long getNumRowBlocks();

	public abstract long getNumColBlocks();

	public abstract DataCharacteristics setDimension(long nr, long nc);

	public abstract int getNumDims();

	public abstract long getDim(int i);

	public long[] getDims() {
		return getLongDims();
	}

	public abstract long[] getLongDims();

	public abstract int[] getIntDims();

	public abstract DataCharacteristics setDim(int i, long dim);

	public abstract DataCharacteristics setDims(long[] dims);

	public abstract long getNumBlocks(int i);

	public abstract DataCharacteristics setNonZeros(long nnz);

	public abstract long getNonZeros();

	public abstract DataCharacteristics setNonZerosBound(long nnz);

	public abstract long getNonZerosBound();

	public abstract double getSparsity();

	public abstract boolean dimsKnown();

	public abstract boolean dimsKnown(boolean includeNnz);

	public abstract boolean rowsKnown();

	public abstract boolean colsKnown();

	public abstract boolean nnzKnown();

	public abstract boolean isUltraSparse();

	public abstract boolean mightHaveEmptyBlocks();

	public abstract boolean equalDims(Object anObject);

	@Override
	public abstract boolean equals(Object anObject);
	
	@Override
	public abstract int hashCode();
}
