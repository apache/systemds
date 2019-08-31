/*
 * Copyright 2019 Graz University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package org.tugraz.sysds.runtime.meta;

import org.tugraz.sysds.runtime.util.UtilFunctions;

import java.util.Arrays;


public class TensorCharacteristics extends DataCharacteristics
{
	private static final long serialVersionUID = 8300479822915546000L;

	public static final int[] DEFAULT_BLOCK_SIZE = {1024, 128, 32, 16, 8, 8};
	private long[] _dims;
	private int[] _blkSizes;
	private long _nnz = -1;
	
	public TensorCharacteristics() {}
	
	public TensorCharacteristics(long[] dims, long nnz) {
		int[] blkSizes = new int[dims.length];
		Arrays.fill(blkSizes, DEFAULT_BLOCK_SIZE[dims.length - 2]);
		set(dims, blkSizes, nnz);
	}
	
	public TensorCharacteristics(long[] dims, int[] blkSizes) {
		set(dims, blkSizes, -1);
	}

	public TensorCharacteristics(long[] dims, int[] blkSizes, long nnz) {
		set(dims, blkSizes, nnz);
	}

	public TensorCharacteristics(DataCharacteristics that) {
		set(that);
	}

	@Override
	public DataCharacteristics set(long[] dims, int[] blkSizes) {
		set(dims, blkSizes, -1);
		return this;
	}

	@Override
	public DataCharacteristics set(long[] dims, int[] blkSizes, long nnz) {
		_dims = dims;
		_blkSizes = blkSizes;
		return this;
	}

	@Override
	public DataCharacteristics set(DataCharacteristics that) {
		long[] dims = that.getDims().clone();
		int[] blockSizes = that.getBlockSizes().clone();
		set(dims, blockSizes, that.getNonZeros());
		return this;
	}

	@Override
	public void setNonZeros(long nnz) {
		_nnz = nnz;
	}

	@Override
	public boolean dimsKnown() {
		for (long dim : _dims) {
			if (dim < 0)
				return false;
		}
		return true;
	}

	@Override
	public boolean dimsKnown(boolean includeNnz) {
		return dimsKnown() && (!includeNnz || nnzKnown());
	}

	@Override
	public boolean nnzKnown() {
		return _nnz >= 0;
	}

	@Override
	public int getNumDims() {
		return _dims.length;
	}

	@Override
	public long getDim(int i) {
		return _dims[i];
	}

	@Override
	public long[] getDims() {
		return _dims;
	}

	@Override
	public TensorCharacteristics setDim(int i, long dim) {
		_dims[i] = dim;
		return this;
	}

	@Override
	public TensorCharacteristics setDims(long[] dims) {
		_dims = dims;
		return this;
	}

	@Override
	public int getBlockSize(int i) {
		return _blkSizes[i];
	}

	@Override
	public int[] getBlockSizes() {
		return _blkSizes;
	}

	@Override
	public DataCharacteristics setBlockSize(int i, int blksize) {
		_blkSizes[i] = blksize;
		return this;
	}

	@Override
	public DataCharacteristics setBlockSizes(int[] blkSizes) {
		_blkSizes = blkSizes;
		return this;
	}

	@Override
	public long getLength() {
		return UtilFunctions.prod(_dims);
	}

	@Override
	public long getNumBlocks() {
		long ret = 1;
		for( int i=0; i<getNumDims(); i++ )
			ret *= getNumBlocks(i);
		return ret;
	}

	@Override
	public long getNumBlocks(int i) {
		return Math.max((long) Math.ceil((double)getDim(i) / getBlockSize(i)), 1);
	}

	@Override
	public long getNonZeros() {
		return _nnz;
	}

	@Override
	public String toString() {
		return "["+Arrays.toString(_dims)+", nnz="+_nnz
			+ ", blocks "+Arrays.toString(_blkSizes)+"]";
	}
	
	@Override
	public boolean equals (Object anObject) {
		if( !(anObject instanceof TensorCharacteristics) )
			return false;
		TensorCharacteristics tc = (TensorCharacteristics) anObject;
		return Arrays.equals(_dims, tc._dims)
			&& Arrays.equals(_blkSizes, tc._blkSizes)
			&& _nnz == tc._nnz;
	}
	
	@Override
	public int hashCode() {
		return UtilFunctions.intHashCode(UtilFunctions.intHashCode(
			Arrays.hashCode(_dims), Arrays.hashCode(_blkSizes)),
			Long.hashCode(_nnz));
	}
}
