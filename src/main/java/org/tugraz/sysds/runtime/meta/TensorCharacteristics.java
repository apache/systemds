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

import org.tugraz.sysds.runtime.util.UtilFunctions;


public class TensorCharacteristics implements Serializable
{
	private static final long serialVersionUID = 8300479822915546000L;

	private long[] _dims;
	private int[] _blkSizes;
	private long _nnz = -1;
	
	public TensorCharacteristics() {}
	
	public TensorCharacteristics(long[] dims, long nnz) {
		int[] blkSizes = new int[dims.length];
		Arrays.fill(blkSizes, -1);
		set(dims, blkSizes, nnz);
	}
	
	public TensorCharacteristics(long[] dims, int[] blkSizes) {
		set(dims, blkSizes, -1);
	}

	public TensorCharacteristics(long[] dims, int[] blkSizes, long nnz) {
		set(dims, blkSizes, nnz);
	}
	
	public TensorCharacteristics(TensorCharacteristics that) {
		set(that);
	}

	public TensorCharacteristics set(long[] dims, int[] blkSizes) {
		set(dims, blkSizes, -1);
		return this;
	}
	
	public TensorCharacteristics set(long[] dims, int[] blkSizes, long nnz) {
		_dims = dims;
		_blkSizes = blkSizes;
		return this;
	}
	
	public TensorCharacteristics set(TensorCharacteristics that) {
		set(that._dims, that._blkSizes, that._nnz);
		return this;
	}
	
	public int getNumDims() {
		return _dims.length;
	}
	
	public long getDim(int i) {
		return _dims[i];
	}
	
	public TensorCharacteristics setDim(int i, long dim) {
		_dims[i] = dim;
		return this;
	}
	
	public TensorCharacteristics setDims(long[] dims) {
		_dims = dims;
		return this;
	}
	
	public long getBlockSize(int i) {
		return _blkSizes[i];
	}
	
	public TensorCharacteristics setBlockSize(int i, int blksize) {
		_blkSizes[i] = blksize;
		return this;
	}
	
	public TensorCharacteristics setBlockSizes(int[] blkSizes) {
		_blkSizes = blkSizes;
		return this;
	}
	
	public long getLength() {
		return UtilFunctions.prod(_dims);
	}
	
	public long getNumBlocks() {
		long ret = 1;
		for( int i=0; i<getNumDims(); i++ )
			ret *= getNumBlocks(i);
		return ret;
	}
	
	public long getNumBlocks(int i) {
		return Math.max((long) Math.ceil((double)getDim(i) / getBlockSize(i)), 1);
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
