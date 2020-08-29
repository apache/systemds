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

package org.apache.sysds.runtime.controlprogram.federated;

import java.util.Arrays;

import org.apache.sysds.runtime.util.IndexRange;

public class FederatedRange implements Comparable<FederatedRange> {
	private long[] _beginDims;
	private long[] _endDims;
	
	/**
	 * Create a range with the indexes of each dimension between their respective <code>beginDims</code> and
	 * <code>endDims</code> values.
	 * @param beginDims the beginning indexes for each dimension
	 * @param endDims the ending indexes for each dimension
	 */
	public FederatedRange(long[] beginDims, long[] endDims) {
		_beginDims = beginDims;
		_endDims = endDims;
	}
	
	/**
	 * Does a deep copy of another <code>FederatedRange</code> object.
	 * @param other the <code>FederatedRange</code> to copy
	 */
	public FederatedRange(FederatedRange other) {
		this(other._beginDims.clone(), other._endDims.clone());
	}
	
	public FederatedRange(FederatedRange other, long clen) {
		this(other._beginDims.clone(), other._endDims.clone());
		_endDims[1] = clen;
	}
	
	public void setBeginDim(int dim, long value) {
		_beginDims[dim] = value;
	}
	
	public void setEndDim(int dim, long value) {
		_endDims[dim] = value;
	}
	
	public long[] getBeginDims() {
		return _beginDims;
	}
	
	public long[] getEndDims() {
		return _endDims;
	}
	
	public int[] getBeginDimsInt() {
		return Arrays.stream(_beginDims).mapToInt(i -> (int) i).toArray();
	}
	
	public int[] getEndDimsInt() {
		return Arrays.stream(_endDims).mapToInt(i -> (int) i).toArray();
	}
	
	public long getSize() {
		long size = 1;
		for (int i = 0; i < _beginDims.length; i++)
			size *= getSize(i);
		return size;
	}
	
	public long getSize(int dim) {
		return _endDims[dim] - _beginDims[dim];
	}
	
	@Override
	public int compareTo(FederatedRange o) {
		for (int i = 0; i < _beginDims.length; i++) {
			if ( _beginDims[i] < o._beginDims[i])
				return -1;
			if ( _beginDims[i] > o._beginDims[i])
				return 1;
		}
		return 0;
	}
	
	@Override
	public String toString() {
		return Arrays.toString(_beginDims) + " - " + Arrays.toString(_endDims);
	}

	public FederatedRange shift(long rshift, long cshift) {
		//row shift
		_beginDims[0] += rshift;
		_endDims[0] += rshift;
		//column shift
		_beginDims[1] += cshift;
		_endDims[1] += cshift;
		return this;
	}
	
	public FederatedRange transpose() {
		long tmpBeg = _beginDims[0];
		long tmpEnd = _endDims[0];
		_beginDims[0] = _beginDims[1];
		_endDims[0] = _endDims[1];
		_beginDims[1] = tmpBeg;
		_endDims[1] = tmpEnd;
		return this;
	}

	public IndexRange asIndexRange() {
		return new IndexRange(_beginDims[0], _endDims[0], _beginDims[1], _endDims[1]);
	}
}
