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

package org.apache.sysds.runtime.privacy.finegrained;

import java.util.Arrays;

/**
 * A DataRange instance marks a part of a CachableData data object.
 * The beginDims marks the beginning for all dimensions and
 * the endDims marks the end for all dimensions. 
 * DataRange is very similar to org.apache.sysds.runtime.util.IndexRange, 
 * except that DataRange supports more than two dimensions. 
 */
public class DataRange {

	private long[] _beginDims;
	private long[] _endDims;

	public DataRange(long[] beginDims, long[] endDims){
		_beginDims = beginDims;
		_endDims = endDims;
	}

	public long[] getBeginDims(){
		return _beginDims;
	}

	public long[] getEndDims(){
		return _endDims;
	}

	/**
	 * Returns true if this data range overlaps with the given data range. 
	 * An overlap means that the data ranges have some overlap in all dimension. 
	 * @param dataRange for which the overlap is checked
	 * @return true if the data ranges overlap or false if not
	 */
	public boolean overlaps(DataRange dataRange){
		long[] dataRangeBegin = dataRange.getBeginDims();
		long[] dataRangeEnd = dataRange.getEndDims();

		if (_beginDims.length != dataRangeBegin.length 
			|| _endDims.length != dataRangeEnd.length)
		{
			return false;
		}

		for ( int i = 0; i < _beginDims.length; i++ )
			if ( dimensionOutOfRange(dataRangeBegin, dataRangeEnd, i) )
				return false;

		return true;
	}

	/**
	 * Returns true if the given index is in the data range.
	 * Being in the data range means that the index has to be in the range for all dimensions.
	 * @param index of an element for which it is checked if it is in the range
	 * @return true if the index is in the range and false otherwise
	 */
	public boolean contains(long[] index){
		if ( _beginDims.length != index.length )
			return false;
		for ( int i = 0; i < _beginDims.length; i++ )
			if ( _beginDims[i] > index[i] || _endDims[i] < index[i] )
				return false;
		return true;
	}

	/**
	 * Returns true if the given DataRange is not overlapping in the given dimension
	 * @param dataRangeBegin begin dimensions
	 * @param dataRangeEnd end dimensions
	 * @param i dimension
	 * @return true if out of range
	 */
	private boolean dimensionOutOfRange(long[] dataRangeBegin, long[] dataRangeEnd, int i){
		return (_beginDims[i] < dataRangeBegin[i] && _endDims[i] < dataRangeBegin[i]) 
				|| (_beginDims[i] > dataRangeBegin[i] && _beginDims[i] > dataRangeEnd[i] );
	}

	@Override
	public String toString(){
		return Arrays.toString(new String[]{
			Arrays.toString(_beginDims), Arrays.toString(_endDims)});
	}

	@Override
	public boolean equals(Object other){
		if ( other instanceof DataRange ) {
			DataRange otherRange = (DataRange)other;
			return Arrays.equals(_beginDims,otherRange.getBeginDims())
				&& Arrays.equals(_endDims,otherRange.getEndDims());
		}
		return false;
	}
}
