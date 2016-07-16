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

package org.apache.sysml.runtime.compress;

import java.util.Arrays;

import org.apache.sysml.runtime.compress.estim.CompressedSizeEstimator;
import org.apache.sysml.runtime.compress.estim.CompressedSizeInfo;

/** 
 * Class to represent information about co-coding a group of columns. 
 * 
 */
public class PlanningCoCodingGroup 
{
	private int[] _colIndexes;
	private long _estSize;
	private float _cardRatio;

	/**
	 * Constructor for a one-column group; i.e. do not co-code a given column.
	 * 
	 */
	public PlanningCoCodingGroup(int col, long estSize, float cardRatio) {
		_colIndexes = new int[]{col};
		_estSize = estSize;
		_cardRatio = cardRatio;
	}

	/**
	 * Constructor for merging two disjoint groups of columns
	 * 
	 * @param grp1   first group of columns to merge
	 * @param grp2   second group to merge
	 * @param numRowsWeight numRows x sparsity
	 */
	public PlanningCoCodingGroup(PlanningCoCodingGroup grp1, PlanningCoCodingGroup grp2,
			CompressedSizeEstimator bitmapSizeEstimator, float numRowsWeight) 
	{
		// merge sorted non-empty arrays
		_colIndexes = new int[grp1._colIndexes.length + grp2._colIndexes.length];		
		int grp1Ptr = 0, grp2Ptr = 0;
		for (int mergedIx = 0; mergedIx < _colIndexes.length; mergedIx++) {
			if (grp1._colIndexes[grp1Ptr] < grp2._colIndexes[grp2Ptr]) {
				_colIndexes[mergedIx] = grp1._colIndexes[grp1Ptr++];
				if (grp1Ptr == grp1._colIndexes.length) {
					System.arraycopy(grp2._colIndexes, grp2Ptr, _colIndexes,
							mergedIx + 1, grp2._colIndexes.length - grp2Ptr);
					break;
				}
			} else {
				_colIndexes[mergedIx] = grp2._colIndexes[grp2Ptr++];
				if (grp2Ptr == grp2._colIndexes.length) {
					System.arraycopy(grp1._colIndexes, grp1Ptr, _colIndexes,
							mergedIx + 1, grp1._colIndexes.length - grp1Ptr);
					break;
				}
			}
		}
		
		// estimating size info
		CompressedSizeInfo groupSizeInfo = bitmapSizeEstimator
				.estimateCompressedColGroupSize(_colIndexes);
		_estSize = groupSizeInfo.getMinSize();
		_cardRatio = groupSizeInfo.getEstCarinality() / numRowsWeight;
	}

	public int[] getColIndices() {
		return _colIndexes;
	}

	/**
	 * @return estimated compressed size of the grouped columns
	 */
	public long getEstSize() {
		return _estSize;
	}

	public float getCardinalityRatio() {
		return _cardRatio;
	}

	@Override
	public String toString() {
		return Arrays.toString(_colIndexes);
	}
}