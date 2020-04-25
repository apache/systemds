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

package org.apache.sysds.runtime.compress.cocode;

import java.util.Arrays;

import org.apache.sysds.runtime.compress.cocode.PlanningCoCoder.GroupableColInfo;
import org.apache.sysds.runtime.compress.estim.CompressedSizeEstimator;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;

/**
 * Class to represent information about co-coding a group of columns.
 */
public class PlanningCoCodingGroup {
	private int[] _colIndexes;
	private PlanningCoCodingGroup _leftGrp;
	private PlanningCoCodingGroup _rightGrp;

	private long _estSize;
	private double _cardRatio;

	/**
	 * Constructor for a one-column group; i.e. do not co-code a given column.
	 * 
	 * @param col  column
	 * @param info groupable column info
	 */
	public PlanningCoCodingGroup(int col, GroupableColInfo info) {
		_colIndexes = new int[] {col};
		_estSize = info.size;
		_cardRatio = info.cardRatio;
	}

	/**
	 * Constructor for merging two disjoint groups of columns
	 * 
	 * @param grp1    first column group to merge
	 * @param grp2    second column group to merge
	 * @param estim   bitmap size estimator
	 * @param numRows number of rows
	 */
	public PlanningCoCodingGroup(PlanningCoCodingGroup grp1, PlanningCoCodingGroup grp2, CompressedSizeEstimator estim,
		int numRows) {
		_colIndexes = getMergedIndexes(grp1._colIndexes, grp2._colIndexes);

		// estimating size info
		CompressedSizeInfoColGroup groupSizeInfo = estim.estimateCompressedColGroupSize(_colIndexes);
		
		_estSize = groupSizeInfo.getMinSize();
		_cardRatio = groupSizeInfo.getEstCard() / numRows;

		_leftGrp = grp1;
		_rightGrp = grp2;
	}

	public int[] getColIndices() {
		return _colIndexes;
	}

	/**
	 * Obtain estimated compressed size of the grouped columns.
	 * 
	 * @return estimated compressed size of the grouped columns
	 */
	public long getEstSize() {
		return _estSize;
	}

	public double getChangeInSize() {
		if(_leftGrp == null || _rightGrp == null)
			return 0;

		return getEstSize() - _leftGrp.getEstSize() - _rightGrp.getEstSize();
	}

	public double getCardinalityRatio() {
		return _cardRatio;
	}

	public PlanningCoCodingGroup getLeftGroup() {
		return _leftGrp;
	}

	public PlanningCoCodingGroup getRightGroup() {
		return _rightGrp;
	}

	@Override
	public int hashCode() {
		return Arrays.hashCode(_colIndexes);
	}

	@Override
	public boolean equals(Object that) {
		if(!(that instanceof PlanningCoCodingGroup))
			return false;

		PlanningCoCodingGroup thatgrp = (PlanningCoCodingGroup) that;
		return Arrays.equals(_colIndexes, thatgrp._colIndexes);
	}

	@Override
	public String toString() {
		return Arrays.toString(_colIndexes);
	}

	public static int[] getMergedIndexes(int[] indexes1, int[] indexes2) {
		// merge sorted non-empty arrays
		int[] ret = new int[indexes1.length + indexes2.length];
		int grp1Ptr = 0, grp2Ptr = 0;
		for(int mergedIx = 0; mergedIx < ret.length; mergedIx++) {
			if(indexes1[grp1Ptr] < indexes2[grp2Ptr]) {
				ret[mergedIx] = indexes1[grp1Ptr++];
				if(grp1Ptr == indexes1.length) {
					System.arraycopy(indexes2, grp2Ptr, ret, mergedIx + 1, indexes2.length - grp2Ptr);
					break;
				}
			}
			else {
				ret[mergedIx] = indexes2[grp2Ptr++];
				if(grp2Ptr == indexes2.length) {
					System.arraycopy(indexes1, grp1Ptr, ret, mergedIx + 1, indexes1.length - grp1Ptr);
					break;
				}
			}
		}

		return ret;
	}

	public static class ColIndexes {
		final int[] _colIndexes;

		public ColIndexes(int[] colIndexes) {
			_colIndexes = colIndexes;
		}

		@Override
		public int hashCode() {
			return Arrays.hashCode(_colIndexes);
		}

		@Override
		public boolean equals(Object that) {
			if(!(that instanceof ColIndexes))
				return false;

			ColIndexes thatgrp = (ColIndexes) that;
			return Arrays.equals(_colIndexes, thatgrp._colIndexes);
		}
	}
}
