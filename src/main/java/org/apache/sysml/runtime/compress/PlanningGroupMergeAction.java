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

import org.apache.sysml.runtime.compress.estim.CompressedSizeEstimator;

/**
 * Internal data structure for tracking potential merges of column groups in
 * co-coding calculations.
 * 
 */
class PlanningGroupMergeAction implements Comparable<PlanningGroupMergeAction> 
{
	private PlanningCoCodingGroup _leftGrp;   //left input
	private PlanningCoCodingGroup _rightGrp;  //right input
	private PlanningCoCodingGroup _mergedGrp; //output
	private long _changeInSize;

	
	public PlanningGroupMergeAction(CompressedSizeEstimator sizeEstimator,
			float numRowsWeight, PlanningCoCodingGroup leftGrp, PlanningCoCodingGroup rightGrp) {
		_leftGrp = leftGrp;
		_rightGrp = rightGrp;
		_mergedGrp = new PlanningCoCodingGroup(leftGrp, rightGrp, sizeEstimator, numRowsWeight);

		// Negative size change ==> Decrease in size
		_changeInSize = _mergedGrp.getEstSize() 
				- leftGrp.getEstSize() - rightGrp.getEstSize();
	}

	public int compareTo(PlanningGroupMergeAction o) {
		// We only sort by the change in size
		return (int) Math.signum(_changeInSize - o._changeInSize);
	}

	@Override
	public String toString() {
		return String.format("Merge %s and %s", _leftGrp, _rightGrp);
	}

	public PlanningCoCodingGroup getLeftGrp() {
		return _leftGrp;
	}

	public PlanningCoCodingGroup getRightGrp() {
		return _rightGrp;
	}

	public PlanningCoCodingGroup getMergedGrp() {
		return _mergedGrp;
	}

	public long getChangeInSize() {
		return _changeInSize;
	}
}
