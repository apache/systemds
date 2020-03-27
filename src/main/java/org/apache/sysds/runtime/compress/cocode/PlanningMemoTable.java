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
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map.Entry;

import org.apache.sysds.runtime.compress.cocode.PlanningCoCodingGroup.ColIndexes;
import org.apache.sysds.runtime.compress.estim.CompressedSizeEstimator;

public class PlanningMemoTable {
	private HashMap<ColIndexes, PlanningCoCodingGroup> _memo = new HashMap<>();
	private double _optChangeInSize = 0;
	private int[] _stats = new int[3];

	public PlanningCoCodingGroup getOrCreate(PlanningCoCodingGroup c1, PlanningCoCodingGroup c2,
		CompressedSizeEstimator estim, int numRows) {
		ColIndexes c1c2Indexes = new ColIndexes(
			PlanningCoCodingGroup.getMergedIndexes(c1.getColIndices(), c2.getColIndices()));

		// probe memo table for existing column group (avoid extraction)
		PlanningCoCodingGroup c1c2 = _memo.get(c1c2Indexes);

		// create non-existing group and maintain global stats
		incrStats(0, 1, 0); // probed plans
		if(c1c2 == null) {
			c1c2 = new PlanningCoCodingGroup(c1, c2, estim, numRows);
			_memo.put(c1c2Indexes, c1c2);
			_optChangeInSize = Math.min(_optChangeInSize, c1c2.getChangeInSize());
			incrStats(0, 0, 1); // created plans
		}

		return c1c2;
	}

	public void remove(PlanningCoCodingGroup grp) {
		// remove atomic groups
		_memo.remove(new ColIndexes(grp.getColIndices()));
		_memo.remove(new ColIndexes(grp.getLeftGroup().getColIndices()));
		_memo.remove(new ColIndexes(grp.getRightGroup().getColIndices()));

		_optChangeInSize = 0;

		// remove overlapping groups and recompute min size
		Iterator<Entry<ColIndexes, PlanningCoCodingGroup>> iter = _memo.entrySet().iterator();
		while(iter.hasNext()) {
			PlanningCoCodingGroup tmp = iter.next().getValue();
			if(Arrays.equals(tmp.getLeftGroup().getColIndices(), grp.getLeftGroup().getColIndices()) ||
				Arrays.equals(tmp.getLeftGroup().getColIndices(), grp.getRightGroup().getColIndices()) ||
				Arrays.equals(tmp.getRightGroup().getColIndices(), grp.getLeftGroup().getColIndices()) ||
				Arrays.equals(tmp.getRightGroup().getColIndices(), grp.getRightGroup().getColIndices())) {
				iter.remove();
			}
			else
				_optChangeInSize = Math.min(_optChangeInSize, tmp.getChangeInSize());
		}
	}

	public void incrStats(int v1, int v2, int v3) {
		_stats[0] += v1;
		_stats[1] += v2;
		_stats[2] += v3;
	}

	public double getOptChangeInSize() {
		return _optChangeInSize;
	}

	public int[] getStats() {
		return _stats;
	}
}
