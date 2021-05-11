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

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Queue;

import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.estim.CompressedSizeEstimator;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;

/**
 * Column group partitioning by number of distinct items estimated. This allows us to join columns based on the worst
 * case estimate of the joined sizes. Then once we decide to join, if the worst case is okay, we then analyze the actual
 * cardinality of the join.
 * 
 * This method allows us to compress many more columns than the BinPacking
 * 
 */
public class CoCodeCost extends AColumnCoCoder {

	/**
	 * This value specifies the maximum distinct count allowed int a coCoded group. Note that this value is the number
	 * of distinct tuples not the total number of values. That value can be calculated by multiplying the number of
	 * tuples with columns in the coCoded group.
	 */
	private final int largestDistinct;

	private final static int toSmallForAnalysis = 64;

	protected CoCodeCost(CompressedSizeEstimator sizeEstimator, CompressionSettings cs, int numRows) {
		super(sizeEstimator, cs, numRows);
		largestDistinct = Math.min(4096, Math.max(256, (int) (sizeEstimator.getNumRows() * cs.coCodePercentage)));
	}

	@Override
	protected CompressedSizeInfo coCodeColumns(CompressedSizeInfo colInfos, int k) {
		colInfos.setInfo(join(colInfos.getInfo()));
		return colInfos;
	}

	private List<CompressedSizeInfoColGroup> join(List<CompressedSizeInfoColGroup> currentGroups) {

		Comparator<CompressedSizeInfoColGroup> comp = Comparator.comparing(CompressedSizeInfoColGroup::getNumVals);
		Queue<CompressedSizeInfoColGroup> que = new PriorityQueue<>(currentGroups.size(), comp);
		List<CompressedSizeInfoColGroup> ret = new ArrayList<>();

		for(CompressedSizeInfoColGroup g : currentGroups)
			que.add(g);

		boolean finished = false;
		while(!finished) {
			if(que.peek() != null) {
				CompressedSizeInfoColGroup l = que.poll();
				if(que.peek() != null) {
					CompressedSizeInfoColGroup r = que.poll();
					int worstCaseJoinedSize = l.getNumVals() * r.getNumVals();
					if(worstCaseJoinedSize < toSmallForAnalysis)
						que.add(joinWithoutAnalysis(l, r));
					else if(worstCaseJoinedSize < largestDistinct) {
						CompressedSizeInfoColGroup g = joinWithAnalysis(l, r);
						if(g.getNumVals() < largestDistinct)
							que.add(joinWithAnalysis(l, r));
						else {
							ret.add(l);
							que.add(r);
						}
					}
					else {
						ret.add(l);
						que.add(r);
					}
				}
				else
					ret.add(l);
			}
			else
				finished = true;
		}

		for(CompressedSizeInfoColGroup g : que)
			ret.add(g);

		return ret;
	}
}
