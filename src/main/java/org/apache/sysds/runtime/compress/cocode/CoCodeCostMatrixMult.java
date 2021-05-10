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
import java.util.Arrays;
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
public class CoCodeCostMatrixMult extends AColumnCoCoder {

	/**
	 * This value specifies the maximum distinct count allowed int a coCoded group. Note that this value is the number
	 * of distinct tuples not the total number of values. That value can be calculated by multiplying the number of
	 * tuples with columns in the coCoded group.
	 */
	private final int largestDistinct;

	private final int toSmallForAnalysis;

	protected CoCodeCostMatrixMult(CompressedSizeEstimator e, CompressionSettings cs, int numRows) {
		super(e, cs, numRows);
		largestDistinct = Math.max(256, (int) (_est.getNumRows() * _est.getNumColumns() * cs.coCodePercentage * 0.2));
		toSmallForAnalysis = Math.min(Math.max(256, largestDistinct / 4), 1028);
		LOG.debug("CocodeCost largest Distinct: " + largestDistinct + " toSmallForAnalysis: " + toSmallForAnalysis);
	}

	@Override
	protected CompressedSizeInfo coCodeColumns(CompressedSizeInfo colInfos, int k) {
		colInfos.setInfo(join(colInfos.getInfo()));
		return colInfos;
	}

	private List<CompressedSizeInfoColGroup> join(List<CompressedSizeInfoColGroup> currentGroups) {

		Queue<CostOfJoin> que = new PriorityQueue<>(currentGroups.size());

		List<CompressedSizeInfoColGroup> ret = new ArrayList<>();
		for(CompressedSizeInfoColGroup g : currentGroups)
			que.add(new CostOfJoin(g));

		while(true) {
			if(que.peek() != null) {
				final CostOfJoin l = que.poll();
				if(que.peek() != null) {
					final CostOfJoin r = que.poll();
					final double costIndividual = (l.cost + r.cost);
					final CostOfJoin g = new CostOfJoin(joinWithAnalysis(l.elm, r.elm));
					if(g.cost < costIndividual)
						que.add(g);
					else {
						ret.add(l.elm);
						que.add(r);
					}
				}
				else {
					ret.add(l.elm);
					break;
				}
			}
			else
				break;
		}
		for(CostOfJoin g : que)
			ret.add(g.elm);

		return ret;
	}

	private class CostOfJoin implements Comparable<CostOfJoin> {
		protected final CompressedSizeInfoColGroup elm;
		protected final double cost;

		protected CostOfJoin(CompressedSizeInfoColGroup elm) {
			this.elm = elm;
			final int nRows = _est.getNumRows();
			final double commonFraction = elm.getMostCommonFraction();
			final double rowsToProcess = commonFraction > 0.2 ? nRows * (1 - Math.min(commonFraction, 0.95)) : nRows;
			this.cost = rowsToProcess + elm.getNumVals() * elm.getColumns().length;

		}

		@Override
		public int compareTo(CostOfJoin o) {
			return cost == o.cost ? 0 : cost > o.cost ? 1 : -1;
		}

		@Override
		public String toString() {
			StringBuilder sb = new StringBuilder();
			sb.append("\n");
			sb.append(cost);
			sb.append(" - ");
			sb.append(Arrays.toString(elm.getColumns()));
			return sb.toString();
		}
	}
}
