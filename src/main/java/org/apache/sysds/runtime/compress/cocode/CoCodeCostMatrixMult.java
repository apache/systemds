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
import org.apache.sysds.runtime.compress.estim.CompressedSizeEstimatorSample;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;

public class CoCodeCostMatrixMult extends AColumnCoCoder {

	protected CoCodeCostMatrixMult(CompressedSizeEstimator e, CompressionSettings cs) {
		super(e, cs);
	}

	@Override
	protected CompressedSizeInfo coCodeColumns(CompressedSizeInfo colInfos, int k) {

		List<CompressedSizeInfoColGroup> joinRes = join(colInfos.getInfo());

		if(_cs.samplingRatio < 0.1 && _est instanceof CompressedSizeEstimatorSample) {
			LOG.debug("Performing second join with double sample rate");
			CompressedSizeEstimatorSample estS = (CompressedSizeEstimatorSample) _est;
			estS.sampleData(estS.getSample().getNumRows() * 2);
			List<int[]> colG = new ArrayList<>(joinRes.size());
			for(CompressedSizeInfoColGroup g : joinRes)
				colG.add(g.getColumns());

			joinRes = join(estS.computeCompressedSizeInfos(colG, k));
		}

		colInfos.setInfo(joinRes);

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
					if(LOG.isDebugEnabled())
						LOG.debug("\nl:      " + l + "\nr:      " + r + "\njoined: " + g);
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
			if(elm == null) {
				this.cost = Double.POSITIVE_INFINITY;
			}
			else {

				final int nCols = elm.getColumns().length;
				final double nRows = _est.getNumRows();
				final double preAggregateCost = nRows;

				final int numberTuples = elm.getNumVals();
				final double tupleSparsity = elm.getTupleSparsity();
				final double postScalingCost = (nCols > 1 && tupleSparsity > 0.4) ? numberTuples *
					nCols : numberTuples * nCols * tupleSparsity;

				this.cost = preAggregateCost + postScalingCost;
			}
		}

		@Override
		public int compareTo(CostOfJoin o) {
			return cost == o.cost ? 0 : cost > o.cost ? 1 : -1;
		}

		@Override
		public String toString() {
			StringBuilder sb = new StringBuilder();
			sb.append(cost);
			sb.append(" - ");
			sb.append(elm.getBestCompressionType());
			sb.append(" nrVals: ");
			sb.append(elm.getNumVals());
			sb.append(" ");
			sb.append(Arrays.toString(elm.getColumns()));

			return sb.toString();
		}
	}
}
