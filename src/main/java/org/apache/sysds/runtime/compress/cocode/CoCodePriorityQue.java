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
import org.apache.sysds.runtime.compress.cost.ICostEstimate;
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
public class CoCodePriorityQue extends AColumnCoCoder {

	protected CoCodePriorityQue(CompressedSizeEstimator sizeEstimator, ICostEstimate costEstimator,
		CompressionSettings cs) {
		super(sizeEstimator, costEstimator, cs);
	}

	@Override
	protected CompressedSizeInfo coCodeColumns(CompressedSizeInfo colInfos, int k) {
		colInfos.setInfo(join(colInfos.getInfo()));
		return colInfos;
	}

	private List<CompressedSizeInfoColGroup> join(List<CompressedSizeInfoColGroup> currentGroups) {
		Comparator<CompressedSizeInfoColGroup> comp = Comparator.comparing(x -> _cest.getCostOfColumnGroup(x));
		Queue<CompressedSizeInfoColGroup> que = new PriorityQueue<>(currentGroups.size(), comp);
		List<CompressedSizeInfoColGroup> ret = new ArrayList<>();
		for(CompressedSizeInfoColGroup g : currentGroups)
			if(g != null)
				que.add(g);

		CompressedSizeInfoColGroup l = null;
		if(_cest.isCompareAll()) {
			double costBeforeJoin = _cest.getCostOfCollectionOfGroups(que);
			l = que.poll();
			while(que.peek() != null) {

				CompressedSizeInfoColGroup r = que.poll();
				final CompressedSizeInfoColGroup g = joinWithAnalysis(l, r);
				if(g != null) {
					final double costOfJoin = _cest.getCostOfCollectionOfGroups(que, g);
					if(costOfJoin < costBeforeJoin) {
						costBeforeJoin = costOfJoin;
						que.add(g);
					}
					else {
						que.add(r);
						ret.add(l);
					}
				}
				else {
					que.add(r);
					ret.add(l);
				}

				l = que.poll();
			}
		}
		else {
			l = que.poll();
			while(que.peek() != null) {
				CompressedSizeInfoColGroup r = que.peek();
				if(_cest.shouldTryJoin(l, r)) {
					CompressedSizeInfoColGroup g = joinWithAnalysis(l, r);
					if(g != null) {
						double costOfJoin = _cest.getCostOfColumnGroup(g);
						double costIndividual = _cest.getCostOfColumnGroup(l) + _cest.getCostOfColumnGroup(r);

						if(costOfJoin < costIndividual) {
							que.poll();
							que.add(g);
						}
						else
							ret.add(l);
					}
					else
						ret.add(l);
				}
				else
					ret.add(l);

				l = que.poll();
			}
		}
		if(l != null)
			ret.add(l);

		for(CompressedSizeInfoColGroup g : que)
			ret.add(g);

		return ret;
	}
}
