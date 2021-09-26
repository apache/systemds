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
		colInfos.setInfo(join(colInfos.getInfo(), _sest, _cest, 1));
		return colInfos;
	}

	protected static List<CompressedSizeInfoColGroup> join(List<CompressedSizeInfoColGroup> currentGroups,
		CompressedSizeEstimator sEst, ICostEstimate cEst, int minNumGroups) {
		Comparator<CompressedSizeInfoColGroup> comp = Comparator.comparing(x -> cEst.getCostOfColumnGroup(x));
		Queue<CompressedSizeInfoColGroup> que = new PriorityQueue<>(currentGroups.size(), comp);
		List<CompressedSizeInfoColGroup> ret = new ArrayList<>();

		for(CompressedSizeInfoColGroup g : currentGroups)
			if(g != null)
				que.add(g);

		CompressedSizeInfoColGroup l = null;

		l = que.poll();
		int groupNr = ret.size() + que.size();
		while(que.peek() != null && groupNr >= minNumGroups) {
			CompressedSizeInfoColGroup r = que.peek();
			CompressedSizeInfoColGroup g = sEst.estimateJoinCompressedSize(l, r);
			if(g != null) {
				double costOfJoin = cEst.getCostOfColumnGroup(g);
				double costIndividual = cEst.getCostOfColumnGroup(l) + cEst.getCostOfColumnGroup(r);

				if(costOfJoin < costIndividual) {
					que.poll();
					int numColumns = g.getColumns().length;
					if(minNumGroups != 0 && numColumns > 8)
						ret.add(g); // Add this column group to ret, since it already is very CoCoded.
					else if(numColumns > 128)
						ret.add(g);
					else
						que.add(g);
				}
				else
					ret.add(l);
			}
			else
				ret.add(l);

			l = que.poll();
			groupNr = ret.size() + que.size();
		}

		if(l != null)
			ret.add(l);

		for(CompressedSizeInfoColGroup g : que)
			ret.add(g);

		return ret;
	}
}
