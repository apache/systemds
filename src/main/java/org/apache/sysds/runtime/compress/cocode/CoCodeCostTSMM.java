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

public class CoCodeCostTSMM extends AColumnCoCoder {

	protected CoCodeCostTSMM(CompressedSizeEstimator sizeEstimator, ICostEstimate costEstimator,
		CompressionSettings cs) {
		super(sizeEstimator, costEstimator, cs);
	}

	@Override
	protected CompressedSizeInfo coCodeColumns(CompressedSizeInfo colInfos, int k) {

		List<CompressedSizeInfoColGroup> joinRes = join(colInfos.getInfo());

		colInfos.setInfo(joinRes);

		return colInfos;
	}

	private List<CompressedSizeInfoColGroup> join(List<CompressedSizeInfoColGroup> currentGroups) {

		Comparator<CompressedSizeInfoColGroup> comp = Comparator.comparing(CompressedSizeInfoColGroup::getNumVals);
		Queue<CompressedSizeInfoColGroup> que = new PriorityQueue<>(currentGroups.size(), comp);

		List<CompressedSizeInfoColGroup> ret = new ArrayList<>();
		for(CompressedSizeInfoColGroup g : currentGroups)
			que.add(g);

		double currentCost = getCost(que, ret);
		while(true) {
			if(que.peek() != null) {
				final CompressedSizeInfoColGroup l = que.poll();
				if(que.peek() != null) {
					final CompressedSizeInfoColGroup r = que.poll();
					final CompressedSizeInfoColGroup g = joinWithAnalysis(l, r);
					final double newCost = getCost(que, ret, g);
					if(newCost < currentCost) {
						currentCost = newCost;
						que.add(g);
					}
					else {
						ret.add(l);
						que.add(r);
					}
				}
				else {
					ret.add(l);
					break;
				}
			}
			else
				break;
		}

		for(CompressedSizeInfoColGroup g : que)
			ret.add(g);

		return ret;
	}

	private double getCost(Queue<CompressedSizeInfoColGroup> que, List<CompressedSizeInfoColGroup> ret) {
		CompressedSizeInfoColGroup[] queValues = que.toArray(new CompressedSizeInfoColGroup[que.size()]);
		return getCost(queValues, ret);
	}

	private double getCost(Queue<CompressedSizeInfoColGroup> que, List<CompressedSizeInfoColGroup> ret,
		CompressedSizeInfoColGroup g) {
		CompressedSizeInfoColGroup[] queValues = que.toArray(new CompressedSizeInfoColGroup[que.size()]);
		double cost = getCost(queValues, ret);
		cost += getCostOfSelfTSMM(g);
		for(int i = 0; i < queValues.length; i++)
			cost += getCostOfLeftTransposedMM(queValues[i], g);

		for(int i = 0; i < ret.size(); i++)
			cost += getCostOfLeftTransposedMM(ret.get(i), g);
		return cost;
	}

	private double getCost(CompressedSizeInfoColGroup[] queValues, List<CompressedSizeInfoColGroup> ret) {
		double cost = 0;
		for(int i = 0; i < queValues.length; i++) {
			cost += getCostOfSelfTSMM(queValues[i]);
			for(int j = i + 1; j < queValues.length; j++)
				cost += getCostOfLeftTransposedMM(queValues[i], queValues[j]);

			for(CompressedSizeInfoColGroup g : ret)
				cost += getCostOfLeftTransposedMM(queValues[i], g);

		}
		for(int i = 0; i < ret.size(); i++) {
			cost += getCostOfSelfTSMM(ret.get(i));
			for(int j = i + 1; j < ret.size(); j++)
				cost += getCostOfLeftTransposedMM(ret.get(i), ret.get(j));

		}
		return cost;
	}

	private static double getCostOfSelfTSMM(CompressedSizeInfoColGroup g) {
		double cost = 0;
		final int nCol = g.getColumns().length;
		cost += g.getNumVals() * (nCol * (nCol + 1)) / 2;
		return cost;
	}

	private double getCostOfLeftTransposedMM(CompressedSizeInfoColGroup gl, CompressedSizeInfoColGroup gr) {
		final int nRows = _sest.getNumRows();
		final int nColsL = gl.getColumns().length;
		final int nColsR = gl.getColumns().length;

		// final double preAggLeft = (nRows / (1 - gl.getMostCommonFraction())) * nColsL;
		// final double preAggRight = (nRows / (1 - gr.getMostCommonFraction())) * nColsR;

		final double preAggLeft = nRows;
		final double preAggRight = nRows;

		final double tsL = gl.getTupleSparsity();
		final double tsR = gr.getTupleSparsity();

		// final double tsL = 1;
		// final double tsR = 1;

		final int nvL = gl.getNumVals();
		final int nvR = gr.getNumVals();

		final double postScaleLeft = nColsL > 1 && tsL > 0.4 ? nvL * nColsL : nvL * nColsL * tsL;
		final double postScaleRight = nColsR > 1 && tsR > 0.4 ? nvR * nColsR : nvR * nColsR * tsR;

		final double costLeft = preAggLeft + postScaleLeft * 5;
		final double costRight = preAggRight + postScaleRight * 5;

		return Math.min(costLeft, costRight);
	}

}
