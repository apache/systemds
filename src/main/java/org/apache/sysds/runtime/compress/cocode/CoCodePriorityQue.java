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
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;

import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.DMLCompressionException;
import org.apache.sysds.runtime.compress.cost.ACostEstimate;
import org.apache.sysds.runtime.compress.estim.AComEst;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.util.CommonThreadPool;

/**
 * Column group partitioning by number of distinct items estimated. This allows us to join columns based on the worst
 * case estimate of the joined sizes. Then once we decide to join, if the worst case is okay, we then analyze the actual
 * cardinality of the join.
 * 
 * This method allows us to compress many more columns than the BinPacking
 * 
 */
public class CoCodePriorityQue extends AColumnCoCoder {

	private static final int COL_COMBINE_THRESHOLD = 1024;

	protected CoCodePriorityQue(AComEst sizeEstimator, ACostEstimate costEstimator,
		CompressionSettings cs) {
		super(sizeEstimator, costEstimator, cs);
	}

	@Override
	protected CompressedSizeInfo coCodeColumns(CompressedSizeInfo colInfos, int k) {
		colInfos.setInfo(join(colInfos.getInfo(), _sest, _cest, 1, k));
		return colInfos;
	}

	protected static List<CompressedSizeInfoColGroup> join(List<CompressedSizeInfoColGroup> groups,
		AComEst sEst, ACostEstimate cEst, int minNumGroups, int k) {

		if(groups.size() > COL_COMBINE_THRESHOLD && k > 1)
			return combineMultiThreaded(groups, sEst, cEst, minNumGroups, k);
		else
			return combineSingleThreaded(groups, sEst, cEst, minNumGroups);
	}

	private static List<CompressedSizeInfoColGroup> combineMultiThreaded(List<CompressedSizeInfoColGroup> groups,
		AComEst sEst, ACostEstimate cEst, int minNumGroups, int k) {
		try {
			final ExecutorService pool = CommonThreadPool.get(k);
			final List<PQTask> tasks = new ArrayList<>();
			final int blkSize = Math.max(groups.size() / k, 500);
			for(int i = 0; i < groups.size(); i += blkSize)
				tasks.add(new PQTask(groups, i, Math.min(i + blkSize, groups.size()), sEst, cEst, minNumGroups));

			List<CompressedSizeInfoColGroup> ret = null;
			for(Future<List<CompressedSizeInfoColGroup>> t : pool.invokeAll(tasks)) {

				List<CompressedSizeInfoColGroup> p = t.get();
				if(ret == null)
					ret = p;
				else
					ret.addAll(p);
			}
			return ret;
		}
		catch(Exception e) {
			throw new DMLCompressionException("Failed parallel priority que cocoding", e);
		}
	}

	private static List<CompressedSizeInfoColGroup> combineSingleThreaded(List<CompressedSizeInfoColGroup> groups,
		AComEst sEst, ACostEstimate cEst, int minNumGroups) {
		return combineBlock(groups, 0, groups.size(), sEst, cEst, minNumGroups);
	}

	private static List<CompressedSizeInfoColGroup> combineBlock(List<CompressedSizeInfoColGroup> groups, int start,
		int end, AComEst sEst, ACostEstimate cEst, int minNumGroups) {
		Queue<CompressedSizeInfoColGroup> que = getQue(end - start, cEst);

		for(int i = start; i < end; i++) {
			CompressedSizeInfoColGroup g = groups.get(i);
			if(g != null)
				que.add(g);
		}

		return combineBlock(que, sEst, cEst, minNumGroups);
	}

	private static List<CompressedSizeInfoColGroup> combineBlock(Queue<CompressedSizeInfoColGroup> que,
		AComEst sEst, ACostEstimate cEst, int minNumGroups) {

		List<CompressedSizeInfoColGroup> ret = new ArrayList<>();
		CompressedSizeInfoColGroup l = null;
		l = que.poll();
		int groupNr = ret.size() + que.size();
		while(que.peek() != null && groupNr >= minNumGroups) {
			CompressedSizeInfoColGroup r = que.peek();
			CompressedSizeInfoColGroup g = sEst.combine(l, r);
			if(g != null) {
				double costOfJoin = cEst.getCost(g);
				double costIndividual = cEst.getCost(l) + cEst.getCost(r);

				if(costOfJoin < costIndividual) {
					que.poll();
					int numColumns = g.getColumns().length;
					if(numColumns > 128)
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

	private static Queue<CompressedSizeInfoColGroup> getQue(int size, ACostEstimate cEst) {
		Comparator<CompressedSizeInfoColGroup> comp = Comparator.comparing(x -> cEst.getCost(x));
		Queue<CompressedSizeInfoColGroup> que = new PriorityQueue<>(size, comp);
		return que;
	}

	protected static class PQTask implements Callable<List<CompressedSizeInfoColGroup>> {

		private final List<CompressedSizeInfoColGroup> _groups;
		private final int _start;
		private final int _end;
		private final AComEst _sEst;
		private final ACostEstimate _cEst;
		private final int _minNumGroups;

		protected PQTask(List<CompressedSizeInfoColGroup> groups, int start, int end, AComEst sEst,
			ACostEstimate cEst, int minNumGroups) {
			_groups = groups;
			_start = start;
			_end = end;
			_sEst = sEst;
			_cEst = cEst;
			_minNumGroups = minNumGroups;
		}

		@Override
		public List<CompressedSizeInfoColGroup> call() {
			try {
				return combineBlock(_groups, _start, _end, _sEst, _cEst, _minNumGroups);
			}
			catch(Exception e) {
				throw new DMLCompressionException("Falied PQTask ", e);
			}
		}
	}
}
