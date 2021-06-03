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
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.colgroup.AColGroup.CompressionType;
import org.apache.sysds.runtime.compress.estim.CompressedSizeEstimator;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.compress.utils.Util;

public class PlanningCoCoder {

	protected static final Log LOG = LogFactory.getLog(PlanningCoCoder.class.getName());

	/**
	 * The Valid coCoding techniques
	 */
	public enum PartitionerType {
		BIN_PACKING, STATIC, COST, COST_MATRIX_MULT, COST_TSMM;

		public static boolean isCostBased( PartitionerType pt) {
			switch(pt) {
				case COST_MATRIX_MULT:
				case COST_TSMM:
					return true;
				default:
					return false;
			}
		}
	}

	/**
	 * Main entry point of CoCode.
	 * 
	 * This package groups together ColGroups across columns, to improve compression further,
	 * 
	 * @param est      The size estimator used for estimating ColGroups potential sizes.
	 * @param colInfos The information already gathered on the individual ColGroups of columns.
	 * @param numRows  The number of rows in the input matrix.
	 * @param k        The concurrency degree allowed for this operation.
	 * @param cs       The compression settings used in the compression.
	 * @return The estimated (hopefully) best groups of ColGroups.
	 */
	public static CompressedSizeInfo findCoCodesByPartitioning(CompressedSizeEstimator est, CompressedSizeInfo colInfos,
		int numRows, int k, CompressionSettings cs) {

		// establish memo table for extracted column groups
		Memorizer mem = null;
		List<CompressedSizeInfoColGroup> constantGroups = null;
		if(cs.columnPartitioner == PartitionerType.BIN_PACKING) {
			constantGroups = new ArrayList<>();
			List<CompressedSizeInfoColGroup> newGroups = new ArrayList<>();
			mem = new Memorizer();
			for(CompressedSizeInfoColGroup g : colInfos.getInfo()) {
				if(g.getBestCompressionType(cs) == CompressionType.CONST)
					constantGroups.add(g);
				else {
					mem.put(g);
					newGroups.add(g);
				}
			}
			colInfos.setInfo(newGroups);
		}

		// Use column group partitioner to create partitions of columns
		CompressedSizeInfo bins = createColumnGroupPartitioner(cs.columnPartitioner, est, cs, numRows)
			.coCodeColumns(colInfos, k);

		if(cs.columnPartitioner == PartitionerType.BIN_PACKING) {
			getCoCodingGroupsBruteForce(bins, k, est, mem, cs);
			bins.getInfo().addAll(constantGroups);
		}

		return bins;
	}

	private static AColumnCoCoder createColumnGroupPartitioner(PartitionerType type, CompressedSizeEstimator est,
		CompressionSettings cs, int numRows) {
		switch(type) {
			case BIN_PACKING:
				return new CoCodeBinPacking(est, cs);
			case STATIC:
				return new CoCodeStatic(est, cs);
			case COST:
				return new CoCodeCost(est, cs);
			case COST_MATRIX_MULT:
				return new CoCodeCostMatrixMult(est, cs);
			case COST_TSMM:
				return new CoCodeCostTSMM(est, cs);
			default:
				throw new RuntimeException("Unsupported column group partitioner: " + type.toString());
		}
	}

	/**
	 * This methods verifies the coCoded bins actually are the best combinations within each individual Group based on
	 * the sample.
	 * 
	 * @param bins The bins constructed based on lightweight estimations
	 * @param k    The number of threads allowed to be used.
	 * @param est  The Estimator to be used.
	 * @return
	 */
	private static CompressedSizeInfo getCoCodingGroupsBruteForce(CompressedSizeInfo bins, int k,
		CompressedSizeEstimator est, Memorizer mem, CompressionSettings cs) {

		List<CompressedSizeInfoColGroup> finalGroups = new ArrayList<>();
		// For each bin of columns that is allowed to potentially cocode.
		for(CompressedSizeInfoColGroup bin : bins.getInfo()) {
			final int len = bin.getColumns().length;
			if(len == 0)
				continue;
			else if(len == 1)
				// early termination
				finalGroups.add(bin);
			else
				finalGroups.addAll(coCodeBruteForce(bin, est, mem, cs));
		}

		bins.setInfo(finalGroups);
		return bins;
	}

	private static List<CompressedSizeInfoColGroup> coCodeBruteForce(CompressedSizeInfoColGroup bin,
		CompressedSizeEstimator est, Memorizer mem, CompressionSettings cs) {

		List<int[]> workset = new ArrayList<>(bin.getColumns().length);

		for(int i = 0; i < bin.getColumns().length; i++)
			workset.add(new int[] {bin.getColumns()[i]});

		// process merging iterations until no more change
		while(workset.size() > 1) {
			long changeInSize = 0;
			CompressedSizeInfoColGroup tmp = null;
			int[] selected1 = null, selected2 = null;
			for(int i = 0; i < workset.size(); i++) {
				for(int j = i + 1; j < workset.size(); j++) {
					final int[] c1 = workset.get(i);
					final int[] c2 = workset.get(j);
					final long sizeC1 = mem.get(c1).getMinSize();
					final long sizeC2 = mem.get(c2).getMinSize();

					mem.incst1();
					// pruning filter : skip dominated candidates
					// Since even if the entire size of one of the column lists is removed,
					// it still does not improve compression
					if(-Math.min(sizeC1, sizeC2) > changeInSize)
						continue;

					// Join the two column groups.
					// and Memorize the new join.
					final CompressedSizeInfoColGroup c1c2Inf = mem.getOrCreate(c1, c2, est, cs);
					final long sizeC1C2 = c1c2Inf.getMinSize();

					long newSizeChangeIfSelected = sizeC1C2 - sizeC1 - sizeC2;
					// Select the best join of either the currently selected
					// or keep the old one.
					if((tmp == null && newSizeChangeIfSelected < changeInSize) || tmp != null &&
						(newSizeChangeIfSelected < changeInSize || newSizeChangeIfSelected == changeInSize &&
							c1c2Inf.getColumns().length < tmp.getColumns().length)) {
						changeInSize = newSizeChangeIfSelected;
						tmp = c1c2Inf;
						selected1 = c1;
						selected2 = c2;
					}
				}
			}

			if(tmp != null) {
				workset.remove(selected1);
				workset.remove(selected2);
				workset.add(tmp.getColumns());
			}
			else
				break;
		}

		LOG.debug(mem.stats());
		mem.resetStats();

		List<CompressedSizeInfoColGroup> ret = new ArrayList<>(workset.size());

		for(int[] w : workset)
			ret.add(mem.get(w));

		return ret;
	}

	public static class Memorizer {
		private final Map<ColIndexes, CompressedSizeInfoColGroup> mem;
		private int st1 = 0, st2 = 0, st3 = 0, st4 = 0;

		public Memorizer() {
			mem = new HashMap<>();
		}

		public void put(CompressedSizeInfoColGroup g) {
			mem.put(new ColIndexes(g.getColumns()), g);
		}

		public CompressedSizeInfoColGroup get(CompressedSizeInfoColGroup g) {
			return mem.get(new ColIndexes(g.getColumns()));
		}

		public CompressedSizeInfoColGroup get(int[] c) {
			return mem.get(new ColIndexes(c));
		}

		public CompressedSizeInfoColGroup getOrCreate(int[] c1, int[] c2, CompressedSizeEstimator est,
			CompressionSettings cs) {
			final int[] c = Util.join(c1, c2);
			final ColIndexes cI = new ColIndexes(Util.join(c1, c2));
			CompressedSizeInfoColGroup g = mem.get(cI);
			st2++;
			if(g == null) {
				final CompressedSizeInfoColGroup left = mem.get(new ColIndexes(c1));
				final CompressedSizeInfoColGroup right = mem.get(new ColIndexes(c2));
				final boolean leftConst = left.getBestCompressionType(cs) == CompressionType.CONST &&
					left.getNumOffs() == 0;
				final boolean rightConst = right.getBestCompressionType(cs) == CompressionType.CONST &&
					right.getNumOffs() == 0;
				if(leftConst)
					g = CompressedSizeInfoColGroup.addConstGroup(c, right, cs.validCompressions);
				else if(rightConst)
					g = CompressedSizeInfoColGroup.addConstGroup(c, left, cs.validCompressions);
				else {
					st3++;
					g = est.estimateJoinCompressedSize(left, right);
				}

				if(leftConst || rightConst)
					st4++;

				mem.put(cI, g);
			}
			return g;
		}

		public void incst1() {
			st1++;
		}

		public String stats() {
			return st1 + " " + st2 + " " + st3 + " " + st4;
		}

		public void resetStats() {
			st1 = 0;
			st2 = 0;
			st3 = 0;
			st4 = 0;
		}

		@Override
		public String toString() {
			return mem.toString();
		}
	}

	private static class ColIndexes {
		final int[] _indexes;

		public ColIndexes(int[] indexes) {
			_indexes = indexes;
		}

		@Override
		public int hashCode() {
			return Arrays.hashCode(_indexes);
		}

		@Override
		public boolean equals(Object that) {
			ColIndexes thatGrp = (ColIndexes) that;
			return Arrays.equals(_indexes, thatGrp._indexes);
		}
	}
}
