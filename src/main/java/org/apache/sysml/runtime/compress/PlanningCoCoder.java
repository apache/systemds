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

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.PriorityQueue;
import java.util.TreeMap;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.compress.estim.CompressedSizeEstimator;

public class PlanningCoCoder 
{
	//constants for weight computation
	private final static float GROUPABILITY_THRESHOLD = 0.00064f;
	private final static boolean USE_BIN_WEIGHT = false;
	private final static float PARTITION_WEIGHT = 0.05F; //higher values lead to more grouping
	private final static float PARTITION_SIZE = PARTITION_WEIGHT * GROUPABILITY_THRESHOLD;
	private final static float BIN_WEIGHT_PARAM = -0.65f; //lower values lead to more grouping

	/**
	 * 
	 * @param sizeEstimator
	 * @param availCols
	 * @param colsCardinalities
	 * @param compressedSize
	 * @param numRows
	 * @param sparsity
	 * @return
	 * @throws DMLRuntimeException 
	 */
	public static List<int[]> findCocodesByPartitioning(CompressedSizeEstimator sizeEstimator, List<Integer> availCols, 
			List<Integer> colsCardinalities, List<Long> compressedSize, int numRows, double sparsity, int k) 
		throws DMLRuntimeException 
	{
		List<int[]> retGroups = new ArrayList<int[]>();
		
		// filtering out non-groupable columns as singleton groups
		// weighted of each column is the ratio of its cardinality to the number
		// of rows scaled by the matrix sparsity
		int numCols = availCols.size();
		List<Integer> groupCols = new ArrayList<Integer>();
		List<Float> groupColWeights = new ArrayList<Float>();
		HashMap<Integer, GroupableColInfo> groupColsInfo = new HashMap<Integer, GroupableColInfo>();
		for (int i = 0; i < numCols; i++) {
			int colIx = availCols.get(i);
			int cardinality = colsCardinalities.get(i);
			float weight = ((float) cardinality) / numRows;
			if (weight <= GROUPABILITY_THRESHOLD) {
				groupCols.add(colIx);
				groupColWeights.add(weight);
				groupColsInfo.put(colIx, new GroupableColInfo(weight,compressedSize.get(i)));
			} else {
				retGroups.add(new int[] { colIx });
			}
		}
		
		// bin packing based on PARTITION_WEIGHT and column weights
		float weight = computeWeightForCoCoding(numRows, sparsity);
		TreeMap<Float, List<List<Integer>>> bins = new PlanningBinPacker(
				weight, groupCols, groupColWeights).packFirstFit();

		// brute force grouping within each partition
		retGroups.addAll( (k > 1) ?
				getCocodingGroupsBruteForce(bins, groupColsInfo, sizeEstimator, numRows, k) :
				getCocodingGroupsBruteForce(bins, groupColsInfo, sizeEstimator, numRows));
			
		return retGroups;
	}
	
	/**
	 * 
	 * @param bins
	 * @param groupColsInfo
	 * @param estim
	 * @param rlen
	 * @return
	 */
	private static List<int[]> getCocodingGroupsBruteForce(TreeMap<Float, List<List<Integer>>> bins, HashMap<Integer, GroupableColInfo> groupColsInfo, CompressedSizeEstimator estim, int rlen) 
	{
		List<int[]> retGroups = new ArrayList<int[]>();		
		for (List<List<Integer>> binList : bins.values()) {
			for (List<Integer> bin : binList) {
				// building an array of singleton CoCodingGroup
				ArrayList<PlanningCoCodingGroup> sgroups = new ArrayList<PlanningCoCodingGroup>();
				for (Integer col : bin)
					sgroups.add(new PlanningCoCodingGroup(col, groupColsInfo.get(col)));
				// brute force co-coding	
				PlanningCoCodingGroup[] outputGroups = findCocodesBruteForce(
						estim, rlen, sgroups.toArray(new PlanningCoCodingGroup[0]));
				for (PlanningCoCodingGroup grp : outputGroups)
					retGroups.add(grp.getColIndices());
			}
		}
		
		return retGroups;
	}
	
	/**
	 * 
	 * @param bins
	 * @param groupColsInfo
	 * @param estim
	 * @param rlen
	 * @param k
	 * @return
	 * @throws DMLRuntimeException 
	 */
	private static List<int[]> getCocodingGroupsBruteForce(TreeMap<Float, List<List<Integer>>> bins, HashMap<Integer, GroupableColInfo> groupColsInfo, CompressedSizeEstimator estim, int rlen, int k) 
		throws DMLRuntimeException 
	{
		List<int[]> retGroups = new ArrayList<int[]>();		
		try {
			ExecutorService pool = Executors.newFixedThreadPool( k );
			ArrayList<CocodeTask> tasks = new ArrayList<CocodeTask>();
			for (List<List<Integer>> binList : bins.values())
				for (List<Integer> bin : binList) {
					// building an array of singleton CoCodingGroup
					ArrayList<PlanningCoCodingGroup> sgroups = new ArrayList<PlanningCoCodingGroup>();
					for (Integer col : bin)
						sgroups.add(new PlanningCoCodingGroup(col, groupColsInfo.get(col)));
					tasks.add(new CocodeTask(estim, sgroups, rlen));
				}
			List<Future<PlanningCoCodingGroup[]>> rtask = pool.invokeAll(tasks);	
			for( Future<PlanningCoCodingGroup[]> lrtask : rtask )
				for (PlanningCoCodingGroup grp : lrtask.get())
					retGroups.add(grp.getColIndices());
			pool.shutdown();
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
		
		return retGroups;
	}

	/**
	 * Identify columns to code together. Uses a greedy approach that merges
	 * pairs of column groups into larger groups. Each phase of the greedy
	 * algorithm considers all combinations of pairs to merge.
	 * 
	 */
	private static PlanningCoCodingGroup[] findCocodesBruteForce(
			CompressedSizeEstimator sizeEstimator, float numRowsWeight,
			PlanningCoCodingGroup[] singltonGroups) 
	{
		// Populate a priority queue with all available 2-column cocodings.
		PriorityQueue<PlanningGroupMergeAction> q = new PriorityQueue<PlanningGroupMergeAction>();
		for (int leftIx = 0; leftIx < singltonGroups.length; leftIx++) {
			PlanningCoCodingGroup leftGrp = singltonGroups[leftIx];
			for (int rightIx = leftIx + 1; rightIx < singltonGroups.length; rightIx++) {
				PlanningCoCodingGroup rightGrp = singltonGroups[rightIx];
				// at least one of the two groups should be low-cardinality
				float cardRatio = leftGrp.getCardinalityRatio() + rightGrp.getCardinalityRatio(); 
				if ( cardRatio < GROUPABILITY_THRESHOLD) {
					PlanningGroupMergeAction potentialMerge = new PlanningGroupMergeAction(
							sizeEstimator, numRowsWeight, leftGrp, rightGrp);
					if (potentialMerge.getChangeInSize() < 0) {
						q.add(potentialMerge);
					}
				}
			}
		}
		PlanningCoCodingGroup[] colGroups = singltonGroups;
		
		// Greedily merge groups until we can no longer reduce the number of
		// runs by merging groups
		while (q.size() > 0) {
			PlanningGroupMergeAction merge = q.poll();

			// The queue can contain merge actions involving column groups that
			// have already been merged.
			// Filter those actions out.
			int leftIx = findInArray(colGroups, merge.getLeftGrp());
			int rightIx = findInArray(colGroups, merge.getRightGrp());
			if (leftIx < 0 || rightIx < 0) {
				// One or more of the groups to be merged has already been made
				// part of another group.
				// Drop the merge action.
			} else {
				PlanningCoCodingGroup mergedGrp = merge.getMergedGrp();

				PlanningCoCodingGroup[] newColGroups = new PlanningCoCodingGroup[colGroups.length - 1];
				int targetIx = 0;
				for (int i = 0; i < colGroups.length; i++) {
					if (i != leftIx && i != rightIx) {
						newColGroups[targetIx] = colGroups[i];
						targetIx++;
					}
				}

				// New group goes at the end to (hopefully) speed up future
				// linear search operations
				newColGroups[newColGroups.length - 1] = mergedGrp;

				// Consider merging the new group with all the other
				// pre-existing groups.
				for (int i = 0; i < newColGroups.length - 1; i++) {
					PlanningCoCodingGroup newLeftGrp = newColGroups[i];
					PlanningCoCodingGroup newRightGrp = mergedGrp;
					if (newLeftGrp.getCardinalityRatio()
							+ newRightGrp.getCardinalityRatio() < GROUPABILITY_THRESHOLD) {
						PlanningGroupMergeAction newPotentialMerge = new PlanningGroupMergeAction(
								sizeEstimator, numRowsWeight, newLeftGrp,
								newRightGrp);
						if (newPotentialMerge.getChangeInSize() < 0) {
							q.add(newPotentialMerge);
						}
					}
				}
				colGroups = newColGroups;
			}
		}
		return colGroups;
	}

	/**
	 * 
	 * @param numRows
	 * @param sparsity
	 * @return
	 */
	private static float computeWeightForCoCoding(int numRows, double sparsity)
	{
		if( USE_BIN_WEIGHT ) { //new method (non-conclusive)
			//return (float) Math.pow(numRows*sparsity,BIN_WEIGHT_PARAM);
			return (float) Math.pow(numRows,BIN_WEIGHT_PARAM);
		}
		else {
			return PARTITION_SIZE;
		}
	}
	
	/**
	 * 
	 * @param arr
	 * @param val
	 * @return
	 */
	private static int findInArray(Object[] arr, Object val) {
		for (int i = 0; i < arr.length; i++) {
			if (arr[i].equals(val)) {
				return i;
			}
		}
		return -1;
	}
	
	/**
	 * 
	 */
	protected static class GroupableColInfo {
		float cardRatio;
		long size;

		public GroupableColInfo(float lcardRatio, long lsize) {
			cardRatio = lcardRatio;
			size = lsize;
		}
	}
	
	/**
	 * 
	 */
	private static class CocodeTask implements Callable<PlanningCoCodingGroup[]> 
	{
		private CompressedSizeEstimator _estim = null;
		private ArrayList<PlanningCoCodingGroup> _sgroups = null;
		private int _rlen = -1;
		
		protected CocodeTask( CompressedSizeEstimator estim, ArrayList<PlanningCoCodingGroup> sgroups, int rlen )  {
			_estim = estim;
			_sgroups = sgroups;
			_rlen = rlen;
		}
		
		@Override
		public PlanningCoCodingGroup[] call() throws DMLRuntimeException {
			// brute force co-coding	
			return findCocodesBruteForce(_estim, _rlen, 
					_sgroups.toArray(new PlanningCoCodingGroup[0]));
		}
	}
}
