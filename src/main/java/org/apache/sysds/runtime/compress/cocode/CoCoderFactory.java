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
import java.util.List;

import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.cost.ACostEstimate;
import org.apache.sysds.runtime.compress.estim.AComEst;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfoColGroup;
import org.apache.sysds.runtime.compress.utils.IntArrayList;

public interface CoCoderFactory {

	/**
	 * The Valid coCoding techniques
	 */
	public enum PartitionerType {
		BIN_PACKING, STATIC, PRIORITY_QUE, GREEDY, AUTO;
	}

	/**
	 * Main entry point of CoCode.
	 * 
	 * This package groups together ColGroups across columns, to improve compression further,
	 * 
	 * @param est           The size estimator used for estimating ColGroups potential sizes and construct compression
	 *                      info objects
	 * @param colInfos      The information already gathered on the individual ColGroups of columns.
	 * @param k             The concurrency degree allowed for this operation.
	 * @param costEstimator The Cost estimator to estimate the cost of the compression
	 * @param cs            The compression settings used in the compression.
	 * @return The estimated (hopefully) best groups of ColGroups.
	 */
	public static CompressedSizeInfo findCoCodesByPartitioning(AComEst est, CompressedSizeInfo colInfos,
		int k, ACostEstimate costEstimator, CompressionSettings cs) {

		// Use column group partitioner to create partitions of columns
		AColumnCoCoder co = createColumnGroupPartitioner(cs.columnPartitioner, est, costEstimator, cs);

		// Find out if any of the groups are empty.
		boolean containsEmpty = false;
		for(CompressedSizeInfoColGroup g : colInfos.compressionInfo) {
			if(g.isEmpty()) {
				containsEmpty = true;
				break;
			}
		}

		// if there are no empty columns then try cocode algorithms for all columns
		if(!containsEmpty)
			return co.coCodeColumns(colInfos, k);

		// extract all empty columns
		IntArrayList emptyCols = new IntArrayList();
		List<CompressedSizeInfoColGroup> notEmpty = new ArrayList<>();

		for(CompressedSizeInfoColGroup g : colInfos.compressionInfo) {
			if(g.isEmpty())
				emptyCols.appendValue(g.getColumns()[0]);
			else
				notEmpty.add(g);
		}

		final int nRow = colInfos.compressionInfo.get(0).getNumRows();

		if(notEmpty.isEmpty()) { // if all empty (unlikely but could happen)
			CompressedSizeInfoColGroup empty = new CompressedSizeInfoColGroup(emptyCols.extractValues(true), nRow);
			return new CompressedSizeInfo(empty);
		}

		// cocode all not empty columns
		colInfos.compressionInfo = notEmpty;
		colInfos = co.coCodeColumns(colInfos, k);

		// add empty columns back as single columns
		colInfos.compressionInfo.add(new CompressedSizeInfoColGroup(emptyCols.extractValues(true), nRow));
		return colInfos;
	}

	private static AColumnCoCoder createColumnGroupPartitioner(PartitionerType type, AComEst est,
		ACostEstimate costEstimator, CompressionSettings cs) {
		switch(type) {
			case AUTO:
				return new CoCodeHybrid(est, costEstimator, cs);
			case GREEDY:
				return new CoCodeGreedy(est, costEstimator, cs);
			case BIN_PACKING:
				return new CoCodeBinPacking(est, costEstimator, cs);
			case STATIC:
				return new CoCodeStatic(est, costEstimator, cs);
			case PRIORITY_QUE:
				return new CoCodePriorityQue(est, costEstimator, cs);
			default:
				throw new RuntimeException("Unsupported column group partitioner: " + type.toString());
		}
	}
}
