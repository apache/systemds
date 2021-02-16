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

import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.estim.CompressedSizeEstimator;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;

public class PlanningCoCoder {

	// private static final Log LOG = LogFactory.getLog(PlanningCoCoder.class.getName());

	public enum PartitionerType {
		BIN_PACKING, STATIC, COST, COST_REEVALUATE;

		public static boolean isCost(PartitionerType x) {
			switch(x) {
				case COST:
				case COST_REEVALUATE:
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
	 * @param sizeEstimator The size estimator used for estimating ColGroups potential sizes.
	 * @param colInfos      The information already gathered on the individual ColGroups of columns.
	 * @param numRows       The number of rows in the input matrix.
	 * @param k             The concurrency degree allowed for this operation.
	 * @param cs            The Compression Settings used in the compression.
	 * @return The Estimated (hopefully) best groups of ColGroups.
	 */
	public static CompressedSizeInfo findCoCodesByPartitioning(CompressedSizeEstimator sizeEstimator,
		CompressedSizeInfo colInfos, int numRows, int k, CompressionSettings cs) {

		// use column group partitioner to create partitions of columns
		CompressedSizeInfo bins = createColumnGroupPartitioner(cs.columnPartitioner, sizeEstimator, cs, numRows)
			.partitionColumns(colInfos);

		if(cs.columnPartitioner == PartitionerType.COST) {
			return bins;
		}
		else if(cs.columnPartitioner == PartitionerType.COST_REEVALUATE) {

		}

		return bins;
	}

	private static AColumnGroupPartitioner createColumnGroupPartitioner(PartitionerType type,
		CompressedSizeEstimator sizeEstimator, CompressionSettings cs, int numRows) {
		switch(type) {
			case BIN_PACKING:
				return new ColumnGroupPartitionerBinPacking(sizeEstimator, cs, numRows);
			case STATIC:
				return new ColumnGroupPartitionerStatic(sizeEstimator, cs, numRows);
			case COST:
			case COST_REEVALUATE:
				return new ColumnGroupPartitionerCost(sizeEstimator, cs, numRows);
			default:
				throw new RuntimeException("Unsupported column group partitioner: " + type.toString());
		}
	}

}
