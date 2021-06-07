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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.compress.CompressionSettings;
import org.apache.sysds.runtime.compress.estim.CompressedSizeEstimator;
import org.apache.sysds.runtime.compress.estim.CompressedSizeInfo;

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

		// Use column group partitioner to create partitions of columns
		CompressedSizeInfo bins = createColumnGroupPartitioner(cs.columnPartitioner, est, cs, numRows)
			.coCodeColumns(colInfos, k);

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
}
