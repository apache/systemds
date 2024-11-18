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

package org.apache.sysds.resource.cost;

import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;


public class RDDStats {
	long distributedSize;
	int numPartitions;
	boolean hashPartitioned;
	// NOTE: checkpointing not fully considered by the cost estimator currently
	boolean checkpoint;
	double cost;
	boolean isCollected;

	/**
	 * Initiates RDD statistics object bound
	 * to an existing {@code VarStats} object.
	 * Uses HDFS block size to adjust automatically the
	 * number of partitions for the current RDD.
	 *
	 * @param sourceStats bound variables statistics
	 */
	public RDDStats(VarStats sourceStats) {
		// required cpVar initiated for not scalars
		if (sourceStats == null) {
			throw new RuntimeException("RDDStats cannot be initialized without valid input variable statistics");
		}
		checkpoint = false;
		isCollected = false;
		hashPartitioned = false;
		// RDD specific characteristics not initialized -> simulates lazy evaluation
		distributedSize = estimateDistributedSize(sourceStats);
		numPartitions = getNumPartitions();
		cost = 0;
	}

	/**
	 * Initiates RDD statistics object for
	 * intermediate variables (not bound to {@code VarStats}).
	 * Intended to be used for intermediate shuffle estimations.
	 *
	 * @param size distributed size of the object
	 * @param partitions target number of partitions;
	 *                      -1 for fitting to HDFS block size
	 */
	public RDDStats(long size, int partitions) {
		checkpoint = false;
		isCollected = false;
		hashPartitioned = false;
		// RDD specific characteristics not initialized -> simulates lazy evaluation
		distributedSize = size;
		if (partitions < 0) {
			numPartitions = getNumPartitions();
		} else {
			numPartitions = partitions;
		}
		cost = -1;
	}

	private int getNumPartitions() {
		if (distributedSize < 0) {
			throw new RuntimeException("Estimating number of partitions requires valid distributed RDD size");
		} else if (distributedSize > 0) {
			long hdfsBlockSize = InfrastructureAnalyzer.getHDFSBlockSize();
			return (int) Math.max((distributedSize + hdfsBlockSize - 1) / hdfsBlockSize, 1);
		}
		return -1; // for scalars
	}

	/**
	 * Meant to be used at testing
	 * @return estimated time (seconds) for generation of the current RDD
	 */
	public double getCost() {
		return cost;
	}

	/**
	 * Meant to be used at testing
	 * @return flag if the current RDD is collected
	 */
	public boolean isCollected() {
		return isCollected;
	}

	private static long estimateDistributedSize(VarStats sourceStats) {
		if (sourceStats.isScalar())
			return 0; // 0 so it is non-negative
		if (sourceStats.getCells() < 0)
			throw new RuntimeException("Estimated size for RDD object is negative");
		return OptimizerUtils.estimatePartitionedSizeExactSparsity(
				sourceStats.getM(),
				sourceStats.getN(),
				ConfigurationManager.getBlocksize(),
				sourceStats.getSparsity()
		);
	}
}
