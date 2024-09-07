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

import org.apache.sysds.hops.OptimizerUtils;

public class RDDStats {
	enum CheckpointStatus {
		NO_CHECKPOINT,
		MEMORY_AND_DISK,
		MEMORY_AND_DISK_SER
	}

	public static final long hdfsBlockSize = 134217728; // 128MB
	public static final int blockSize = 1000;
	long n;
	long m;
	double sparsity;
	long distributedSize;
	long numBlocks;
	int numPartitions;
	boolean hashPartitioned;
	boolean checkpoint;
	double cost;
	boolean isCollected;
	// TODO: add boolean isCheckpointed and logic for it

//	public static void setDefaults() {
//		blockSize = ConfigurationManager.getBlocksize();
//		hdfsBlockSize = InfrastructureAnalyzer.getHDFSBlockSize();
//	}

//	public RDDStats(VarStats cpVar) {
//		totalSize = OptimizerUtils.estimateSizeExactSparsity(cpVar.getM(), cpVar.getN(), cpVar.getS());
//		numPartitions = (int) Math.max(Math.min(totalSize / hdfsBlockSize, cpVar.characteristics.getNumBlocks()), 1);
//		numBlocks = cpVar.characteristics.getNumBlocks();
//		this.cpVar = cpVar;
//		rlen = cpVar.getM();
//		clen = cpVar.getN();
//		numValues = rlen*clen;
//		sparsity = cpVar.getS();
//		numParallelTasks = (int) Math.min(numPartitions, SparkExecutionContext.getDefaultParallelism(false));
//	}

	public RDDStats(VarStats sourceStats) {
		// required cpVar initiated for not scalars
		if (sourceStats == null || sourceStats.isScalar()) {
			throw new RuntimeException("RDDStats cannot be initialized for scalar objects");
		}
		// cpVar carries all info about the general object characteristics
		n = sourceStats.getN();
		m = sourceStats.getM();
		sparsity = sourceStats.getSparsity();
		numBlocks = sourceStats.characteristics.getNumBlocks();
		checkpoint = false;
		// RDD specific characteristics not initialized -> simulates lazy evaluation
		distributedSize = -1;
		numPartitions = -1;
		hashPartitioned = false;
		cost = 0;
		isCollected = false;
	}

	public RDDStats(long nRows, long nCols, long nnz, long size, int partitions) {
		n = nCols;
		m = nRows;
		sparsity = OptimizerUtils.getSparsity(m, n, nnz);
		numBlocks = Math.max(((m + blockSize - 1) / blockSize), 1) *
				Math.max(((n + blockSize - 1) / blockSize), 1);
		checkpoint = false;
		// RDD specific characteristics not initialized -> simulates lazy evaluation
		distributedSize = size;
		if (partitions < 0) {
			numPartitions = getNumPartitions(hdfsBlockSize);
		} else {
			numPartitions = partitions;
		}
		cost = -1;
		isCollected = false;
	}

	/**
	 * Loads the RDD characteristics for binary data only
	 */
	public long loadCharacteristics() {
		distributedSize = OptimizerUtils.estimatePartitionedSizeExactSparsity(m, n, blockSize, sparsity);
		numPartitions = getNumPartitions(hdfsBlockSize);
		return distributedSize;
	}

	/**
	 * Loads the RDD characteristics for non-binary data by passing the data size
	 * @param size RDD size in memory
	 */
	public void loadCharacteristics(long size) {
		distributedSize = size;
		numPartitions = getNumPartitions(hdfsBlockSize);
	}

	/**
	 * Loads the RDD characteristics for the case the number of partitions is not defined by teh HDFS block size
	 * @param targetPartitions enforced numbed or partitions
	 */
	public void loadCharacteristics(int targetPartitions) {
		distributedSize = OptimizerUtils.estimatePartitionedSizeExactSparsity(m, n, blockSize, sparsity);
		numPartitions = targetPartitions;
	}

	private int getNumPartitions(long hdfsBlockSize) {
		return (int) Math.max((distributedSize + hdfsBlockSize - 1) / hdfsBlockSize, 1);
	}

	/**
	 * Meant to be used at testing
	 * @return estimated time (seconds) for generation of the current RDD
	 */
	public double getCost() {
		return cost;
	}

	//	public static RDDStats transformNumPartitions(RDDStats oldRDD, long newNumPartitions) {
//		if (oldRDD.cpVar == null) {
//			throw new DMLRuntimeException("Cannot transform RDDStats without VarStats");
//		}
//		RDDStats newRDD = new RDDStats(oldRDD.cpVar);
//		newRDD.numPartitions = newNumPartitions;
//		return newRDD;
//	}
//
//	public static RDDStats transformNumBlocks(RDDStats oldRDD, long newNumBlocks) {
//		if (oldRDD.cpVar == null) {
//			throw new DMLRuntimeException("Cannot transform RDDStats without VarStats");
//		}
//		RDDStats newRDD = new RDDStats(oldRDD.cpVar);
//		newRDD.numBlocks = newNumBlocks;
//		return newRDD;
//	}
}
