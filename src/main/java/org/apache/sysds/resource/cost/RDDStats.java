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
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;

public class RDDStats {
	private static final long hdfsBlockSize = 134217728; // 128MB
	@SuppressWarnings("unused")
	private static final int blockSize = 1000; // TODO: think of more efficient way (does not changes, init once)

	VarStats cpStats;
	long distributedMemory;
	long numBlocks;
	long numPartitions;
	int numParallelTasks;
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
		// RDD specific characteristics not initialized -> simulates lazy evaluation
		distributedMemory = -1;
		numPartitions = -1;
		numBlocks = -1;
		numParallelTasks = -1;
		// cpVar carries all info about the general object characteristics
		cpStats = sourceStats;
	}

	public void loadCharacteristics() {
		distributedMemory = OptimizerUtils.estimateSizeExactSparsity(cpStats.getM(), cpStats.getN(), cpStats.getS());
		numBlocks = cpStats.characteristics.getNumBlocks();
		numPartitions = (int) Math.max(Math.min(distributedMemory / hdfsBlockSize, numBlocks), 1);
		numParallelTasks = (int) Math.min(numPartitions, SparkExecutionContext.getDefaultParallelism(false));
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
