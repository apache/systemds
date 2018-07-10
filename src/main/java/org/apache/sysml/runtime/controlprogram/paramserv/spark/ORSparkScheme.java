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

package org.apache.sysml.runtime.controlprogram.paramserv.spark;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.sysml.runtime.controlprogram.paramserv.DataPartitionScheme;
import org.apache.sysml.runtime.controlprogram.paramserv.ParamservUtils;
import org.apache.sysml.runtime.instructions.spark.data.PartitionedBroadcast;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;

/**
 * Spark data partitioner Overlap_Reshuffle:
 * for each worker, reshuffle the row block using the according partial permutations,
 * and then return the a list of reshuffled block of worker size
 */
public class ORSparkScheme extends DataPartitionScheme {

	private static final long serialVersionUID = 6867567406403580311L;

	protected ORSparkScheme() {
		// No-args constructor used for deserialization
	}

	@Override
	public Result doPartitioning(int numWorkers, MatrixBlock features, MatrixBlock labels) {
		List<MatrixBlock> pfs = partition(numWorkers, features);
		List<MatrixBlock> pls = partition(numWorkers, labels);
		return new Result(ParamservUtils.convertToMatrixObject(pfs), ParamservUtils.convertToMatrixObject(pls));
	}

	private List<MatrixBlock> partition(int numWorkers, MatrixBlock mb) {
		return IntStream.range(0, numWorkers).mapToObj(w -> {
			MatrixBlock newMB = null;
			PartitionedBroadcast<MatrixBlock> globalPerm = _globalPerms.get(w);

			// For each column, calculate the shifted place to this row block
			for (int i = 1; i < globalPerm.getNumRowBlocks() + 1; i++) {
				MatrixBlock partialPerm = globalPerm.getBlock(i, _rblkID);
				MatrixBlock reshuffledMB = DRSparkScheme.doShuffling(mb, partialPerm);
				if (newMB == null) {
					newMB = reshuffledMB;
				} else {
					newMB = ParamservUtils.rbindMatrix(newMB, reshuffledMB);
					reshuffledMB.cleanupBlock(true, false);
				}
			}
			return newMB;
		}).collect(Collectors.toList());
	}
}
