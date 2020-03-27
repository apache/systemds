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

package org.apache.sysds.runtime.controlprogram.paramserv.dp;

import java.io.Serializable;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.LongStream;

import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.controlprogram.paramserv.ParamservUtils;
import org.apache.sysds.runtime.instructions.spark.data.PartitionedBroadcast;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import scala.Tuple2;

public abstract class DataPartitionSparkScheme implements Serializable {

	protected final class Result {
		protected final List<Tuple2<Integer, Tuple2<Long, MatrixBlock>>> pFeatures; // WorkerID => (rowID, matrix)
		protected final List<Tuple2<Integer, Tuple2<Long, MatrixBlock>>> pLabels;

		protected Result(List<Tuple2<Integer, Tuple2<Long, MatrixBlock>>> pFeatures, List<Tuple2<Integer, Tuple2<Long, MatrixBlock>>> pLabels) {
			this.pFeatures = pFeatures;
			this.pLabels = pLabels;
		}
	}

	private static final long serialVersionUID = -3462829818083371171L;

	protected List<PartitionedBroadcast<MatrixBlock>> _globalPerms; // a list of global permutations
	protected PartitionedBroadcast<MatrixBlock> _workerIndicator; // a matrix indicating to which worker the given row belongs

	protected void setGlobalPermutation(List<PartitionedBroadcast<MatrixBlock>> gps) {
		_globalPerms = gps;
	}

	protected void setWorkerIndicator(PartitionedBroadcast<MatrixBlock> wi) {
		_workerIndicator = wi;
	}

	/**
	 * Do non-reshuffled data partitioning according to worker indicator
	 * @param rblkID row block ID
	 * @param mb Matrix
	 * @return list of tuple (workerID, (row block ID, matrix row))
	 */
	protected List<Tuple2<Integer, Tuple2<Long, MatrixBlock>>> nonShuffledPartition(int rblkID, MatrixBlock mb) {
		MatrixBlock indicator = _workerIndicator.getBlock(rblkID, 1);
		return LongStream.range(0, mb.getNumRows()).mapToObj(r -> {
			int workerID = (int) indicator.getValue((int) r, 0);
			MatrixBlock rowMB = ParamservUtils.sliceMatrixBlock(mb, r + 1, r + 1);
			long shiftedPosition = r + (rblkID - 1) * OptimizerUtils.DEFAULT_BLOCKSIZE;
			return new Tuple2<>(workerID, new Tuple2<>(shiftedPosition, rowMB));
		}).collect(Collectors.toList());
	}

	public abstract Result doPartitioning(int numWorkers, int rblkID, MatrixBlock features, MatrixBlock labels);
}
