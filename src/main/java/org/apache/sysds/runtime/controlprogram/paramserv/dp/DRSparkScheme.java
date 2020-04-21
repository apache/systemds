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

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.controlprogram.paramserv.ParamservUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import scala.Tuple2;

/**
 * Spark data partitioner Disjoint_Random:
 *
 * For the current row block, find all the shifted place for each row (WorkerID {@literal =>} (row block ID, matrix)
 */
public class DRSparkScheme extends DataPartitionSparkScheme {

	private static final long serialVersionUID = -7655310624144544544L;

	protected DRSparkScheme() {
		// No-args constructor used for deserialization
	}

	@Override
	public Result doPartitioning(int numWorkers, int rblkID, MatrixBlock features, MatrixBlock labels) {
		List<Tuple2<Integer, Tuple2<Long, MatrixBlock>>> pfs = partition(rblkID, features);
		List<Tuple2<Integer, Tuple2<Long, MatrixBlock>>> pls = partition(rblkID, labels);
		return new Result(pfs, pls);
	}

	private List<Tuple2<Integer, Tuple2<Long, MatrixBlock>>> partition(int rblkID, MatrixBlock mb) {
		MatrixBlock partialPerm = _globalPerms.get(0).getBlock(rblkID, 1);

		// For each row, find out the shifted place
		return IntStream.range(0, mb.getNumRows()).mapToObj(r -> {
			MatrixBlock rowMB = ParamservUtils.sliceMatrixBlock(mb, r + 1, r + 1);
			long shiftedPosition = (long) partialPerm.getValue(r, 0);

			// Get the shifted block and position
			int shiftedBlkID = (int) (shiftedPosition / OptimizerUtils.DEFAULT_BLOCKSIZE + 1);

			MatrixBlock indicator = _workerIndicator.getBlock(shiftedBlkID, 1);
			int workerID = (int) indicator.getValue((int) shiftedPosition / OptimizerUtils.DEFAULT_BLOCKSIZE, 0);
			return new Tuple2<>(workerID, new Tuple2<>(shiftedPosition, rowMB));
		}).collect(Collectors.toList());
	}

}
