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

import java.util.Iterator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.sysml.runtime.controlprogram.paramserv.DataPartitionScheme;
import org.apache.sysml.runtime.controlprogram.paramserv.ParamservUtils;
import org.apache.sysml.runtime.instructions.spark.data.PartitionedBroadcast;
import org.apache.sysml.runtime.matrix.data.IJV;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.util.DataConverter;

/**
 * Spark data partitioner Disjoint_Random:
 *
 * For each worker, use a partial permutation P which aims to shift the current block to target block,
 * because according to the global permutation, the current row block will shifted to other row block,
 * where P is fetched from global permutation with (cblkID, current rblkID),
 * and then reshuffle the current row block to generate a complete matrix (i.e., same row size with complete one).
 * Finally, slice the reshuffled complete matrix in disjoint way.
 */
public class DRSparkScheme extends DataPartitionScheme {

	private static final long serialVersionUID = -7655310624144544544L;

	protected DRSparkScheme() {
		// No-args constructor used for deserialization
	}

	@Override
	public Result doPartitioning(int workersNum, MatrixBlock features, MatrixBlock labels) {
		List<MatrixBlock> pfs = partition(workersNum, features);
		List<MatrixBlock> pls = partition(workersNum, labels);
		return new Result(ParamservUtils.convertToMatrixObject(pfs), ParamservUtils.convertToMatrixObject(pls));
	}

	private List<MatrixBlock> partition(int k, MatrixBlock mb) {
		MatrixBlock newMB = null;
		PartitionedBroadcast<MatrixBlock> globalPerm = _globalPerms.get(0);

		// For each column, calculate the shifted place to this row block
		for (int i = 1; i < globalPerm.getNumRowBlocks() + 1; i++) {
			MatrixBlock partialPerm = globalPerm.getBlock(i, _rblkID);
			MatrixBlock reshuffledMB = doShuffling(mb, partialPerm);

			if (newMB == null) {
				newMB = reshuffledMB;
			} else {
				newMB = ParamservUtils.rbindMatrix(newMB, reshuffledMB);
				reshuffledMB.cleanupBlock(true, false);
			}
		}

		// Do data partition
		int batchSize = (int) Math.ceil((double) newMB.getNumRows() / k);
		MatrixBlock rowPerm = newMB;

		return IntStream.range(0, k).mapToObj(i -> {
			int begin = i * batchSize;
			int end = Math.min((i + 1) * batchSize, rowPerm.getNumRows());
			return rowPerm.slice(begin, end - 1);
		}).collect(Collectors.toList());
	}

	protected static MatrixBlock doShuffling(MatrixBlock mb, MatrixBlock perm) {
		double[] data = new double[mb.getNumRows()];
		Iterator<IJV> iter = perm.getSparseBlockIterator();
		while (iter.hasNext()) {
			data[iter.next().getJ()] = 1.0;
		}
		MatrixBlock select = DataConverter.convertToMatrixBlock(data, true);
		return mb.removeEmptyOperations(new MatrixBlock(), true, true, select);
	}

}
