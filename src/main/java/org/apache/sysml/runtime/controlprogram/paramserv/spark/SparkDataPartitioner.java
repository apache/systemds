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

import static org.apache.sysml.runtime.controlprogram.paramserv.DataPartitionScheme.SEED;

import java.io.Serializable;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.sysml.parser.Statement;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.controlprogram.paramserv.DataPartitionScheme;
import org.apache.sysml.runtime.controlprogram.paramserv.ParamservUtils;
import org.apache.sysml.runtime.instructions.spark.data.PartitionedBroadcast;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;

public class SparkDataPartitioner implements Serializable {

	private static final long serialVersionUID = 6841548626711057448L;
	private DataPartitionScheme _scheme;

	protected SparkDataPartitioner(Statement.PSScheme scheme, SparkExecutionContext sec, int numEntries, int numWorkers) {
		switch (scheme) {
			case DISJOINT_CONTIGUOUS:
				_scheme = new DCSparkScheme();
				break;
			case DISJOINT_ROUND_ROBIN:
				_scheme = new DRRSparkScheme();
				break;
			case DISJOINT_RANDOM:
				_scheme = new DRSparkScheme();
				// Create the global permutation
				createGlobalPermutations(sec, numEntries, 1);
				break;
			case OVERLAP_RESHUFFLE:
				_scheme = new ORSparkScheme();
				// Create the global permutation seperately for each worker
				createGlobalPermutations(sec, numEntries, numWorkers);
				break;
		}
	}

	private void createGlobalPermutations(SparkExecutionContext sec, int numEntries, int numPerm) {
		List<PartitionedBroadcast<MatrixBlock>> perms = IntStream.range(0, numPerm).mapToObj(i -> {
			MatrixBlock perm = ParamservUtils.generatePermutation(numEntries, SEED);
			return sec.getBroadcastForMatrixObject(ParamservUtils.newMatrixObject(perm));
		}).collect(Collectors.toList());
		_scheme.setGlobalPermutation(perms);
	}

	public DataPartitionScheme.Result doPartitioning(int workersNum, MatrixBlock features, MatrixBlock labels, long rowID) {
		// Set the rowID in order to get the according permutation
		_scheme.setRowID((int) rowID);
		return _scheme.doPartitioning(workersNum, features, labels);
	}
}
