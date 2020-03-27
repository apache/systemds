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

import static org.apache.sysds.runtime.controlprogram.paramserv.ParamservUtils.SEED;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.sysds.parser.Statement;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.controlprogram.paramserv.ParamservUtils;
import org.apache.sysds.runtime.instructions.spark.data.PartitionedBroadcast;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;

public class SparkDataPartitioner implements Serializable {

	private static final long serialVersionUID = 6841548626711057448L;
	private DataPartitionSparkScheme _scheme;

	protected SparkDataPartitioner(Statement.PSScheme scheme, SparkExecutionContext sec, int numEntries, int numWorkers) {
		switch (scheme) {
			case DISJOINT_CONTIGUOUS:
				_scheme = new DCSparkScheme();
				// Create the worker id indicator
				createDCIndicator(sec, numWorkers, numEntries);
				break;
			case DISJOINT_ROUND_ROBIN:
				_scheme = new DRRSparkScheme();
				// Create the worker id indicator
				createDRIndicator(sec, numWorkers, numEntries);
				break;
			case DISJOINT_RANDOM:
				_scheme = new DRSparkScheme();
				// Create the global permutation
				createGlobalPermutations(sec, numEntries, 1);
				// Create the worker id indicator
				createDCIndicator(sec, numWorkers, numEntries);
				break;
			case OVERLAP_RESHUFFLE:
				_scheme = new ORSparkScheme();
				// Create the global permutation seperately for each worker
				createGlobalPermutations(sec, numEntries, numWorkers);
				break;
		}
	}

	private void createDRIndicator(SparkExecutionContext sec, int numWorkers, int numEntries) {
		double[] vector = IntStream.range(0, numEntries).mapToDouble(n -> n % numWorkers).toArray();
		MatrixBlock vectorMB = DataConverter.convertToMatrixBlock(vector, true);
		_scheme.setWorkerIndicator(sec.getBroadcastForMatrixObject(ParamservUtils.newMatrixObject(vectorMB)));
	}

	private void createDCIndicator(SparkExecutionContext sec, int numWorkers, int numEntries) {
		double[] vector = new double[numEntries];
		int batchSize = (int) Math.ceil((double) numEntries / numWorkers);
		for (int i = 1; i < numWorkers; i++) {
			int begin = batchSize * i;
			int end = Math.min(begin + batchSize, numEntries);
			Arrays.fill(vector, begin, end, i);
		}
		MatrixBlock vectorMB = DataConverter.convertToMatrixBlock(vector, true);
		_scheme.setWorkerIndicator(sec.getBroadcastForMatrixObject(ParamservUtils.newMatrixObject(vectorMB)));
	}

	private void createGlobalPermutations(SparkExecutionContext sec, int numEntries, int numPerm) {
		List<PartitionedBroadcast<MatrixBlock>> perms = IntStream.range(0, numPerm).mapToObj(i -> {
			MatrixBlock perm = MatrixBlock.sampleOperations(numEntries, numEntries, false, SEED+i);
			// Create the source-target id vector from the permutation ranging from 1 to number of entries
			double[] vector = new double[numEntries];
			for (int j = 0; j < perm.getDenseBlockValues().length; j++) {
				vector[(int) perm.getDenseBlockValues()[j] - 1] = j;
			}
			MatrixBlock vectorMB = DataConverter.convertToMatrixBlock(vector, true);
			return sec.getBroadcastForMatrixObject(ParamservUtils.newMatrixObject(vectorMB));
		}).collect(Collectors.toList());
		_scheme.setGlobalPermutation(perms);
	}

	public DataPartitionSparkScheme.Result doPartitioning(int numWorkers, MatrixBlock features, MatrixBlock labels,
			long rowID) {
		// Set the rowID in order to get the according permutation
		return _scheme.doPartitioning(numWorkers, (int) rowID, features, labels);
	}
}
