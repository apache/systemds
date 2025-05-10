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

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.paramserv.ParamservUtils;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

/**
 * Data partitioner Overlap_Reshuffle:
 * for each worker, use a new permutation multiply P %*% X,
 * where P is constructed for example with P=table(seq(1,nrow(X),sample(nrow(X), nrow(X))))
 */
public class ORLocalScheme extends DataPartitionLocalScheme {

	public static List<MatrixBlock> partition(int k, MatrixBlock mb, List<MatrixBlock> permutations) {
		return IntStream.range(0, k).mapToObj(i -> {
			MatrixBlock permutation = permutations.get(i);
			return permutation.aggregateBinaryOperations(permutation, mb, new MatrixBlock(),
					InstructionUtils.getMatMultOperator(k));
		}).collect(Collectors.toList());
	}

	private static List<MatrixObject> doPartitioning(int k, MatrixBlock mb, List<MatrixBlock> permutations) {
		return partition(k, mb, permutations).stream()
			.map(m -> ParamservUtils.newMatrixObject(m, false)).collect(Collectors.toList());
	}

	@Override
	public Result doPartitioning(int workersNum, MatrixBlock features, MatrixBlock labels) {
		// Generate a different permutation matrix for each worker
		List<MatrixBlock> permutations = IntStream.range(0, workersNum)
			.mapToObj(i -> ParamservUtils.generatePermutation(features.getNumRows(), SEED+i))
			.collect(Collectors.toList());
		List<MatrixObject> pfs = doPartitioning(workersNum, features, permutations);
		List<MatrixObject> pls = doPartitioning(workersNum, labels, permutations);
		return new Result(pfs, pls);
	}
}
