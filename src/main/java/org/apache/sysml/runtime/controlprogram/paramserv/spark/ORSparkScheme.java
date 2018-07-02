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

import org.apache.sysml.runtime.controlprogram.paramserv.ORLocalScheme;
import org.apache.sysml.runtime.controlprogram.paramserv.ParamservUtils;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;

/**
 * Data partitioner Overlap_Reshuffle:
 * for each worker, use a new permutation multiply P %*% X,
 * where P is constructed for example with P=table(seq(1,nrow(X),sample(nrow(X), nrow(X))))
 */
public class ORSparkScheme extends DataPartitionSparkScheme {

	private static final long serialVersionUID = 6867567406403580311L;

	protected ORSparkScheme() {
		// No-args constructor used for deserialization
	}

	@Override
	public Result doPartitioning(int workersNum, MatrixBlock features, MatrixBlock labels) {
		// Generate a different permutation matrix for each worker
		List<MatrixBlock> permutations = IntStream.range(0, workersNum).mapToObj(
				i -> ParamservUtils.generatePermutation((int) features.getNumRows())).collect(Collectors.toList());
		List<MatrixBlock> pfs = ORLocalScheme.partition(workersNum, features, permutations);
		List<MatrixBlock> pls = ORLocalScheme.partition(workersNum, labels, permutations);
		return new Result(pfs, pls);
	}
}
