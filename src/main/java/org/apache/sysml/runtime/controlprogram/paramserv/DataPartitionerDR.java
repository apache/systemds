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

package org.apache.sysml.runtime.controlprogram.paramserv;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.operators.AggregateBinaryOperator;

/**
 * Data partitioner Disjoint_Random:
 * for each worker, use a permutation multiply P[beg:end,] %*% X,
 * where P is constructed for example with P=table(seq(1,nrow(X)),sample(nrow(X), nrow(X))),
 * i.e., sampling without replacement to ensure disjointness.
 */
public class DataPartitionerDR extends NonDeterministicDataPartitioner {

	@Override
	public List<MatrixObject> doPartitioning(int k, MatrixObject mo, MatrixBlock permutation) {
		MatrixBlock data = mo.acquireRead();
		int batchSize = (int) Math.ceil(mo.getNumRows() / k);
		List<MatrixObject> pMatrices = IntStream.range(0, k).mapToObj(i -> {
			int begin = i * batchSize + 1;
			int end = (int) Math.min((i + 1) * batchSize, mo.getNumRows());
			MatrixBlock slicedPerm = permutation.slice(begin - 1, end - 1, 0, permutation.getNumColumns() - 1, new MatrixBlock());
			AggregateBinaryOperator op = InstructionUtils.getMatMultOperator(k);
			MatrixBlock output = slicedPerm.aggregateBinaryOperations(slicedPerm, data, new MatrixBlock(), op);
			MatrixObject result = ParamservUtils.newMatrixObject();
			result.acquireModify(output);
			result.release();
			return result;
		}).collect(Collectors.toList());
		mo.release();
		return pMatrices;
	}
}
