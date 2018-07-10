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

import org.apache.sysml.runtime.controlprogram.paramserv.DRRScheme;
import org.apache.sysml.runtime.controlprogram.paramserv.DataPartitionScheme;
import org.apache.sysml.runtime.controlprogram.paramserv.ParamservUtils;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;

/**
 * Disjoint_Round_Robin data partitioner:
 * for each worker, use a permutation multiply
 * or simpler a removeEmpty such as removeEmpty
 * (target=X, margin=rows, select=(seq(1,nrow(X))%%k)==id)
 */
public class DRRSparkScheme extends DataPartitionScheme {

	private static final long serialVersionUID = -3130831851505549672L;

	protected DRRSparkScheme() {
		// No-args constructor used for deserialization
	}

	@Override
	public Result doPartitioning(int workersNum, MatrixBlock features, MatrixBlock labels) {
		List<MatrixBlock> pfs = IntStream.range(0, workersNum).mapToObj(i -> DRRScheme.removeEmpty(features, workersNum, i)).collect(Collectors.toList());
		List<MatrixBlock> pls = IntStream.range(0, workersNum).mapToObj(i -> DRRScheme.removeEmpty(labels, workersNum, i)).collect(Collectors.toList());
		return new Result(ParamservUtils.convertToMatrixObject(pfs), ParamservUtils.convertToMatrixObject(pls));
	}
}
