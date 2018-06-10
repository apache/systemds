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

import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;

public abstract class NonDeterministicDataPartitioner extends DataPartitioner {
	@Override
	public void doPartitioning(List<LocalPSWorker> workers, MatrixObject features, MatrixObject labels) {
		// Generate the permutation
		long range = features.getNumRows();
		int size = (int) range;
		MatrixBlock permutation = ParamservUtils.generatePermutation(range, size);
		List<MatrixObject> pfs = doPartitioning(workers.size(), features, permutation);
		List<MatrixObject> pls = doPartitioning(workers.size(), labels, permutation);
		setPartitionedData(workers, pfs, pls);
	}

	/**
	 * Do matrix partitioning in non deterministic manner
	 * @param k           The size of workers
	 * @param mo          Matrix
	 * @param permutation Permutation matrix
	 * @return List of partitioned matrix
	 */
	public abstract List<MatrixObject> doPartitioning(int k, MatrixObject mo, MatrixBlock permutation);

}
