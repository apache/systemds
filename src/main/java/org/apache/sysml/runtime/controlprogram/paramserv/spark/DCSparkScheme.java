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

import org.apache.sysml.runtime.controlprogram.paramserv.DCLocalScheme;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;

/**
 * Disjoint_Contiguous data partitioner:
 * <p>
 * for each worker, use a right indexing
 * operation X[beg:end,] to obtain contiguous,
 * non-overlapping partitions of rows.
 */
public class DCSparkScheme extends DataPartitionSparkScheme {

	private static final long serialVersionUID = -2786906947020788787L;

	protected DCSparkScheme() {
		// No-args constructor used for deserialization
	}

	@Override
	public Result doPartitioning(int workersNum, MatrixBlock features, MatrixBlock labels) {
		List<MatrixBlock> pfs = DCLocalScheme.partition(workersNum, features);
		List<MatrixBlock> pls = DCLocalScheme.partition(workersNum, labels);
		return new Result(pfs, pls);
	}
}
