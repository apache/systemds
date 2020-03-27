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

import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import scala.Tuple2;

/**
 * Spark Disjoint_Round_Robin data partitioner:
 */
public class DRRSparkScheme extends DataPartitionSparkScheme {

	private static final long serialVersionUID = -3130831851505549672L;

	protected DRRSparkScheme() {
		// No-args constructor used for deserialization
	}

	@Override
	public Result doPartitioning(int numWorkers, int rblkID, MatrixBlock features, MatrixBlock labels) {
		List<Tuple2<Integer, Tuple2<Long, MatrixBlock>>> pfs = nonShuffledPartition(rblkID, features);
		List<Tuple2<Integer, Tuple2<Long, MatrixBlock>>> pls = nonShuffledPartition(rblkID, labels);
		return new Result(pfs, pls);
	}
}
