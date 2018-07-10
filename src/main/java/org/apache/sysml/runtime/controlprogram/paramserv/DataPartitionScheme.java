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

import java.io.Serializable;
import java.util.List;

import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.instructions.spark.data.PartitionedBroadcast;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;

public abstract class DataPartitionScheme implements Serializable {

	public final class Result {
		public final List<MatrixObject> pFeatures;
		public final List<MatrixObject> pLabels;

		public Result(List<MatrixObject> pFeatures, List<MatrixObject> pLabels) {
			this.pFeatures = pFeatures;
			this.pLabels = pLabels;
		}
	}

	private static final long serialVersionUID = -3462829818083371171L;
	public static long SEED = -1; // Used for generating permutation

	protected List<PartitionedBroadcast<MatrixBlock>> _globalPerms; // a list of global permutations
	protected int _rblkID; // Current row block id

	public void setGlobalPermutation(List<PartitionedBroadcast<MatrixBlock>> gps) {
		_globalPerms = gps;
	}

	public void setRowID(int rowID) {
		_rblkID = rowID;
	}

	public abstract Result doPartitioning(int workersNum, MatrixBlock features, MatrixBlock labels);
}
