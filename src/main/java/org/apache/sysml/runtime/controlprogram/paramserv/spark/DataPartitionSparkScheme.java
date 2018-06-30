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

import java.io.Serializable;
import java.util.List;

import org.apache.sysml.runtime.matrix.data.MatrixBlock;

public abstract class DataPartitionSparkScheme implements Serializable {

	private static final long serialVersionUID = 6302587386967681329L;

	final class Result {
		public final List<MatrixBlock> pFeatures;
		public final List<MatrixBlock> pLabels;

		public Result(List<MatrixBlock> pFeatures, List<MatrixBlock> pLabels) {
			this.pFeatures = pFeatures;
			this.pLabels = pLabels;
		}
	}

	protected abstract Result doPartitioning(int workersNum, MatrixBlock features, MatrixBlock labels);
}
