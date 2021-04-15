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

import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import java.util.List;

/**
 * Keep Data on Worker Federated scheme
 *
 * When the parameter server runs in federated mode it cannot pull in the data which is already on the workers.
 * All entries in the federation map of the input matrix are separated into MatrixObjects and returned as a list.
 * Only supports row federated matrices atm.
 */
public class KeepDataOnWorkerFederatedScheme extends DataPartitionFederatedScheme {
	@Override
	public Result partition(MatrixObject features, MatrixObject labels, int seed) {
		List<MatrixObject> pFeatures = sliceFederatedMatrix(features);
		List<MatrixObject> pLabels = sliceFederatedMatrix(labels);
		BalanceMetrics balanceMetrics = getBalanceMetrics(pFeatures);
		List<Double> weightingFactors = getWeightingFactors(pFeatures, balanceMetrics);
		return new Result(pFeatures, pLabels, pFeatures.size(), balanceMetrics, weightingFactors);
	}
}
