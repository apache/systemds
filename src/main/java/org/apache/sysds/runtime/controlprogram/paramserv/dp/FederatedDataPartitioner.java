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

import org.apache.sysds.parser.Statement;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;

public class FederatedDataPartitioner {
	private final DataPartitionFederatedScheme _scheme;
	private final int _seed;

	public FederatedDataPartitioner(Statement.FederatedPSScheme scheme, int seed) {
		_seed = seed;
		switch (scheme) {
			case KEEP_DATA_ON_WORKER:
				_scheme = new KeepDataOnWorkerFederatedScheme();
				break;
			case SHUFFLE:
				_scheme = new ShuffleFederatedScheme();
				break;
			case REPLICATE_TO_MAX:
				_scheme = new ReplicateToMaxFederatedScheme();
				break;
			case SUBSAMPLE_TO_MIN:
				_scheme = new SubsampleToMinFederatedScheme();
				break;
			case BALANCE_TO_AVG:
				_scheme = new BalanceToAvgFederatedScheme();
				break;
			default:
				throw new DMLRuntimeException(String.format("FederatedDataPartitioner: not support data partition scheme '%s'", scheme));
		}
	}

	public DataPartitionFederatedScheme.Result doPartitioning(MatrixObject features, MatrixObject labels) {
		return _scheme.partition(features, labels, _seed);
	}
}
