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

import org.apache.commons.lang3.tuple.Pair;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedData;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederatedUDF;
import org.apache.sysds.runtime.controlprogram.paramserv.ParamservUtils;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;

import java.util.List;
import java.util.concurrent.Future;

/**
 * Shuffle Federated scheme
 *
 * When the parameter server runs in federated mode it cannot pull in the data which is already on the workers.
 * Therefore, a UDF is sent to manipulate the data locally. In this case it is shuffled by generating a permutation
 * matrix with a global seed and doing a mat mult.
 *
 * Then all entries in the federation map of the input matrix are separated into MatrixObjects and returned as a list.
 * Only supports row federated matrices atm.
 */
public class ShuffleFederatedScheme extends DataPartitionFederatedScheme {
	@Override
	public Result partition(MatrixObject features, MatrixObject labels, int seed) {
		List<MatrixObject> pFeatures = sliceFederatedMatrix(features);
		List<MatrixObject> pLabels = sliceFederatedMatrix(labels);
		BalanceMetrics balanceMetrics = getBalanceMetrics(pFeatures);
		List<Double> weightingFactors = getWeightingFactors(pFeatures, balanceMetrics);

		for(int i = 0; i < pFeatures.size(); i++) {
			// Works, because the map contains a single entry
			FederatedData featuresData = pFeatures.get(i).getFedMapping().getFederatedData()[0];
			FederatedData labelsData = pLabels.get(i).getFedMapping().getFederatedData()[0];

			Future<FederatedResponse> udfResponse = featuresData.executeFederatedOperation(new FederatedRequest(FederatedRequest.RequestType.EXEC_UDF,
					featuresData.getVarID(), new shuffleDataOnFederatedWorker(new long[]{featuresData.getVarID(), labelsData.getVarID()}, seed)));

			try {
				FederatedResponse response = udfResponse.get();
				if(!response.isSuccessful())
					throw new DMLRuntimeException("FederatedDataPartitioner ShuffleFederatedScheme: shuffle UDF returned fail");
			}
			catch(Exception e) {
				throw new DMLRuntimeException("FederatedDataPartitioner ShuffleFederatedScheme: executing shuffle UDF failed" + e.getMessage());
			}
		}

		return new Result(pFeatures, pLabels, pFeatures.size(), balanceMetrics, weightingFactors);
	}

	/**
	 * Shuffle UDF executed on the federated worker
	 */
	private static class shuffleDataOnFederatedWorker extends FederatedUDF {
		private static final long serialVersionUID = 3228664618781333325L;
		private final int _seed;

		protected shuffleDataOnFederatedWorker(long[] inIDs, int seed) {
			super(inIDs);
			_seed = seed;
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			MatrixObject features = (MatrixObject) data[0];
			MatrixObject labels = (MatrixObject) data[1];

			// generate permutation matrix
			MatrixBlock permutationMatrixBlock = ParamservUtils.generatePermutation(Math.toIntExact(features.getNumRows()), _seed);
			shuffle(features, permutationMatrixBlock);
			shuffle(labels, permutationMatrixBlock);
			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS);
		}

		@Override
		public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
			return null;
		}
	}
}