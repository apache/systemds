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
import org.apache.sysds.runtime.meta.DataCharacteristics;

import java.util.List;
import java.util.concurrent.Future;

/**
 * Replicate to Max Federated scheme
 *
 * When the parameter server runs in federated mode it cannot pull in the data which is already on the workers.
 * Therefore, a UDF is sent to manipulate the data locally. In this case the global maximum number of examples is taken
 * and the worker replicates data to match that number of examples. The generation is done by multiplying with a
 * Permutation Matrix with a global seed. These selected examples are appended to the original data.
 *
 * Then all entries in the federation map of the input matrix are separated into MatrixObjects and returned as a list.
 * Only supports row federated matrices atm.
 */
public class ReplicateToMaxFederatedScheme extends DataPartitionFederatedScheme {
	@Override
	public Result partition(MatrixObject features, MatrixObject labels, int seed) {
		List<MatrixObject> pFeatures = sliceFederatedMatrix(features);
		List<MatrixObject> pLabels = sliceFederatedMatrix(labels);
		List<Double> weightingFactors = getWeightingFactors(pFeatures, getBalanceMetrics(pFeatures));

		int max_rows = 0;
		for (MatrixObject pFeature : pFeatures) {
			max_rows = (pFeature.getNumRows() > max_rows) ? Math.toIntExact(pFeature.getNumRows()) : max_rows;
		}

		for(int i = 0; i < pFeatures.size(); i++) {
			// Works, because the map contains a single entry
			FederatedData featuresData = pFeatures.get(i).getFedMapping().getFederatedData()[0];
			FederatedData labelsData = pLabels.get(i).getFedMapping().getFederatedData()[0];

			Future<FederatedResponse> udfResponse = featuresData.executeFederatedOperation(new FederatedRequest(FederatedRequest.RequestType.EXEC_UDF,
					featuresData.getVarID(), new replicateDataOnFederatedWorker(new long[]{featuresData.getVarID(), labelsData.getVarID()}, seed, max_rows)));

			try {
				FederatedResponse response = udfResponse.get();
				if(!response.isSuccessful())
					throw new DMLRuntimeException("FederatedDataPartitioner ReplicateFederatedScheme: replicate UDF returned fail");
			}
			catch(Exception e) {
				throw new DMLRuntimeException("FederatedDataPartitioner ReplicateFederatedScheme: executing replicate UDF failed" + e.getMessage());
			}

			DataCharacteristics update = pFeatures.get(i).getDataCharacteristics().setRows(max_rows);
			pFeatures.get(i).updateDataCharacteristics(update);
			update = pLabels.get(i).getDataCharacteristics().setRows(max_rows);
			pLabels.get(i).updateDataCharacteristics(update);
		}

		return new Result(pFeatures, pLabels, pFeatures.size(), getBalanceMetrics(pFeatures), weightingFactors);
	}

	/**
	 * Replicate UDF executed on the federated worker
	 */
	private static class replicateDataOnFederatedWorker extends FederatedUDF {
		private static final long serialVersionUID = -6930898456315100587L;
		private final int _seed;
		private final int _max_rows;

		protected replicateDataOnFederatedWorker(long[] inIDs, int seed, int max_rows) {
			super(inIDs);
			_seed = seed;
			_max_rows = max_rows;
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			MatrixObject features = (MatrixObject) data[0];
			MatrixObject labels = (MatrixObject) data[1];

			// replicate up to the max
			if(features.getNumRows() < _max_rows) {
				int num_rows_needed = _max_rows - Math.toIntExact(features.getNumRows());
				// generate replication matrix
				MatrixBlock replicateMatrixBlock = ParamservUtils.generateReplicationMatrix(num_rows_needed, Math.toIntExact(features.getNumRows()), _seed);
				replicateTo(features, replicateMatrixBlock);
				replicateTo(labels, replicateMatrixBlock);
			}

			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS);
		}

		@Override
		public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
			return null;
		}
	}
}
