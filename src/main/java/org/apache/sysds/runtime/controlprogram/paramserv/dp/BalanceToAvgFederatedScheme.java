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

import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedData;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederatedUDF;
import org.apache.sysds.runtime.controlprogram.paramserv.ParamservUtils;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.DataCharacteristics;

import java.util.List;
import java.util.concurrent.Future;

public class BalanceToAvgFederatedScheme extends DataPartitionFederatedScheme {
	@Override
	public Result doPartitioning(MatrixObject features, MatrixObject labels) {
		List<MatrixObject> pFeatures = sliceFederatedMatrix(features);
		List<MatrixObject> pLabels = sliceFederatedMatrix(labels);

		int average_num_rows = (int) Math.round(pFeatures.stream().map(CacheableData::getNumRows).mapToInt(Long::intValue).average().orElse(Double.NaN));

		for(int i = 0; i < pFeatures.size(); i++) {
			// Works, because the map contains a single entry
			FederatedData featuresData = (FederatedData) pFeatures.get(i).getFedMapping().getMap().values().toArray()[0];
			FederatedData labelsData = (FederatedData) pLabels.get(i).getFedMapping().getMap().values().toArray()[0];

			Future<FederatedResponse> udfResponse = featuresData.executeFederatedOperation(new FederatedRequest(FederatedRequest.RequestType.EXEC_UDF,
					featuresData.getVarID(), new balanceDataOnFederatedWorker(new long[]{featuresData.getVarID(), labelsData.getVarID()}, average_num_rows)));

			try {
				FederatedResponse response = udfResponse.get();
				if(!response.isSuccessful())
					throw new DMLRuntimeException("FederatedDataPartitioner BalanceFederatedScheme: balance UDF returned fail");
			}
			catch(Exception e) {
				throw new DMLRuntimeException("FederatedDataPartitioner BalanceFederatedScheme: executing balance UDF failed" + e.getMessage());
			}

			DataCharacteristics update = pFeatures.get(i).getDataCharacteristics().setRows(average_num_rows);
			pFeatures.get(i).updateDataCharacteristics(update);
			update = pLabels.get(i).getDataCharacteristics().setRows(average_num_rows);
			pLabels.get(i).updateDataCharacteristics(update);
		}

		return new Result(pFeatures, pLabels, pFeatures.size(), getBalanceMetrics(pFeatures));
	}

	/**
	 * Balance UDF executed on the federated worker
	 */
	private static class balanceDataOnFederatedWorker extends FederatedUDF {
		private static final long serialVersionUID = 6631958250346625546L;
		private final int _average_num_rows;
		
		protected balanceDataOnFederatedWorker(long[] inIDs, int average_num_rows) {
			super(inIDs);
			_average_num_rows = average_num_rows;
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			MatrixObject features = (MatrixObject) data[0];
			MatrixObject labels = (MatrixObject) data[1];

			if(features.getNumRows() > _average_num_rows) {
				// generate subsampling matrix
				MatrixBlock subsampleMatrixBlock = ParamservUtils.generateSubsampleMatrix(_average_num_rows, Math.toIntExact(features.getNumRows()), System.currentTimeMillis());
				subsampleTo(features, subsampleMatrixBlock);
				subsampleTo(labels, subsampleMatrixBlock);
			}
			else if(features.getNumRows() < _average_num_rows) {
				int num_rows_needed = _average_num_rows - Math.toIntExact(features.getNumRows());
				// generate replication matrix
				MatrixBlock replicateMatrixBlock = ParamservUtils.generateReplicationMatrix(num_rows_needed, Math.toIntExact(features.getNumRows()), System.currentTimeMillis());
				replicateTo(features, replicateMatrixBlock);
				replicateTo(labels, replicateMatrixBlock);
			}

			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS);
		}
	}
}
