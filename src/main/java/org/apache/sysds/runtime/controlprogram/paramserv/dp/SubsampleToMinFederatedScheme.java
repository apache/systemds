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

public class SubsampleToMinFederatedScheme extends DataPartitionFederatedScheme {
	@Override
	public Result doPartitioning(MatrixObject features, MatrixObject labels) {
		List<MatrixObject> pFeatures = sliceFederatedMatrix(features);
		List<MatrixObject> pLabels = sliceFederatedMatrix(labels);
		List<Double> scalingFactors = getScalingFactors(pFeatures, getBalanceMetrics(pFeatures));

		int min_rows = Integer.MAX_VALUE;
		for (MatrixObject pFeature : pFeatures) {
			min_rows = (pFeature.getNumRows() < min_rows) ? Math.toIntExact(pFeature.getNumRows()) : min_rows;
		}

		for(int i = 0; i < pFeatures.size(); i++) {
			// Works, because the map contains a single entry
			FederatedData featuresData = (FederatedData) pFeatures.get(i).getFedMapping().getMap().values().toArray()[0];
			FederatedData labelsData = (FederatedData) pLabels.get(i).getFedMapping().getMap().values().toArray()[0];

			Future<FederatedResponse> udfResponse = featuresData.executeFederatedOperation(new FederatedRequest(FederatedRequest.RequestType.EXEC_UDF,
					featuresData.getVarID(), new subsampleDataOnFederatedWorker(new long[]{featuresData.getVarID(), labelsData.getVarID()}, min_rows)));

			try {
				FederatedResponse response = udfResponse.get();
				if(!response.isSuccessful())
					throw new DMLRuntimeException("FederatedDataPartitioner SubsampleFederatedScheme: subsample UDF returned fail");
			}
			catch(Exception e) {
				throw new DMLRuntimeException("FederatedDataPartitioner SubsampleFederatedScheme: executing subsample UDF failed" + e.getMessage());
			}

			DataCharacteristics update = pFeatures.get(i).getDataCharacteristics().setRows(min_rows);
			pFeatures.get(i).updateDataCharacteristics(update);
			update = pLabels.get(i).getDataCharacteristics().setRows(min_rows);
			pLabels.get(i).updateDataCharacteristics(update);
		}

		return new Result(pFeatures, pLabels, pFeatures.size(), getBalanceMetrics(pFeatures), scalingFactors);
	}

	/**
	 * Subsample UDF executed on the federated worker
	 */
	private static class subsampleDataOnFederatedWorker extends FederatedUDF {
		private static final long serialVersionUID = 2213790859544004286L;
		private final int _min_rows;
		
		protected subsampleDataOnFederatedWorker(long[] inIDs, int min_rows) {
			super(inIDs);
			_min_rows = min_rows;
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			MatrixObject features = (MatrixObject) data[0];
			MatrixObject labels = (MatrixObject) data[1];

			// subsample down to minimum
			if(features.getNumRows() > _min_rows) {
				// generate subsampling matrix
				MatrixBlock subsampleMatrixBlock = ParamservUtils.generateSubsampleMatrix(_min_rows, Math.toIntExact(features.getNumRows()), System.currentTimeMillis());
				subsampleTo(features, subsampleMatrixBlock);
				subsampleTo(labels, subsampleMatrixBlock);
			}

			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS);
		}
	}
}
