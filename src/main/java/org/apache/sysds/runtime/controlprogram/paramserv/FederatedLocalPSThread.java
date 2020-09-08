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

package org.apache.sysds.runtime.controlprogram.paramserv;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.api.mlcontext.Matrix;
import org.apache.sysds.common.Types;
import org.apache.sysds.lops.Federated;
import org.apache.sysds.parser.DataIdentifier;
import org.apache.sysds.parser.Statement;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.*;
import org.apache.sysds.runtime.controlprogram.parfor.stat.Timing;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.FunctionCallCPInstruction;
import org.apache.sysds.runtime.instructions.cp.ListObject;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.transform.encode.Encoder;
import org.apache.sysds.runtime.util.ProgramConverter;
import org.apache.sysds.utils.Statistics;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.sql.Blob;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.Future;

public class FederatedLocalPSThread extends LocalPSWorker implements Callable<Void> {
	FederatedData _featuresData;
	FederatedData _labelsData;

	public FederatedLocalPSThread(int workerID, String updFunc, Statement.PSFrequency freq, int epochs, long batchSize, ExecutionContext ec, ParamServer ps) {
		super(workerID, updFunc, freq, epochs, batchSize, ec, ps);
	}

	@Override
	protected ListObject computeGradients(ListObject params, long dataSize, int batchIter, int i, int j) {
		System.out.println("[+] Control thread started computing gradients method");

		// TODO: kind of a workaround. Assumes only one entry in the federation map
		_features.getFedMapping().forEachParallel((range, data) -> {
			_featuresData = data;
			return null;
		});
		_labels.getFedMapping().forEachParallel((range, data) -> {
			_labelsData = data;
			return null;
		});

		// put params on federated worker
		long paramsVarID = FederationUtils.getNextFedDataID();
		long jVarID = FederationUtils.getNextFedDataID();
		Future<FederatedResponse> putParamsResponse = _featuresData.executeFederatedOperation(new FederatedRequest(FederatedRequest.RequestType.PUT_VAR, paramsVarID, params));
		Future<FederatedResponse> putJResponse = _featuresData.executeFederatedOperation(new FederatedRequest(FederatedRequest.RequestType.PUT_VAR, jVarID, j));
		try {
			if(!putParamsResponse.get().isSuccessful() || !putJResponse.get().isSuccessful()) {
				throw new DMLRuntimeException("FederatedLocalPSThread: put was not successful");
			}
		}
		catch(Exception e) {
			throw new DMLRuntimeException("FederatedLocalPSThread: failed to execute put" + e.getMessage());
		}

		// execute the udf on the remote worker
		Future<FederatedResponse> udfResponse = _featuresData.executeFederatedOperation(new FederatedRequest(FederatedRequest.RequestType.EXEC_UDF,
				_featuresData.getVarID(),
				new federatedComputeGradients(new long[]{_featuresData.getVarID(), _labelsData.getVarID()},
											  _batchSize,
						 					  dataSize,
											  j
				)
		));

		try {
			Object[] responseData = udfResponse.get().getData();
			System.out.println("[+] Got response: " + responseData[0]);
			return null;
		}
		catch(Exception e) {
			throw new DMLRuntimeException("FederatedLocalPSThread: failed to execute UDF" + e.getMessage());
		}
	}

	// INPUTS Theory: Works with everything that is serializable
	private static class federatedComputeGradients extends FederatedUDF {
		long _batchSize;
		long _dataSize;
		int _j;

		protected federatedComputeGradients(long[] inIDs, long batchSize, long dataSize, int j) {
			super(inIDs);
			_batchSize = batchSize;
			_dataSize = dataSize;
			_j = j;
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			System.out.println("Starting UDF Execution");

			MatrixObject features = (MatrixObject) data[0];
			MatrixObject labels = (MatrixObject) data[1];
			ListObject params = (ListObject) data[2];

			System.out.println("Features: " + features);
			System.out.println("Labels: " + labels);
			System.out.println("Params: " + params);

			ec.setVariable(Statement.PS_MODEL, params);

			// slice batch from feature and label matrix
			long begin = _j * _batchSize + 1;
			long end = Math.min((_j + 1) * _batchSize, _dataSize);
			MatrixObject bFeatures = ParamservUtils.sliceMatrix(features, begin, end);
			MatrixObject bLabels = ParamservUtils.sliceMatrix(labels, begin, end);

			System.out.println("bFeatures: " + bFeatures);
			System.out.println("bLabels: " + bLabels);

			ec.setVariable(Statement.PS_FEATURES, bFeatures);
			ec.setVariable(Statement.PS_LABELS, bLabels);

			System.out.println("Starting instruction");

			// TODO: Logging would be nice
			/*_inst.processInstruction(ec);
			// Get the gradients
			ListObject gradients = ec.getListObject(_output.getName());*/
			// TODO: Cleanup

			//return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS, new Object[]{gradients});
			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS, new Object[]{new ArrayList<>()});
		}
	}
}
