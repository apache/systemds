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
import org.apache.sysds.runtime.transform.encode.Encoder;
import org.apache.sysds.utils.Statistics;

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

		System.out.println("[+] UDF with the following IDs: " + _featuresData.getVarID() + ", " + _labelsData.getVarID());
		// TODO: Warning. This assumes features and labels exist on the same worker
		Future<FederatedResponse> response = _featuresData.executeFederatedOperation(new FederatedRequest(FederatedRequest.RequestType.EXEC_UDF,
				_featuresData.getVarID(),
				new federatedComputeGradients(new long[]{_featuresData.getVarID(), _labelsData.getVarID()})
		));

		try {
			Object[] responseData = response.get().getData();
			System.out.println("[+] Got response");
			System.out.println("[+] Got response: " + responseData[0]);
			System.out.println("[+] Got response: " + responseData[1]);
			return null;
		}
		catch(Exception e) {
			throw new DMLRuntimeException("FederatedLocalPSThread: failed to execute UDF" + e.getMessage());
		}
	}

	private static class federatedComputeGradients extends FederatedUDF {
		protected federatedComputeGradients(long[] inIDs) {
			super(inIDs);
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			System.out.println("Federated Worker");
			/* System.out.println(data[0]);
			System.out.println(data[1]); */
			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS, new Object[]{"test1", "test2"});


			/*ec.setVariable(Statement.PS_MODEL, params);
			long begin = j * batchSize + 1;
			long end = Math.min((j + 1) * batchSize, dataSize);

			// Get batch features and labels
			MatrixObject bFeatures = ParamservUtils.sliceMatrix(features, begin, end);
			MatrixObject bLabels = ParamservUtils.sliceMatrix(labels, begin, end);

			ec.setVariable(Statement.PS_FEATURES, bFeatures);
			ec.setVariable(Statement.PS_LABELS, bLabels);

			if (LOG.isDebugEnabled()) {
				LOG.debug(String.format("%s: got batch data [size:%d kb] of index from %d to %d [last index: %d]. "
								+ "[Epoch:%d  Total epoch:%d  Iteration:%d  Total iteration:%d]", getWorkerName(),
						bFeatures.getDataSize() / 1024 + bLabels.getDataSize() / 1024, begin, end, dataSize, i + 1, epochs,
						j + 1, batchIter));
			}

			inst.processInstruction(ec);

			// Get the gradients
			ListObject gradients = ec.getListObject(_output.getName());

			ParamservUtils.cleanupData(ec, Statement.PS_FEATURES);
			ParamservUtils.cleanupData(ec, Statement.PS_LABELS);
			//return gradients;

			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS, new MatrixObject(Types.ValueType.FP64, "test"));*/
		}
	}
}
