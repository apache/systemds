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
import org.apache.sysds.runtime.controlprogram.BasicProgramBlock;
import org.apache.sysds.runtime.controlprogram.FunctionProgramBlock;
import org.apache.sysds.runtime.controlprogram.Program;
import org.apache.sysds.runtime.controlprogram.ProgramBlock;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.*;
import org.apache.sysds.runtime.controlprogram.parfor.stat.Timing;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.cp.*;
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
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.Future;

public class FederatedLocalPSThread extends LocalPSWorker implements Callable<Void> {
	FederatedData _featuresData;
	FederatedData _labelsData;
	String _instructionString;
	String _programSerialized;

	public FederatedLocalPSThread(int workerID, String updFunc, Statement.PSFrequency freq, int epochs, long batchSize, ExecutionContext ec, ParamServer ps) {
		super(workerID, updFunc, freq, epochs, batchSize, ec, ps);
	}

	/**
	 * Sets up the federated worker and control thread
	 */
	public void setup() {
		System.out.println("[+] Control thread setting up");
		// prepare features and labels
		_features.getFedMapping().forEachParallel((range, data) -> {
			_featuresData = data;
			return null;
		});
		_labels.getFedMapping().forEachParallel((range, data) -> {
			_labelsData = data;
			return null;
		});

		// serialize program
		// Create a program block for the instruction filtering
		BasicProgramBlock updateProgramBlock = new BasicProgramBlock(_ec.getProgram());
		updateProgramBlock.setInstructions(new ArrayList<>(Arrays.asList(_inst)));
		ArrayList<ProgramBlock> updateProgramBlocks = new ArrayList<>();
		updateProgramBlocks.add(updateProgramBlock);

		_programSerialized = ProgramConverter.serializeProgram(_ec.getProgram(),
				updateProgramBlocks,
				new HashMap<>(),
				false
		);

		// serialize instruction
		_instructionString = _inst.toString();

		// write program to worker and meta data to worker
		Future<FederatedResponse> udfResponse = _featuresData.executeFederatedOperation(new FederatedRequest(FederatedRequest.RequestType.EXEC_UDF,
				_featuresData.getVarID(),
				new setupFederatedWorker(_batchSize, _features.getNumRows(), _instructionString, _programSerialized)
		));

		try {
			FederatedResponse response = udfResponse.get();
			if(!response.isSuccessful())
				throw new DMLRuntimeException("FederatedLocalPSThread: Setup UDF failed");
		}
		catch(Exception e) {
			throw new DMLRuntimeException("FederatedLocalPSThread: failed to execute Setup UDF" + e.getMessage());
		}
		System.out.println("[+] Setup of federated worker successful");
	}

	/**
	 * Setup UDF
	 */
	private static class setupFederatedWorker extends FederatedUDF {
		long _batchSize;
		long _dataSize;
		String _instructionString;
		String _programString;

		protected setupFederatedWorker(long batchSize, long dataSize, String instString, String programString) {
			super(new long[]{});
			_batchSize = batchSize;
			_dataSize = dataSize;
			_instructionString = instString;
			_programString = programString;
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			// parse and set program
			Program prog = ProgramConverter.parseProgram(_programString, 0);
			ec.setProgram(prog);

			// set variables to ec
			ec.setVariable("batchSize", new IntObject(_batchSize));
			ec.setVariable("dataSize", new IntObject(_dataSize));
			ec.setVariable("instructionString", new StringObject(_instructionString));

			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS);
		}
	}

	/**
	 * Overriding the compute gradients of the local worker to use it as a control thread
	 *
	 * @param params		the current model
	 * @param dataSize		the data size; not needed
	 * @param batchIter		named bad; number of batches; only needed for debugging
	 * @param i				current epoch; named bad and only used for debug
	 * @param j				current batch; named bad and really useful
	 * @return
	 */
	@Override
	protected ListObject computeGradients(ListObject params, long dataSize, int batchIter, int i, int j) {
		System.out.println("[+] Control thread started computing gradients method");

		// put batch number on federated worker
		long jVarID = FederationUtils.getNextFedDataID();
		Future<FederatedResponse> putJResponse = _featuresData.executeFederatedOperation(new FederatedRequest(FederatedRequest.RequestType.PUT_VAR, jVarID, new IntObject(j)));

		// put current parameters on federated worker
		long paramsVarID = FederationUtils.getNextFedDataID();
		Future<FederatedResponse> putParamsResponse = _featuresData.executeFederatedOperation(new FederatedRequest(FederatedRequest.RequestType.PUT_VAR, paramsVarID, params));

		try {
			if(!putParamsResponse.get().isSuccessful() || !putJResponse.get().isSuccessful())
				throw new DMLRuntimeException("FederatedLocalPSThread: put was not successful");
		}
		catch(Exception e) {
			throw new DMLRuntimeException("FederatedLocalPSThread: failed to execute put" + e.getMessage());
		}

		System.out.println("[+] Writing of params and current batch successful");

		// create and execute the udf on the remote worker
		Future<FederatedResponse> udfResponse = _featuresData.executeFederatedOperation(new FederatedRequest(FederatedRequest.RequestType.EXEC_UDF,
				_featuresData.getVarID(),
				new federatedComputeGradients(new long[]{_featuresData.getVarID(), _labelsData.getVarID(), jVarID, paramsVarID})
		));

		try {
			Object[] responseData = udfResponse.get().getData();
			System.out.println("[+] Got response: " + responseData[0]);
			System.out.println("[+] Gradients calculation on federated worker successful");
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
			MatrixObject features = (MatrixObject) data[0];
			MatrixObject labels = (MatrixObject) data[1];
			long j = ((IntObject) data[2]).getLongValue();
			ListObject params = (ListObject) data[3];

			long batchSize = ((IntObject) ec.getVariable("batchSize")).getLongValue();
			long dataSize = ((IntObject) ec.getVariable("dataSize")).getLongValue();
			String instructionString = ((StringObject) ec.getVariable("instructionString")).getStringValue();

			// slice batch from feature and label matrix
			long begin = j * batchSize + 1;
			long end = Math.min((j + 1) * batchSize, dataSize);
			MatrixObject bFeatures = ParamservUtils.sliceMatrix(features, begin, end);
			MatrixObject bLabels = ParamservUtils.sliceMatrix(labels, begin, end);

			// prepare execution context
			ec.setVariable(Statement.PS_MODEL, params);
			ec.setVariable(Statement.PS_FEATURES, bFeatures);
			ec.setVariable(Statement.PS_LABELS, bLabels);

			System.out.println("Starting instruction");

			/*_inst.processInstruction(ec);
			// Get the gradients
			ListObject gradients = ec.getListObject(_output.getName());*/

			//return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS, new Object[]{gradients});
			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS, new Object[]{new ArrayList<>()});
		}
	}
}
