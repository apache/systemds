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
import org.apache.sysds.runtime.util.ProgramConverter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.concurrent.Callable;
import java.util.concurrent.Future;
import java.util.stream.Collectors;

import static org.apache.sysds.runtime.util.ProgramConverter.*;

public class FederatedPSControlThread extends PSWorker implements Callable<Void> {
	FederatedData _featuresData;
	FederatedData _labelsData;

	public FederatedPSControlThread(int workerID, String updFunc, Statement.PSFrequency freq, int epochs, long batchSize, ExecutionContext ec, ParamServer ps) {
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

		StringBuilder sb = new StringBuilder();
		sb.append(PROG_BEGIN);
		sb.append( NEWLINE );
		sb.append(ProgramConverter.serializeProgram(_ec.getProgram(),
				updateProgramBlocks,
				new HashMap<>(),
				false
		));
		sb.append(PROG_END);
		String programSerialized = sb.toString();

		// write program and meta data to worker
		Future<FederatedResponse> udfResponse = _featuresData.executeFederatedOperation(new FederatedRequest(FederatedRequest.RequestType.EXEC_UDF,
				_featuresData.getVarID(),
				new setupFederatedWorker(_batchSize,
						_features.getNumRows(),
						programSerialized,
						_inst.getNamespace(),
						_inst.getFunctionName(),
						_ec.getListObject("hyperparams")
				)
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
		String _programString;
		String _namespace;
		String _functionName;
		ListObject _hyperParams;

		protected setupFederatedWorker(long batchSize, long dataSize, String programString, String namespace, String functionName, ListObject hyperParams) {
			super(new long[]{});
			_batchSize = batchSize;
			_dataSize = dataSize;
			_programString = programString;
			_namespace = namespace;
			_functionName = functionName;
			_hyperParams = hyperParams;
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			// parse and set program 	WARNING: the parsing writes all functions to optimized
			Program prog = ProgramConverter.parseProgram(_programString, 0);
			ec.setProgram(prog);

			// set variables to ec
			ec.setVariable("batchSize", new IntObject(_batchSize));
			ec.setVariable("dataSize", new IntObject(_dataSize));
			ec.setVariable("namespace", new StringObject(_namespace));
			ec.setVariable("functionName", new StringObject(_functionName));
			ec.setVariable("hyperparams", _hyperParams);

			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS);
		}
	}

	// Entry point
	@Override
	public Void call() throws Exception {
		try {
			long dataSize = _features.getNumRows();
			int numBatches = (int) Math.ceil((double) dataSize / _batchSize);

			switch (_freq) {
				case BATCH:
					computeBatch(numBatches);
					break;
				case EPOCH:
					computeEpoch(numBatches);
					break;
				default:
					throw new DMLRuntimeException(String.format("%s not support update frequency %s", getWorkerName(), _freq));
			}
		} catch (Exception e) {
			throw new DMLRuntimeException(String.format("%s failed", getWorkerName()), e);
		}
		return null;
	}

	// TODO: Cleanup after - this sends the model unnecessarily often
	protected void computeEpoch(int numBatches) {
		for (int epochCounter = 0; epochCounter < _epochs; epochCounter++) {
			// Pull the global parameters from ps
			ListObject params = pullModel();
			ListObject accGradients = null;
			
			for (int batchCounter = 0; batchCounter < numBatches; batchCounter++) {
				ListObject gradients = computeBatchGradients(params, batchCounter);

				boolean localUpdate = batchCounter < numBatches - 1;
				// Accumulate the intermediate gradients
				accGradients = ParamservUtils.accrueGradients(accGradients, gradients, !localUpdate);

				// Update the local model with gradients
				if(localUpdate)
					params = updateModel(params, gradients, epochCounter, batchCounter, numBatches);
			}

			// Push the gradients to ps
			pushGradients(accGradients);
		}
	}

	// TODO: Cleanup after functionality is fine
	protected void computeBatch(int numBatches) {
		for (int epochCounter = 0; epochCounter < _epochs; epochCounter++) {
			for (int batchCounter = 0; batchCounter < numBatches; batchCounter++) {
				ListObject globalParams = pullModel();
				ListObject gradients = computeBatchGradients(globalParams, batchCounter);

				// Push the gradients to ps
				pushGradients(gradients);
			}
		}
	}

	// TODO: Will not work on federated worker - beware when implementing
	protected ListObject updateModel(ListObject globalParams, ListObject gradients, int i, int j, int batchIter) {
		globalParams = _ps.updateLocalModel(_ec, gradients, globalParams);
		return globalParams;
	}

	protected ListObject pullModel() {
		// Pull the global parameters from ps
		ListObject model = _ps.pull(_workerID);
		return model;
	}

	protected void pushGradients(ListObject gradients) {
		// Push the gradients to ps
		_ps.push(_workerID, gradients);
	}

	// Batch computation logic for federated worker
	protected ListObject computeBatchGradients(ListObject model, int batchCounter) {
		System.out.println("[+] Control thread started computing gradients method");

		// put batch counter on federated worker
		long batchCounterVarID = FederationUtils.getNextFedDataID();
		Future<FederatedResponse> putJResponse = _featuresData.executeFederatedOperation(new FederatedRequest(FederatedRequest.RequestType.PUT_VAR, batchCounterVarID, new IntObject(batchCounter)));

		// put current model on federated worker
		long modelVarID = FederationUtils.getNextFedDataID();
		Future<FederatedResponse> putParamsResponse = _featuresData.executeFederatedOperation(new FederatedRequest(FederatedRequest.RequestType.PUT_VAR, modelVarID, model));

		try {
			if(!putParamsResponse.get().isSuccessful() || !putJResponse.get().isSuccessful())
				throw new DMLRuntimeException("FederatedLocalPSThread: put was not successful");
		}
		catch(Exception e) {
			throw new DMLRuntimeException("FederatedLocalPSThread: failed to execute put" + e.getMessage());
		}
		System.out.println("[+] Writing of model and current batch successful");

		// create and execute the udf on the remote worker
		Future<FederatedResponse> udfResponse = _featuresData.executeFederatedOperation(new FederatedRequest(FederatedRequest.RequestType.EXEC_UDF,
				_featuresData.getVarID(),
				new federatedComputeBatchGradients(new long[]{_featuresData.getVarID(), _labelsData.getVarID(), batchCounterVarID, modelVarID})
		));

		try {
			Object[] responseData = udfResponse.get().getData();
			System.out.println("[+] Gradients calculation on federated worker successful");
			return (ListObject) responseData[0];
		}
		catch(Exception e) {
			throw new DMLRuntimeException("FederatedLocalPSThread: failed to execute UDF" + e.getMessage());
		}
	}

	private static class federatedComputeBatchGradients extends FederatedUDF {
		protected federatedComputeBatchGradients(long[] inIDs) {
			super(inIDs);
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			// read in data by varid
			MatrixObject features = (MatrixObject) data[0];
			MatrixObject labels = (MatrixObject) data[1];
			long j = ((IntObject) data[2]).getLongValue();
			ListObject model = (ListObject) data[3];

			// get data from execution context
			long batchSize = ((IntObject) ec.getVariable("batchSize")).getLongValue();
			long dataSize = ((IntObject) ec.getVariable("dataSize")).getLongValue();
			String namespace = ((StringObject) ec.getVariable("namespace")).getStringValue();
			String functionName = ((StringObject) ec.getVariable("functionName")).getStringValue();

			// slice batch from feature and label matrix
			long begin = j * batchSize + 1;
			long end = Math.min((j + 1) * batchSize, dataSize);
			MatrixObject bFeatures = ParamservUtils.sliceMatrix(features, begin, end);
			MatrixObject bLabels = ParamservUtils.sliceMatrix(labels, begin, end);

			// prepare execution context
			ec.setVariable(Statement.PS_MODEL, model);
			ec.setVariable(Statement.PS_FEATURES, bFeatures);
			ec.setVariable(Statement.PS_LABELS, bLabels);

			// recreate instruction and output
			// TODO: Serialize instruction to ec or maybe put in program
			FunctionProgramBlock func = ec.getProgram().getFunctionProgramBlock(namespace, functionName, true);
			ArrayList<DataIdentifier> inputs = func.getInputParams();
			ArrayList<DataIdentifier> outputs = func.getOutputParams();
			CPOperand[] boundInputs = inputs.stream()
					.map(input -> new CPOperand(input.getName(), input.getValueType(), input.getDataType()))
					.toArray(CPOperand[]::new);
			ArrayList<String> outputNames = outputs.stream().map(DataIdentifier::getName)
					.collect(Collectors.toCollection(ArrayList::new));
			Instruction instruction = new FunctionCallCPInstruction(namespace, functionName, true, boundInputs,
					func.getInputParamNames(), outputNames, "update function");
			DataIdentifier output = outputs.get(0);

			// calculate and return gradients
			instruction.processInstruction(ec);
			ListObject gradients = ec.getListObject(output.getName());
			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS, gradients);
		}
	}

	// Statistics methods
	@Override
	public String getWorkerName() {
		return String.format("Local worker_%d", _workerID);
	}

	@Override
	protected void incWorkerNumber() {

	}

	@Override
	protected void accLocalModelUpdateTime(Timing time) {

	}

	@Override
	protected void accBatchIndexingTime(Timing time) {

	}

	@Override
	protected void accGradientComputeTime(Timing time) {

	}
}
