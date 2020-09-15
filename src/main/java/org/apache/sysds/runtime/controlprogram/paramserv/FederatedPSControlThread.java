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
import org.apache.sysds.runtime.controlprogram.ProgramBlock;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.federated.FederatedData;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederatedUDF;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.controlprogram.parfor.stat.Timing;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.FunctionCallCPInstruction;
import org.apache.sysds.runtime.instructions.cp.IntObject;
import org.apache.sysds.runtime.instructions.cp.ListObject;
import org.apache.sysds.runtime.instructions.cp.StringObject;
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
	int _totalNumBatches;

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

		// calculate number of batches and get data size
		long dataSize = _features.getNumRows();
		_totalNumBatches = (int) Math.ceil((double) dataSize / _batchSize);

		// serialize program
		// Create program blocks for the instruction filtering
		String programSerialized = "";
		ArrayList<ProgramBlock> programBlocks = new ArrayList<>();

		BasicProgramBlock gradientProgramBlock = new BasicProgramBlock(_ec.getProgram());
		gradientProgramBlock.setInstructions(new ArrayList<>(Arrays.asList(_inst)));
		programBlocks.add(gradientProgramBlock);

		if(_freq == Statement.PSFrequency.EPOCH) {
			BasicProgramBlock aggProgramBlock = new BasicProgramBlock(_ec.getProgram());
			aggProgramBlock.setInstructions(new ArrayList<>(Arrays.asList(_ps.getAggInst())));
			programBlocks.add(aggProgramBlock);
		}

		StringBuilder sb = new StringBuilder();
		sb.append(PROG_BEGIN);
		sb.append( NEWLINE );
		sb.append(ProgramConverter.serializeProgram(_ec.getProgram(),
				programBlocks,
				new HashMap<>(),
				false
		));
		sb.append(PROG_END);
		programSerialized = sb.toString();

		// write program and meta data to worker
		Future<FederatedResponse> udfResponse = _featuresData.executeFederatedOperation(new FederatedRequest(FederatedRequest.RequestType.EXEC_UDF,
				_featuresData.getVarID(),
				new setupFederatedWorker(_batchSize,
						dataSize,
						_totalNumBatches,
						programSerialized,
						_inst.getNamespace(),
						_inst.getFunctionName(),
						_ps.getAggInst().getFunctionName(),
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
		long _numBatches;
		String _programString;
		String _namespace;
		String _gradientsFunctionName;
		String _aggregationFunctionName;
		ListObject _hyperParams;

		protected setupFederatedWorker(long batchSize, long dataSize, long numBatches, String programString, String namespace, String gradientsFunctionName, String aggregationFunctionName, ListObject hyperParams) {
			super(new long[]{});
			_batchSize = batchSize;
			_dataSize = dataSize;
			_numBatches = numBatches;
			_programString = programString;
			_namespace = namespace;
			_gradientsFunctionName = gradientsFunctionName;
			_aggregationFunctionName = aggregationFunctionName;
			_hyperParams = hyperParams;
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			// parse and set program - WARNING: the parsing writes all functions to optimized
			ec.setProgram(ProgramConverter.parseProgram(_programString, 0));

			// set variables to ec
			ec.setVariable(Statement.PS_FED_BATCH_SIZE, new IntObject(_batchSize));
			ec.setVariable(Statement.PS_FED_DATA_SIZE, new IntObject(_dataSize));
			ec.setVariable(Statement.PS_FED_NUM_BATCHES, new IntObject(_numBatches));
			ec.setVariable(Statement.PS_FED_NAMESPACE, new StringObject(_namespace));
			ec.setVariable(Statement.PS_FED_GRADIENTS_FNAME, new StringObject(_gradientsFunctionName));
			ec.setVariable(Statement.PS_FED_AGGREGATION_FNAME, new StringObject(_aggregationFunctionName));
			ec.setVariable(Statement.PS_HYPER_PARAMS, _hyperParams);

			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS);
		}
	}

	// Entry point
	@Override
	public Void call() throws Exception {
		try {
			switch (_freq) {
				case BATCH:
					computeBatch(_totalNumBatches);
					break;
				case EPOCH:
					computeEpoch();
					break;
				default:
					throw new DMLRuntimeException(String.format("%s not support update frequency %s", getWorkerName(), _freq));
			}
		} catch (Exception e) {
			throw new DMLRuntimeException(String.format("%s failed", getWorkerName()), e);
		}
		return null;
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
	protected void computeBatch(int numBatches) {
		for (int epochCounter = 0; epochCounter < _epochs; epochCounter++) {
			for (int batchCounter = 0; batchCounter < numBatches; batchCounter++) {
				ListObject model = pullModel();
				pushGradients(computeBatchGradients(model, batchCounter));
			}
		}
	}

	protected ListObject computeBatchGradients(ListObject model, int batchCounter) {
		// put batch counter on federated worker
		long batchCounterVarID = FederationUtils.getNextFedDataID();
		Future<FederatedResponse> putBatchCounterResponse = _featuresData.executeFederatedOperation(new FederatedRequest(FederatedRequest.RequestType.PUT_VAR, batchCounterVarID, new IntObject(batchCounter)));

		// put current model on federated worker
		long modelVarID = FederationUtils.getNextFedDataID();
		Future<FederatedResponse> putParamsResponse = _featuresData.executeFederatedOperation(new FederatedRequest(FederatedRequest.RequestType.PUT_VAR, modelVarID, model));

		try {
			if(!putParamsResponse.get().isSuccessful() || !putBatchCounterResponse.get().isSuccessful())
				throw new DMLRuntimeException("FederatedLocalPSThread: put was not successful");
		}
		catch(Exception e) {
			throw new DMLRuntimeException("FederatedLocalPSThread: failed to execute put" + e.getMessage());
		}

		// create and execute the udf on the remote worker
		Future<FederatedResponse> udfResponse = _featuresData.executeFederatedOperation(new FederatedRequest(FederatedRequest.RequestType.EXEC_UDF,
				_featuresData.getVarID(),
				new federatedComputeBatchGradients(new long[]{_featuresData.getVarID(), _labelsData.getVarID(), batchCounterVarID, modelVarID})
		));

		try {
			Object[] responseData = udfResponse.get().getData();
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
			long batchCounter = ((IntObject) data[2]).getLongValue();
			ListObject model = (ListObject) data[3];

			// get data from execution context
			long batchSize = ((IntObject) ec.getVariable(Statement.PS_FED_BATCH_SIZE)).getLongValue();
			long dataSize = ((IntObject) ec.getVariable(Statement.PS_FED_DATA_SIZE)).getLongValue();
			String namespace = ((StringObject) ec.getVariable(Statement.PS_FED_NAMESPACE)).getStringValue();
			String gradientsFunctionName = ((StringObject) ec.getVariable(Statement.PS_FED_GRADIENTS_FNAME)).getStringValue();

			// slice batch from feature and label matrix
			long begin = batchCounter * batchSize + 1;
			long end = Math.min((batchCounter + 1) * batchSize, dataSize);
			MatrixObject bFeatures = ParamservUtils.sliceMatrix(features, begin, end);
			MatrixObject bLabels = ParamservUtils.sliceMatrix(labels, begin, end);

			// prepare execution context
			ec.setVariable(Statement.PS_MODEL, model);
			ec.setVariable(Statement.PS_FEATURES, bFeatures);
			ec.setVariable(Statement.PS_LABELS, bLabels);

			// recreate gradient instruction and output
			FunctionProgramBlock func = ec.getProgram().getFunctionProgramBlock(namespace, gradientsFunctionName, true);
			ArrayList<DataIdentifier> inputs = func.getInputParams();
			ArrayList<DataIdentifier> outputs = func.getOutputParams();
			CPOperand[] boundInputs = inputs.stream()
					.map(input -> new CPOperand(input.getName(), input.getValueType(), input.getDataType()))
					.toArray(CPOperand[]::new);
			ArrayList<String> outputNames = outputs.stream().map(DataIdentifier::getName)
					.collect(Collectors.toCollection(ArrayList::new));
			Instruction gradientsInstruction = new FunctionCallCPInstruction(namespace, gradientsFunctionName, true, boundInputs,
					func.getInputParamNames(), outputNames, "gradient function");
			DataIdentifier gradientsOutput = outputs.get(0);

			// calculate and return gradients
			gradientsInstruction.processInstruction(ec);
			ListObject gradients = ec.getListObject(gradientsOutput.getName());
			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS, gradients);
		}
	}



	// Epoch computation logic for federated worker
	protected void computeEpoch() {
		for (int epochCounter = 0; epochCounter < _epochs; epochCounter++) {
			// Pull the global parameters from ps
			ListObject model = pullModel();
			pushGradients(computeEpochGradients(model));
		}
	}

	protected ListObject computeEpochGradients(ListObject model) {
		// put current model on federated worker
		long modelVarID = FederationUtils.getNextFedDataID();
		Future<FederatedResponse> putParamsResponse = _featuresData.executeFederatedOperation(new FederatedRequest(FederatedRequest.RequestType.PUT_VAR, modelVarID, model));

		try {
			if(!putParamsResponse.get().isSuccessful())
				throw new DMLRuntimeException("FederatedLocalPSThread: put was not successful");
		}
		catch(Exception e) {
			throw new DMLRuntimeException("FederatedLocalPSThread: failed to execute put" + e.getMessage());
		}

		// create and execute the udf on the remote worker
		Future<FederatedResponse> udfResponse = _featuresData.executeFederatedOperation(new FederatedRequest(FederatedRequest.RequestType.EXEC_UDF,
				_featuresData.getVarID(),
				new federatedComputeEpochGradients(new long[]{_featuresData.getVarID(), _labelsData.getVarID(), modelVarID})
		));

		try {
			Object[] responseData = udfResponse.get().getData();
			return (ListObject) responseData[0];
		}
		catch(Exception e) {
			throw new DMLRuntimeException("FederatedLocalPSThread: failed to execute UDF" + e.getMessage());
		}

	}

	private static class federatedComputeEpochGradients extends FederatedUDF {
		protected federatedComputeEpochGradients(long[] inIDs) {
			super(inIDs);
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			// read in data by varid
			MatrixObject features = (MatrixObject) data[0];
			MatrixObject labels = (MatrixObject) data[1];
			ListObject model = (ListObject) data[2];

			// get data from execution context
			long batchSize = ((IntObject) ec.getVariable(Statement.PS_FED_BATCH_SIZE)).getLongValue();
			long dataSize = ((IntObject) ec.getVariable(Statement.PS_FED_DATA_SIZE)).getLongValue();
			long numBatches = ((IntObject) ec.getVariable(Statement.PS_FED_NUM_BATCHES)).getLongValue();
			String namespace = ((StringObject) ec.getVariable(Statement.PS_FED_NAMESPACE)).getStringValue();
			String gradientsFunctionName = ((StringObject) ec.getVariable(Statement.PS_FED_GRADIENTS_FNAME)).getStringValue();
			String aggregationFuctionName = ((StringObject) ec.getVariable(Statement.PS_FED_AGGREGATION_FNAME)).getStringValue();

			// recreate gradient instruction and output
			FunctionProgramBlock func = ec.getProgram().getFunctionProgramBlock(namespace, gradientsFunctionName, true);
			ArrayList<DataIdentifier> inputs = func.getInputParams();
			ArrayList<DataIdentifier> outputs = func.getOutputParams();
			CPOperand[] boundInputs = inputs.stream()
					.map(input -> new CPOperand(input.getName(), input.getValueType(), input.getDataType()))
					.toArray(CPOperand[]::new);
			ArrayList<String> outputNames = outputs.stream().map(DataIdentifier::getName)
					.collect(Collectors.toCollection(ArrayList::new));
			Instruction gradientsInstruction = new FunctionCallCPInstruction(namespace, gradientsFunctionName, true, boundInputs,
					func.getInputParamNames(), outputNames, "gradient function");
			DataIdentifier gradientsOutput = outputs.get(0);

			// recreate aggregation instruction and output
			func = ec.getProgram().getFunctionProgramBlock(namespace, aggregationFuctionName, true);
			inputs = func.getInputParams();
			outputs = func.getOutputParams();
			boundInputs = inputs.stream()
					.map(input -> new CPOperand(input.getName(), input.getValueType(), input.getDataType()))
					.toArray(CPOperand[]::new);
			outputNames = outputs.stream().map(DataIdentifier::getName)
					.collect(Collectors.toCollection(ArrayList::new));
			Instruction aggregationInstruction = new FunctionCallCPInstruction(namespace, aggregationFuctionName, true, boundInputs,
					func.getInputParamNames(), outputNames, "aggregation function");
			DataIdentifier aggregationOutput = outputs.get(0);


			ListObject accGradients = null;
			// prepare execution context
			ec.setVariable(Statement.PS_MODEL, model);
			for (int batchCounter = 0; batchCounter < numBatches; batchCounter++) {
				// slice batch from feature and label matrix
				long begin = batchCounter * batchSize + 1;
				long end = Math.min((batchCounter + 1) * batchSize, dataSize);
				MatrixObject bFeatures = ParamservUtils.sliceMatrix(features, begin, end);
				MatrixObject bLabels = ParamservUtils.sliceMatrix(labels, begin, end);

				// prepare execution context
				ec.setVariable(Statement.PS_FEATURES, bFeatures);
				ec.setVariable(Statement.PS_LABELS, bLabels);
				boolean localUpdate = batchCounter < numBatches - 1;

				// calculate intermediate gradients
				gradientsInstruction.processInstruction(ec);
				ListObject gradients = ec.getListObject(gradientsOutput.getName());

				// TODO: is this equivalent for momentum based and AMS prob?
				accGradients = ParamservUtils.accrueGradients(accGradients, gradients, false);

				// Update the local model with gradients
				if(localUpdate) {
					// Set gradients in execution Context
					ec.setVariable(Statement.PS_GRADIENTS, gradients);
					// Invoke the aggregate function
					aggregationInstruction.processInstruction(ec);
					// Get the new model
					model = ec.getListObject(aggregationOutput.getName());
					// Set new model in execution context
					ec.setVariable(Statement.PS_MODEL, model);
				}
			}

			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS, accGradients);
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
