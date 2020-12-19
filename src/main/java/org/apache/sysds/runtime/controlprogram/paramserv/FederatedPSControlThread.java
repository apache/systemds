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

import org.apache.commons.lang.NotImplementedException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
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
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest.RequestType;
import org.apache.sysds.runtime.controlprogram.federated.FederatedResponse;
import org.apache.sysds.runtime.controlprogram.federated.FederatedUDF;
import org.apache.sysds.runtime.controlprogram.federated.FederationUtils;
import org.apache.sysds.runtime.controlprogram.parfor.stat.Timing;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.FunctionCallCPInstruction;
import org.apache.sysds.runtime.instructions.cp.IntObject;
import org.apache.sysds.runtime.instructions.cp.ListObject;
import org.apache.sysds.runtime.instructions.cp.StringObject;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.RightScalarOperator;
import org.apache.sysds.runtime.util.ProgramConverter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.concurrent.Callable;
import java.util.concurrent.Future;
import java.util.stream.Collectors;

import static org.apache.sysds.runtime.util.ProgramConverter.*;

public class FederatedPSControlThread extends PSWorker implements Callable<Void> {
	private static final long serialVersionUID = 6846648059569648791L;
	protected static final Log LOG = LogFactory.getLog(ParamServer.class.getName());

	Statement.PSRuntimeBalancing _runtimeBalancing;
	FederatedData _featuresData;
	FederatedData _labelsData;
	final long _localStartBatchNumVarID;
	final long _modelVarID;

	// runtime balancing
	Statement.PSRuntimeBalancing _runtimeBalancing;
	int _numBatchesPerGlobalEpoch;
	int _possibleBatchesPerLocalEpoch;
	boolean _scale = false;
	double _scalingFactor = 1;
	boolean _cycleStartAt0 = false;

	public FederatedPSControlThread(int workerID, String updFunc, Statement.PSFrequency freq,
									Statement.PSRuntimeBalancing runtimeBalancing, int epochs, long batchSize,
									int numBatchesPerGlobalEpoch, ExecutionContext ec, ParamServer ps) {
		super(workerID, updFunc, freq, epochs, batchSize, ec, ps);

		_numBatchesPerGlobalEpoch = numBatchesPerGlobalEpoch;
		_runtimeBalancing = runtimeBalancing;
		// generate the IDs for model and batch counter. These get overwritten on the federated worker each time
		_localStartBatchNumVarID = FederationUtils.getNextFedDataID();
		_modelVarID = FederationUtils.getNextFedDataID();
	}

	/**
	 * Sets up the federated worker and control thread
	 */
	public void setup() {
		// prepare features and labels
		_featuresData = (FederatedData) _features.getFedMapping().getMap().values().toArray()[0];
		_labelsData = (FederatedData) _labels.getFedMapping().getMap().values().toArray()[0];

		// different runtime balancing calculations
		long dataSize = _features.getNumRows();

		// calculate scaled batch size if balancing via batch size. Some cycling may occur if the rows are not
		// divisible by the _numBatchesPerGlobalEpoch
		// In this case, the cycling will always start fresh over epochs.
		// Example: numBatchesPerGlobalEpoch = 2, dataSize = 1 => batchSize will be 1 and therefore cycle once.
		if(_runtimeBalancing == Statement.PSRuntimeBalancing.SCALE_BATCH
				|| _runtimeBalancing == Statement.PSRuntimeBalancing.SCALE_BATCH_AND_WEIGH) {
			long batchSizeNonScaled = _batchSize;
			_batchSize = (int) Math.ceil((double) dataSize / _numBatchesPerGlobalEpoch);
			_cycleStartAt0 = true;

			// calculate weighing factor
			if(_runtimeBalancing == Statement.PSRuntimeBalancing.SCALE_BATCH_AND_WEIGH) {
				_scale = true;
				_scalingFactor = (double) _batchSize / batchSizeNonScaled;
			}
		}
		// Calculate possible batches with batch size
		_possibleBatchesPerLocalEpoch = (int) Math.ceil((double) dataSize / _batchSize);
		// If no runtime balancing is specified, just run possible number of batches
		// WARNING: Will get stuck on miss match
		if(_runtimeBalancing == Statement.PSRuntimeBalancing.NONE) {
			_numBatchesPerGlobalEpoch = _possibleBatchesPerLocalEpoch;
		}

		if(_runtimeBalancing == Statement.PSRuntimeBalancing.SCALE_BATCH
			|| _runtimeBalancing == Statement.PSRuntimeBalancing.SCALE_BATCH_AND_WEIGH) {
			throw new NotImplementedException();
		}

		// serialize program
		// create program blocks for the instruction filtering
		String programSerialized;
		ArrayList<ProgramBlock> programBlocks = new ArrayList<>();

		BasicProgramBlock gradientProgramBlock = new BasicProgramBlock(_ec.getProgram());
		gradientProgramBlock.setInstructions(new ArrayList<>(Arrays.asList(_inst)));
		programBlocks.add(gradientProgramBlock);

		if(_freq == Statement.PSFrequency.EPOCH) {
			BasicProgramBlock aggProgramBlock = new BasicProgramBlock(_ec.getProgram());
			aggProgramBlock.setInstructions(new ArrayList<>(Arrays.asList(_ps.getAggInst())));
			programBlocks.add(aggProgramBlock);
		}

		programSerialized = PROG_BEGIN +
				NEWLINE +
				ProgramConverter.serializeProgram(_ec.getProgram(),
						programBlocks,
						new HashMap<>(),
						false
				) +
				PROG_END;

		// write program and meta data to worker
		Future<FederatedResponse> udfResponse = _featuresData.executeFederatedOperation(
			new FederatedRequest(RequestType.EXEC_UDF, _featuresData.getVarID(),
				new SetupFederatedWorker(_batchSize,
						dataSize,
						_possibleBatchesPerLocalEpoch,
						programSerialized,
						_inst.getNamespace(),
						_inst.getFunctionName(),
						_ps.getAggInst().getFunctionName(),
						_ec.getListObject("hyperparams"),
						_localStartBatchNumVarID,
						_modelVarID
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
	}

	/**
	 * Setup UDF executed on the federated worker
	 */
	private static class SetupFederatedWorker extends FederatedUDF {
		private static final long serialVersionUID = -3148991224792675607L;
		private final long _batchSize;
		private final long _dataSize;
		private final int _possibleBatchesPerLocalEpoch;
		private final String _programString;
		private final String _namespace;
		private final String _gradientsFunctionName;
		private final String _aggregationFunctionName;
		private final ListObject _hyperParams;
		private final long _batchCounterVarID;
		private final long _modelVarID;

		protected SetupFederatedWorker(long batchSize, long dataSize, int possibleBatchesPerLocalEpoch,
			String programString, String namespace, String gradientsFunctionName, String aggregationFunctionName,
			ListObject hyperParams, long batchCounterVarID, long modelVarID)
		{
			super(new long[]{});
			_batchSize = batchSize;
			_dataSize = dataSize;
			_possibleBatchesPerLocalEpoch = possibleBatchesPerLocalEpoch;
			_programString = programString;
			_namespace = namespace;
			_gradientsFunctionName = gradientsFunctionName;
			_aggregationFunctionName = aggregationFunctionName;
			_hyperParams = hyperParams;
			_batchCounterVarID = batchCounterVarID;
			_modelVarID = modelVarID;
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			// parse and set program
			ec.setProgram(ProgramConverter.parseProgram(_programString, 0, false));

			// set variables to ec
			ec.setVariable(Statement.PS_FED_BATCH_SIZE, new IntObject(_batchSize));
			ec.setVariable(Statement.PS_FED_DATA_SIZE, new IntObject(_dataSize));
			ec.setVariable(Statement.PS_FED_POSS_BATCHES_LOCAL, new IntObject(_possibleBatchesPerLocalEpoch));
			ec.setVariable(Statement.PS_FED_NAMESPACE, new StringObject(_namespace));
			ec.setVariable(Statement.PS_FED_GRADIENTS_FNAME, new StringObject(_gradientsFunctionName));
			ec.setVariable(Statement.PS_FED_AGGREGATION_FNAME, new StringObject(_aggregationFunctionName));
			ec.setVariable(Statement.PS_HYPER_PARAMS, _hyperParams);
			ec.setVariable(Statement.PS_FED_BATCHCOUNTER_VARID, new IntObject(_batchCounterVarID));
			ec.setVariable(Statement.PS_FED_MODEL_VARID, new IntObject(_modelVarID));

			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS);
		}
	}

	/**
	 * cleans up the execution context of the federated worker
	 */
	public void teardown() {
		// write program and meta data to worker
		Future<FederatedResponse> udfResponse = _featuresData.executeFederatedOperation(
			new FederatedRequest(RequestType.EXEC_UDF, _featuresData.getVarID(),
			new TeardownFederatedWorker()
		));

		try {
			FederatedResponse response = udfResponse.get();
			if(!response.isSuccessful())
				throw new DMLRuntimeException("FederatedLocalPSThread: Teardown UDF failed");
		}
		catch(Exception e) {
			throw new DMLRuntimeException("FederatedLocalPSThread: failed to execute Teardown UDF" + e.getMessage());
		}
	}

	/**
	 * Teardown UDF executed on the federated worker
	 */
	private static class TeardownFederatedWorker extends FederatedUDF {
		private static final long serialVersionUID = -153650281873318969L;

		protected TeardownFederatedWorker() {
			super(new long[]{});
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			// remove variables from ec
			ec.removeVariable(Statement.PS_FED_BATCH_SIZE);
			ec.removeVariable(Statement.PS_FED_DATA_SIZE);
			ec.removeVariable(Statement.PS_FED_POSS_BATCHES_LOCAL);
			ec.removeVariable(Statement.PS_FED_NAMESPACE);
			ec.removeVariable(Statement.PS_FED_GRADIENTS_FNAME);
			ec.removeVariable(Statement.PS_FED_AGGREGATION_FNAME);
			ec.removeVariable(Statement.PS_FED_BATCHCOUNTER_VARID);
			ec.removeVariable(Statement.PS_FED_MODEL_VARID);
			ParamservUtils.cleanupListObject(ec, Statement.PS_HYPER_PARAMS);
			
			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS);
		}
	}

	/**
	 * Entry point of the functionality
	 *
	 * @return void
	 * @throws Exception incase the execution fails
	 */
	@Override
	public Void call() throws Exception {
		try {
			switch (_freq) {
				case BATCH:
					computeWithBatchUpdates();
					break;
				/*case NBATCH:
					computeWithNBatchUpdates();
					break; */
				case EPOCH:
					computeWithEpochUpdates();
					break;
				default:
					throw new DMLRuntimeException(String.format("%s not support update frequency %s", getWorkerName(), _freq));
			}
		} catch (Exception e) {
			throw new DMLRuntimeException(String.format("%s failed", getWorkerName()), e);
		}
		teardown();
		return null;
	}

	protected ListObject pullModel() {
		// Pull the global parameters from ps
		return _ps.pull(_workerID);
	}

	protected void scaleAndPushGradients(ListObject gradients) {
		// scale gradients - must only include MatrixObjects
		if(_scale && _scalingFactor != 1) {
			gradients.getData().parallelStream().forEach((matrix) -> {
				MatrixObject matrixObject = (MatrixObject) matrix;
				matrixObject.acquireModify(
						matrixObject.acquireReadAndRelease().scalarOperations(
								new RightScalarOperator(Multiply.getMultiplyFnObject(), _scalingFactor), new MatrixBlock())
				);
				matrixObject.release();
			});
		}

		// Push the gradients to ps
		_ps.push(_workerID, gradients);
	}

	static protected int getNextLocalBatchNum(int currentLocalBatchNumber, int possibleBatchesPerLocalEpoch) {
		return currentLocalBatchNumber % possibleBatchesPerLocalEpoch;
	}

	/**
	 * Computes all epochs and updates after each batch
	 */
	protected void computeWithBatchUpdates() {
		for (int epochCounter = 0; epochCounter < _epochs; epochCounter++) {
			int currentLocalBatchNumber = (_cycleStartAt0) ? 0 : _numBatchesPerGlobalEpoch * epochCounter % _possibleBatchesPerLocalEpoch;

			for (int batchCounter = 0; batchCounter < _numBatchesPerGlobalEpoch; batchCounter++) {
				int localStartBatchNum = getNextLocalBatchNum(currentLocalBatchNumber++, _possibleBatchesPerLocalEpoch);
				ListObject model = pullModel();
				ListObject gradients = computeGradientsForNBatches(model, 1, localStartBatchNum);
				scaleAndPushGradients(gradients);
				ParamservUtils.cleanupListObject(model);
				ParamservUtils.cleanupListObject(gradients);
			}
			if( LOG.isInfoEnabled() )
				LOG.info("[+] " + this.getWorkerName() + " completed epoch " + epochCounter);
		}
	}

	/**
	 * Computes all epochs and updates after N batches
	 */
	protected void computeWithNBatchUpdates() {
		throw new NotImplementedException();
	}

	/**
	 * Computes all epochs and updates after each epoch
	 */
	protected void computeWithEpochUpdates() {
		for (int epochCounter = 0; epochCounter < _epochs; epochCounter++) {
			int localStartBatchNum = (_cycleStartAt0) ? 0 : _numBatchesPerGlobalEpoch * epochCounter % _possibleBatchesPerLocalEpoch;

			// Pull the global parameters from ps
			ListObject model = pullModel();
			ListObject gradients = computeGradientsForNBatches(model, _numBatchesPerGlobalEpoch, localStartBatchNum, true);
			scaleAndPushGradients(gradients);

			if( LOG.isInfoEnabled() )
				LOG.info("[+] " + this.getWorkerName() + " completed epoch " + epochCounter);
			ParamservUtils.cleanupListObject(model);
			ParamservUtils.cleanupListObject(gradients);
		}
	}

	protected ListObject computeGradientsForNBatches(ListObject model, int numBatchesToCompute, int localStartBatchNum) {
		return computeGradientsForNBatches(model, numBatchesToCompute, localStartBatchNum, false);
	}

	/**
	 * Computes the gradients of n batches on the federated worker and is able to update the model local.
	 * Returns the gradients.
	 *
	 * @param model the current model from the parameter server
	 * @param localStartBatchNum the batch to start from
	 * @param localUpdate whether to update the model locally
	 *
	 * @return the gradient vector
	 */
	protected ListObject computeGradientsForNBatches(ListObject model,
		int numBatchesToCompute, int localStartBatchNum, boolean localUpdate)
	{
		// put local start batch num on federated worker
		Future<FederatedResponse> putBatchCounterResponse = _featuresData.executeFederatedOperation(
			new FederatedRequest(RequestType.PUT_VAR, _localStartBatchNumVarID, new IntObject(localStartBatchNum)));
		// put current model on federated worker
		Future<FederatedResponse> putParamsResponse = _featuresData.executeFederatedOperation(
			new FederatedRequest(RequestType.PUT_VAR, _modelVarID, model));

		try {
			if(!putParamsResponse.get().isSuccessful() || !putBatchCounterResponse.get().isSuccessful())
				throw new DMLRuntimeException("FederatedLocalPSThread: put was not successful");
		}
		catch(Exception e) {
			throw new DMLRuntimeException("FederatedLocalPSThread: failed to execute put" + e.getMessage());
		}

		// create and execute the udf on the remote worker
		Future<FederatedResponse> udfResponse = _featuresData.executeFederatedOperation(
			new FederatedRequest(RequestType.EXEC_UDF, _featuresData.getVarID(),
				new federatedComputeGradientsForNBatches(new long[]{_featuresData.getVarID(), _labelsData.getVarID(),
				_localStartBatchNumVarID, _modelVarID}, numBatchesToCompute,localUpdate)
		));

		try {
			Object[] responseData = udfResponse.get().getData();
			return (ListObject) responseData[0];
		}
		catch(Exception e) {
			throw new DMLRuntimeException("FederatedLocalPSThread: failed to execute UDF" + e.getMessage());
		}
	}

	/**
	 * This is the code that will be executed on the federated Worker when computing one gradients for n batches
	 */
	private static class federatedComputeGradientsForNBatches extends FederatedUDF {
		private static final long serialVersionUID = -3075901536748794832L;
		int _numBatchesToCompute;
		boolean _localUpdate;

		protected federatedComputeGradientsForNBatches(long[] inIDs, int numBatchesToCompute, boolean localUpdate) {
			super(inIDs);
			_numBatchesToCompute = numBatchesToCompute;
			_localUpdate = localUpdate;
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			// read in data by varid
			MatrixObject features = (MatrixObject) data[0];
			MatrixObject labels = (MatrixObject) data[1];
			int localStartBatchNum = (int) ((IntObject) data[2]).getLongValue();
			ListObject model = (ListObject) data[3];

			// get data from execution context
			long batchSize = ((IntObject) ec.getVariable(Statement.PS_FED_BATCH_SIZE)).getLongValue();
			long dataSize = ((IntObject) ec.getVariable(Statement.PS_FED_DATA_SIZE)).getLongValue();
			int possibleBatchesPerLocalEpoch = (int) ((IntObject) ec.getVariable(Statement.PS_FED_POSS_BATCHES_LOCAL)).getLongValue();
			String namespace = ((StringObject) ec.getVariable(Statement.PS_FED_NAMESPACE)).getStringValue();
			String gradientsFunctionName = ((StringObject) ec.getVariable(Statement.PS_FED_GRADIENTS_FNAME)).getStringValue();
			String aggregationFuctionName = ((StringObject) ec.getVariable(Statement.PS_FED_AGGREGATION_FNAME)).getStringValue();

			// recreate gradient instruction and output
			FunctionProgramBlock func = ec.getProgram().getFunctionProgramBlock(namespace, gradientsFunctionName, false);
			ArrayList<DataIdentifier> inputs = func.getInputParams();
			ArrayList<DataIdentifier> outputs = func.getOutputParams();
			CPOperand[] boundInputs = inputs.stream()
					.map(input -> new CPOperand(input.getName(), input.getValueType(), input.getDataType()))
					.toArray(CPOperand[]::new);
			ArrayList<String> outputNames = outputs.stream().map(DataIdentifier::getName)
					.collect(Collectors.toCollection(ArrayList::new));
			Instruction gradientsInstruction = new FunctionCallCPInstruction(namespace, gradientsFunctionName, false, boundInputs,
					func.getInputParamNames(), outputNames, "gradient function");
			DataIdentifier gradientsOutput = outputs.get(0);

			// recreate aggregation instruction and output if needed
			Instruction aggregationInstruction = null;
			DataIdentifier aggregationOutput = null;
			if(_localUpdate && _numBatchesToCompute > 1) {
				func = ec.getProgram().getFunctionProgramBlock(namespace, aggregationFuctionName, false);
				inputs = func.getInputParams();
				outputs = func.getOutputParams();
				boundInputs = inputs.stream()
						.map(input -> new CPOperand(input.getName(), input.getValueType(), input.getDataType()))
						.toArray(CPOperand[]::new);
				outputNames = outputs.stream().map(DataIdentifier::getName)
						.collect(Collectors.toCollection(ArrayList::new));
				aggregationInstruction = new FunctionCallCPInstruction(namespace, aggregationFuctionName, false, boundInputs,
						func.getInputParamNames(), outputNames, "aggregation function");
				aggregationOutput = outputs.get(0);
			}

			ListObject accGradients = null;
			int currentLocalBatchNumber = localStartBatchNum;
			// prepare execution context
			ec.setVariable(Statement.PS_MODEL, model);
			for (int batchCounter = 0; batchCounter < _numBatchesToCompute; batchCounter++) {
				int localBatchNum = getNextLocalBatchNum(currentLocalBatchNumber++, possibleBatchesPerLocalEpoch);

				// slice batch from feature and label matrix
				long begin = localBatchNum * batchSize + 1;
				long end = Math.min((localBatchNum + 1) * batchSize, dataSize);
				MatrixObject bFeatures = ParamservUtils.sliceMatrix(features, begin, end);
				MatrixObject bLabels = ParamservUtils.sliceMatrix(labels, begin, end);

				// prepare execution context
				ec.setVariable(Statement.PS_FEATURES, bFeatures);
				ec.setVariable(Statement.PS_LABELS, bLabels);

				// calculate gradients for batch
				gradientsInstruction.processInstruction(ec);
				ListObject gradients = ec.getListObject(gradientsOutput.getName());

				// accrue the computed gradients - In the single batch case this is just a list copy
				// is this equivalent for momentum based and AMS prob?
				accGradients = ParamservUtils.accrueGradients(accGradients, gradients, false);

				// update the local model with gradients if needed
				if(_localUpdate && batchCounter < _numBatchesToCompute - 1) {
					// Invoke the aggregate function
					assert aggregationInstruction != null;
					aggregationInstruction.processInstruction(ec);
					// Get the new model
					model = ec.getListObject(aggregationOutput.getName());
					// Set new model in execution context
					ec.setVariable(Statement.PS_MODEL, model);
					// clean up gradients and result
					ParamservUtils.cleanupListObject(ec, aggregationOutput.getName());
				}

				// clean up
				ParamservUtils.cleanupListObject(ec, gradientsOutput.getName());
				ParamservUtils.cleanupData(ec, Statement.PS_FEATURES);
				ParamservUtils.cleanupData(ec, Statement.PS_LABELS);
				ec.removeVariable(ec.getVariable(Statement.PS_FED_BATCHCOUNTER_VARID).toString());
				if( LOG.isInfoEnabled() )
					LOG.info("[+]" + " completed batch " + localBatchNum);
			}

			// model clean up
			ParamservUtils.cleanupListObject(ec, ec.getVariable(Statement.PS_FED_MODEL_VARID).toString());
			ParamservUtils.cleanupListObject(ec, Statement.PS_MODEL);

			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS, accGradients);
		}
	}

	// Statistics methods
	@Override
	public String getWorkerName() {
		return String.format("Federated worker_%d", _workerID);
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
