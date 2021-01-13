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
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.parser.DataIdentifier;
import org.apache.sysds.parser.Statement;
import org.apache.sysds.parser.Statement.PSFrequency;
import org.apache.sysds.parser.Statement.PSRuntimeBalancing;
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
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.FunctionCallCPInstruction;
import org.apache.sysds.runtime.instructions.cp.IntObject;
import org.apache.sysds.runtime.instructions.cp.ListObject;
import org.apache.sysds.runtime.instructions.cp.StringObject;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.RightScalarOperator;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.util.ProgramConverter;
import org.apache.sysds.utils.Statistics;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.concurrent.Callable;
import java.util.concurrent.Future;
import java.util.stream.Collectors;

import static org.apache.sysds.runtime.util.ProgramConverter.*;

public class FederatedPSControlThread extends PSWorker implements Callable<Void> {
	private static final long serialVersionUID = 6846648059569648791L;
	protected static final Log LOG = LogFactory.getLog(ParamServer.class.getName());

	private FederatedData _featuresData;
	private FederatedData _labelsData;
	private final long _localStartBatchNumVarID;
	private final long _modelVarID;

	// runtime balancing
	private final PSRuntimeBalancing _runtimeBalancing;
	private int _numBatchesPerEpoch;
	private int _possibleBatchesPerLocalEpoch;
	private final boolean _weighing;
	private double _weighingFactor = 1;
	private final boolean _cycleStartAt0 = false;

	public FederatedPSControlThread(int workerID, String updFunc, Statement.PSFrequency freq,
		PSRuntimeBalancing runtimeBalancing, boolean weighing, int epochs, long batchSize,
		int numBatchesPerGlobalEpoch, ExecutionContext ec, ParamServer ps)
	{
		super(workerID, updFunc, freq, epochs, batchSize, ec, ps);

		_numBatchesPerEpoch = numBatchesPerGlobalEpoch;
		_runtimeBalancing = runtimeBalancing;
		_weighing = weighing;
		// generate the IDs for model and batch counter. These get overwritten on the federated worker each time
		_localStartBatchNumVarID = FederationUtils.getNextFedDataID();
		_modelVarID = FederationUtils.getNextFedDataID();
	}

	/**
	 * Sets up the federated worker and control thread
	 *
	 * @param weighingFactor Gradients from this worker will be multiplied by this factor if weighing is enabled
	 */
	public void setup(double weighingFactor) {
		incWorkerNumber();

		// prepare features and labels
		_featuresData = (FederatedData) _features.getFedMapping().getMap().values().toArray()[0];
		_labelsData = (FederatedData) _labels.getFedMapping().getMap().values().toArray()[0];

		// weighing factor is always set, but only used when weighing is specified
		_weighingFactor = weighingFactor;

		// different runtime balancing calculations
		long dataSize = _features.getNumRows();

		// calculate scaled batch size if balancing via batch size.
		// In some cases there will be some cycling
		if(_runtimeBalancing == PSRuntimeBalancing.SCALE_BATCH) {
			_batchSize = (int) Math.ceil((double) dataSize / _numBatchesPerEpoch);
		}

		// Calculate possible batches with batch size
		_possibleBatchesPerLocalEpoch = (int) Math.ceil((double) dataSize / _batchSize);

		// If no runtime balancing is specified, just run possible number of batches
		// WARNING: Will get stuck on miss match
		if(_runtimeBalancing == PSRuntimeBalancing.NONE) {
			_numBatchesPerEpoch = _possibleBatchesPerLocalEpoch;
		}

		if( LOG.isInfoEnabled() ) {
			LOG.info("Setup config for worker " + this.getWorkerName());
			LOG.info("Batch size: " + _batchSize + " possible batches: " + _possibleBatchesPerLocalEpoch
					+ " batches to run: " + _numBatchesPerEpoch + " weighing factor: " + _weighingFactor);
		}

		// serialize program
		// create program blocks for the instruction filtering
		String programSerialized;
		ArrayList<ProgramBlock> pbs = new ArrayList<>();

		BasicProgramBlock gradientProgramBlock = new BasicProgramBlock(_ec.getProgram());
		gradientProgramBlock.setInstructions(new ArrayList<>(Collections.singletonList(_inst)));
		pbs.add(gradientProgramBlock);

		if(_freq == PSFrequency.EPOCH) {
			BasicProgramBlock aggProgramBlock = new BasicProgramBlock(_ec.getProgram());
			aggProgramBlock.setInstructions(new ArrayList<>(Collections.singletonList(_ps.getAggInst())));
			pbs.add(aggProgramBlock);
		}

		programSerialized = InstructionUtils.concatStrings(
			PROG_BEGIN, NEWLINE,
			ProgramConverter.serializeProgram(_ec.getProgram(), pbs, new HashMap<>(), false),
			PROG_END);

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

		@Override
		public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
			return null;
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

		@Override
		public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
			return null;
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

	protected void weighAndPushGradients(ListObject gradients) {
		Timing tWeighing = DMLScript.STATISTICS ? new Timing(true) : null;
		// scale gradients - must only include MatrixObjects
		if(_weighing && _weighingFactor != 1) {
			gradients.getData().parallelStream().forEach((matrix) -> {
				MatrixObject matrixObject = (MatrixObject) matrix;
				MatrixBlock input = matrixObject.acquireReadAndRelease().scalarOperations(
					new RightScalarOperator(Multiply.getMultiplyFnObject(), _weighingFactor), new MatrixBlock());
				matrixObject.acquireModify(input);
				matrixObject.release();
			});
		}
		accFedPSGradientWeighingTime(tWeighing);

		// Push the gradients to ps
		_ps.push(_workerID, gradients);
	}

	protected static int getNextLocalBatchNum(int currentLocalBatchNumber, int possibleBatchesPerLocalEpoch) {
		return currentLocalBatchNumber % possibleBatchesPerLocalEpoch;
	}

	/**
	 * Computes all epochs and updates after each batch
	 */
	protected void computeWithBatchUpdates() {
		for (int epochCounter = 0; epochCounter < _epochs; epochCounter++) {
			int currentLocalBatchNumber = (_cycleStartAt0) ? 0 : _numBatchesPerEpoch * epochCounter % _possibleBatchesPerLocalEpoch;

			for (int batchCounter = 0; batchCounter < _numBatchesPerEpoch; batchCounter++) {
				int localStartBatchNum = getNextLocalBatchNum(currentLocalBatchNumber++, _possibleBatchesPerLocalEpoch);
				ListObject model = pullModel();
				ListObject gradients = computeGradientsForNBatches(model, 1, localStartBatchNum);
				weighAndPushGradients(gradients);
				ParamservUtils.cleanupListObject(model);
				ParamservUtils.cleanupListObject(gradients);
			}
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
			int localStartBatchNum = (_cycleStartAt0) ? 0 : _numBatchesPerEpoch * epochCounter % _possibleBatchesPerLocalEpoch;

			// Pull the global parameters from ps
			ListObject model = pullModel();
			ListObject gradients = computeGradientsForNBatches(model, _numBatchesPerEpoch, localStartBatchNum, true);
			weighAndPushGradients(gradients);
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
		Timing tGradients = DMLScript.STATISTICS ? new Timing(true) : null;
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
			accGradientComputeTime(tGradients);
			return (ListObject) responseData[0];
		}
		catch(Exception e) {
			if(tGradients != null)
				tGradients.stop();
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
			}

			// model clean up
			ParamservUtils.cleanupListObject(ec, ec.getVariable(Statement.PS_FED_MODEL_VARID).toString());
			ParamservUtils.cleanupListObject(ec, Statement.PS_MODEL);

			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS, accGradients);
		}

		@Override
		public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
			return null;
		}
	}

	// Statistics methods
	protected void accFedPSGradientWeighingTime(Timing time) {
		if (DMLScript.STATISTICS && time != null)
			Statistics.accFedPSGradientWeighingTime((long) time.stop());
	}

	@Override
	public String getWorkerName() {
		return String.format("Federated worker_%d", _workerID);
	}

	@Override
	protected void incWorkerNumber() {
		if (DMLScript.STATISTICS)
			Statistics.incWorkerNumber();
	}

	@Override
	protected void accLocalModelUpdateTime(Timing time) {
		if (DMLScript.STATISTICS && time != null)
			Statistics.accFedPSWorkerComputing((long) time.stop());
	}

	@Override
	protected void accBatchIndexingTime(Timing time) {
		if (DMLScript.STATISTICS && time != null)
			Statistics.accFedPSWorkerComputing((long) time.stop());
	}

	@Override
	protected void accGradientComputeTime(Timing time) {
		if (DMLScript.STATISTICS && time != null)
			Statistics.accFedPSWorkerComputing((long) time.stop());
	}
}
