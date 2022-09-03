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
import org.apache.sysds.runtime.controlprogram.federated.*;
import org.apache.sysds.runtime.controlprogram.federated.FederatedRequest.RequestType;
import org.apache.sysds.runtime.controlprogram.paramserv.homomorphicEncryption.PublicKey;
import org.apache.sysds.runtime.controlprogram.paramserv.homomorphicEncryption.SEALClient;
import org.apache.sysds.runtime.controlprogram.parfor.stat.Timing;
import org.apache.sysds.runtime.functionobjects.Multiply;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.instructions.cp.*;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.RightScalarOperator;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.util.ProgramConverter;
import org.apache.sysds.utils.stats.ParamServStatistics;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.concurrent.Callable;
import java.util.concurrent.Future;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.apache.sysds.runtime.util.ProgramConverter.*;

public class FederatedPSControlThread extends PSWorker implements Callable<Void> {
	private static final long serialVersionUID = 6846648059569648791L;
	protected static final Log LOG = LogFactory.getLog(ParamServer.class.getName());

	private FederatedData _featuresData;
	private FederatedData _labelsData;
	private final long _modelVarID;

	// runtime balancing
	private final PSRuntimeBalancing _runtimeBalancing;
	private int _numBatchesPerEpoch;
	private int _numBatchesPerNbatch;
	private int _possibleBatchesPerLocalEpoch;
	private final boolean _weighting;
	private double _weightingFactor = 1;
	private boolean _cycleStartAt0 = false;
	private boolean _use_homomorphic_encryption = false;
	private PublicKey _partial_public_key;

	public FederatedPSControlThread(int workerID, String updFunc, Statement.PSFrequency freq,
		PSRuntimeBalancing runtimeBalancing, boolean weighting, int epochs, long batchSize,
		int numBatchesPerGlobalEpoch, ExecutionContext ec, ParamServer ps, int nbatches, boolean modelAvg, boolean use_homomorphic_encryption)
	{
		super(workerID, updFunc, freq, epochs, batchSize, ec, ps, nbatches, modelAvg);

		_numBatchesPerEpoch = numBatchesPerGlobalEpoch;
		_runtimeBalancing = runtimeBalancing;
		_weighting = weighting && (!use_homomorphic_encryption); // FIXME: this disables weighting in favor of homomorphic encryption
		_numBatchesPerNbatch = nbatches;
		// generate the ID for the model
		_modelVarID = FederationUtils.getNextFedDataID();
		_modelAvg = _use_homomorphic_encryption || modelAvg; // we always have to use modelAvg when using homomorphic encryption
		_use_homomorphic_encryption = use_homomorphic_encryption;
	}

	/**
	 * Sets up the federated worker and control thread
	 *
	 * @param weightingFactor Gradients from this worker will be multiplied by this factor if weighting is enabled
	 */
	public void setup(double weightingFactor) {
		incWorkerNumber();
		if (_use_homomorphic_encryption) {
			((HEParamServer)_ps).registerThread(_workerID, this);
		}

		// prepare features and labels
		_featuresData = _features.getFedMapping().getFederatedData()[0];
		_labelsData = _labels.getFedMapping().getFederatedData()[0];

		// weighting factor is always set, but only used when weighting is specified
		_weightingFactor = weightingFactor;

		// different runtime balancing calculations
		long dataSize = _features.getNumRows();
		// calculate scaled batch size if balancing via batch size.
		// In some cases there will be some cycling
		if(_runtimeBalancing == PSRuntimeBalancing.SCALE_BATCH)
			_batchSize = (int) Math.ceil((double) dataSize / _numBatchesPerEpoch);

		// Calculate possible batches with batch size
		_possibleBatchesPerLocalEpoch = (int) Math.ceil((double) dataSize / _batchSize);

		// If no runtime balancing is specified, just run possible number of batches
		// WARNING: Will get stuck on miss match
		if(_runtimeBalancing == PSRuntimeBalancing.NONE)
			_numBatchesPerEpoch = _possibleBatchesPerLocalEpoch;

		// If running in baseline mode set cycle to false
		if(_runtimeBalancing == PSRuntimeBalancing.BASELINE)
			_cycleStartAt0 = true;

		if( LOG.isInfoEnabled() ) {
			LOG.info("Setup config for worker " + this.getWorkerName());
			LOG.info("Batch size: " + _batchSize + " possible batches: " + _possibleBatchesPerLocalEpoch
					+ " batches to run: " + _numBatchesPerEpoch + " weighting factor: " + _weightingFactor);
		}

		// serialize program
		// create program blocks for the instruction filtering
		String programSerialized;
		ArrayList<ProgramBlock> pbs = new ArrayList<>();

		BasicProgramBlock gradientProgramBlock = new BasicProgramBlock(_ec.getProgram());
		gradientProgramBlock.setInstructions(new ArrayList<>(Collections.singletonList(_inst)));
		pbs.add(gradientProgramBlock);

		if(_freq == PSFrequency.EPOCH || _freq == PSFrequency.NBATCHES) {
			BasicProgramBlock aggProgramBlock = new BasicProgramBlock(_ec.getProgram());
			aggProgramBlock.setInstructions(new ArrayList<>(Collections.singletonList(_ps.getAggInst())));
			pbs.add(aggProgramBlock);
		}

		programSerialized = InstructionUtils.concatStrings(
			PROG_BEGIN, NEWLINE,
			ProgramConverter.serializeProgram(_ec.getProgram(), pbs, new HashMap<>()),
			PROG_END);

		// write program and meta data to worker
		Future<FederatedResponse> udfResponse;

		final SetupFederatedWorker udf;
		if (_use_homomorphic_encryption) {
			byte[] a = ((HEParamServer)_ps).generateA();
			// generate pk[i] on each client and return it
			udf = new SetupHEFederatedWorker(a);
		} else {
			udf = new SetupFederatedWorker();
		}

		udf.setParams(_batchSize, dataSize, _possibleBatchesPerLocalEpoch,
				programSerialized, _inst.getNamespace(), _inst.getFunctionName(),
				_ps.getAggInst().getFunctionName(), _ec.getListObject("hyperparams"),
				_modelVarID, _nbatches, _use_homomorphic_encryption || _modelAvg);

		udfResponse = _featuresData.executeFederatedOperation(
				new FederatedRequest(RequestType.EXEC_UDF, _featuresData.getVarID(), udf));

		FederatedResponse response;
		try {
			response = udfResponse.get();
			if(!response.isSuccessful())
				throw new DMLRuntimeException("FederatedLocalPSThread: Setup UDF failed");

		}
		catch(Exception e) {
			throw new DMLRuntimeException("FederatedLocalPSThread: failed to execute Setup UDF" + e.getMessage());
		}
		if (_use_homomorphic_encryption) {
			try {
				_partial_public_key = (PublicKey) response.getData()[0];
			}
			catch (Exception e) {
				throw new DMLRuntimeException("FederatedLocalPSThread: HE Setup UDF didn't return an object");
			}
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
	 * Setup UDF executed on the federated worker
	 */
	private static class SetupFederatedWorker extends FederatedUDF {
		private static final long serialVersionUID = -3148991224792675607L;
		private long _batchSize;
		private long _dataSize;
		private int _possibleBatchesPerLocalEpoch;
		private String _programString;
		private String _namespace;
		private String _gradientsFunctionName;
		private String _aggregationFunctionName;
		private ListObject _hyperParams;
		private long _modelVarID;
		private boolean _modelAvg;
		private int _nbatches;
		private boolean _params_set = false;

		protected SetupFederatedWorker()
		{
			super(new long[]{});
		}

		public void setParams(long batchSize, long dataSize, int possibleBatchesPerLocalEpoch,
						 String programString, String namespace, String gradientsFunctionName, String aggregationFunctionName,
						 ListObject hyperParams, long modelVarID, int nbatches, boolean modelAvg) {
			_batchSize = batchSize;
			_dataSize = dataSize;
			_possibleBatchesPerLocalEpoch = possibleBatchesPerLocalEpoch;
			_programString = programString;
			_namespace = namespace;
			_gradientsFunctionName = gradientsFunctionName;
			_aggregationFunctionName = aggregationFunctionName;
			_hyperParams = hyperParams;
			_modelVarID = modelVarID;
			_modelAvg = modelAvg;
			_nbatches = nbatches;
			_params_set = true;
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			if (!_params_set) {
				return new FederatedResponse(FederatedResponse.ResponseType.ERROR, "params were not set");
			}

			// parse and set program
			ec.setProgram(ProgramConverter.parseProgram(_programString, 0));

			// set variables to ec
			ec.setVariable(Statement.PS_FED_BATCH_SIZE, new IntObject(_batchSize));
			ec.setVariable(Statement.PS_FED_DATA_SIZE, new IntObject(_dataSize));
			ec.setVariable(Statement.PS_FED_POSS_BATCHES_LOCAL, new IntObject(_possibleBatchesPerLocalEpoch));
			ec.setVariable(Statement.PS_FED_NAMESPACE, new StringObject(_namespace));
			ec.setVariable(Statement.PS_FED_GRADIENTS_FNAME, new StringObject(_gradientsFunctionName));
			ec.setVariable(Statement.PS_FED_AGGREGATION_FNAME, new StringObject(_aggregationFunctionName));
			ec.setVariable(Statement.PS_HYPER_PARAMS, _hyperParams);
			ec.setVariable(Statement.PS_FED_MODEL_VARID, new IntObject(_modelVarID));
			ec.setVariable(Statement.PS_NBATCHES, new IntObject(_nbatches));
			ec.setVariable(Statement.PS_MODELAVG, new BooleanObject(_modelAvg));

			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS);
		}

		@Override
		public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
			return null;
		}
	}

	private static class SetupHEFederatedWorker extends SetupFederatedWorker {
		private static final long serialVersionUID = 9128347291804980123L;

		byte[] _partial_pubkey_a;

		protected SetupHEFederatedWorker(byte[] partial_pubkey_a) {
			// delegate everything to parent class. set modelAvg to true, as it is the only supported case
			super();
			_partial_pubkey_a = partial_pubkey_a;
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			// TODO: set other CKKS parameters
			// TODO generate partial public key
			NativeHEHelper.initialize();

			SEALClient sc = new SEALClient(_partial_pubkey_a);
			ec.setSealClient(sc);
			PublicKey partial_pubkey = sc.generatePartialPublicKey();

			FederatedResponse res = super.execute(ec, data);
			if (!res.isSuccessful()) {
				return res;
			}

			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS, partial_pubkey);
		}
	}
	
	/**
	 * Teardown UDF executed on the federated worker
	 */
	private static class SetPublicKeyFederatedWorker extends FederatedUDF {
		private static final long serialVersionUID = -1536502123123318969L;
		private final PublicKey _public_key;

		protected SetPublicKeyFederatedWorker(PublicKey public_key) {
			super(new long[]{});
			_public_key = public_key;
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			ec.getSealClient().setPublicKey(_public_key);
			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS);
		}

		@Override
		public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
			return null;
		}
	}

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
				case NBATCHES:
					computeWithNBatchUpdates();
					break;
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

	protected void weightAndPushGradients(ListObject gradients) {
		assert (!(_weighting && _use_homomorphic_encryption)) : "weights and homomorphic encryption are not supported together";
		// scale gradients - must only include MatrixObjects
		if(_weighting && _weightingFactor != 1) {
			Timing tWeighting = DMLScript.STATISTICS ? new Timing(true) : null;
			gradients.getData().parallelStream().forEach((matrix) -> {
				MatrixObject matrixObject = (MatrixObject) matrix;
				MatrixBlock input = matrixObject.acquireReadAndRelease().scalarOperations(
					new RightScalarOperator(Multiply.getMultiplyFnObject(), _weightingFactor), new MatrixBlock());
				matrixObject.acquireModify(input);
				matrixObject.release();
			});
			accFedPSGradientWeightingTime(tWeighting);
		}
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

				Timing tAgg = DMLScript.STATISTICS ? new Timing(true) : null;
				if (_modelAvg && !_use_homomorphic_encryption)
					// we can't call the agg fn if we use HE, because it is implemented homomorphically in SEALServer::aggregateCiphertexts
					model = _ps.updateLocalModel(_ec, gradients, model);
				else
					ParamservUtils.cleanupListObject(model);
				weightAndPushGradients((_modelAvg && !_use_homomorphic_encryption) ? model : gradients);
				if (tAgg != null) {
					ParamServStatistics.accFedAggregation((long)tAgg.stop());
				}
				ParamservUtils.cleanupListObject(gradients);
			}
		}
	}

	/**
	 * Computes all epochs and updates after N batches
	 */
	protected void computeWithNBatchUpdates() {
		int numSetsPerEpocNbatches = (int) Math.ceil((double)_numBatchesPerEpoch / _numBatchesPerNbatch);
		for (int epochCounter = 0; epochCounter < _epochs; epochCounter++) {
			int currentLocalBatchNumber = (_cycleStartAt0) ? 0 : _numBatchesPerEpoch * epochCounter % _possibleBatchesPerLocalEpoch;

			for (int batchCounter = 0; batchCounter < numSetsPerEpocNbatches; batchCounter++) {
				int localStartBatchNum = getNextLocalBatchNum(currentLocalBatchNumber, numSetsPerEpocNbatches);
				currentLocalBatchNumber = currentLocalBatchNumber + _numBatchesPerNbatch;
				ListObject model = pullModel();
				ListObject gradients = computeGradientsForNBatches(model, _numBatchesPerNbatch, localStartBatchNum, true);

				Timing tAgg = DMLScript.STATISTICS ? new Timing(true) : null;
				weightAndPushGradients(gradients);
				if (tAgg != null) {
					ParamServStatistics.accFedAggregation((long)tAgg.stop());
				}

				ParamservUtils.cleanupListObject(model);
				ParamservUtils.cleanupListObject(gradients);
			}
		}
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

			Timing tAgg = DMLScript.STATISTICS ? new Timing(true) : null;
			weightAndPushGradients(gradients);
			if (tAgg != null) {
				ParamServStatistics.accFedAggregation((long)tAgg.stop());
			}

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
		Timing tFedCommunication = DMLScript.STATISTICS ? new Timing(true) : null;
		// put current model on federated worker
		Future<FederatedResponse> putParamsResponse = _featuresData.executeFederatedOperation(
			new FederatedRequest(RequestType.PUT_VAR, _modelVarID, model));

		try {
			if(!putParamsResponse.get().isSuccessful())
				throw new DMLRuntimeException("FederatedLocalPSThread: put was not successful");
		}
		catch(Exception e) {
			throw new DMLRuntimeException("FederatedLocalPSThread: failed to execute put" + e.getMessage());
		}

		// create and execute the udf on the remote worker
		Object udf;
		if (_use_homomorphic_encryption) {
			udf = new HEComputeGradientsForNBatches(new long[]{_featuresData.getVarID(), _labelsData.getVarID()},
					new long[]{_modelVarID}, numBatchesToCompute, localUpdate, localStartBatchNum);
		} else {
			udf = new federatedComputeGradientsForNBatches(new long[]{_featuresData.getVarID(), _labelsData.getVarID(),
					_modelVarID}, numBatchesToCompute, localUpdate, localStartBatchNum);
		}
		Future<FederatedResponse> udfResponse = _featuresData.executeFederatedOperation(
				new FederatedRequest(RequestType.EXEC_UDF, _featuresData.getVarID(), udf));

		try {
			Object[] responseData = udfResponse.get().getData();
			if(DMLScript.STATISTICS) {
				long total = (long) tFedCommunication.stop();
				long workerComputing = ((DoubleObject) responseData[1]).getLongValue();
				ParamServStatistics.accFedWorkerComputing(workerComputing);
				ParamServStatistics.accFedCommunicationTime(total - workerComputing);
				ParamServStatistics.accFedNetworkTime(total);
			}
			return (ListObject) responseData[0];
		}
		catch(Exception e) {
			if(DMLScript.STATISTICS)
				tFedCommunication.stop();
			throw new DMLRuntimeException("FederatedLocalPSThread: failed to execute UDF" + e.getMessage(), e);
		}
	}

	/**
	 * This is the code that will be executed on the federated Worker when computing one gradients for n batches
	 */
	private static class federatedComputeGradientsForNBatches extends FederatedUDF {
		private static final long serialVersionUID = -3075901536748794832L;
		int _numBatchesToCompute;
		boolean _localUpdate;
		int _localStartBatchNum;

		protected federatedComputeGradientsForNBatches(long[] inIDs, int numBatchesToCompute, boolean localUpdate, int localStartBatchNum) {
			super(inIDs);
			_numBatchesToCompute = numBatchesToCompute;
			_localUpdate = localUpdate;
			_localStartBatchNum = localStartBatchNum;
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			Timing tGradients = new Timing(true);
			// read in data by varid
			MatrixObject features = (MatrixObject) data[0];
			MatrixObject labels = (MatrixObject) data[1];
			ListObject model = (ListObject) data[2];

			// get data from execution context
			long batchSize = ((IntObject) ec.getVariable(Statement.PS_FED_BATCH_SIZE)).getLongValue();
			long dataSize = ((IntObject) ec.getVariable(Statement.PS_FED_DATA_SIZE)).getLongValue();
			int possibleBatchesPerLocalEpoch = (int) ((IntObject) ec.getVariable(Statement.PS_FED_POSS_BATCHES_LOCAL)).getLongValue();
			String namespace = ((StringObject) ec.getVariable(Statement.PS_FED_NAMESPACE)).getStringValue();
			String gradientsFunc = ((StringObject) ec.getVariable(Statement.PS_FED_GRADIENTS_FNAME)).getStringValue();
			String aggFunc = ((StringObject) ec.getVariable(Statement.PS_FED_AGGREGATION_FNAME)).getStringValue();
			boolean modelAvg = ((BooleanObject) ec.getVariable(Statement.PS_MODELAVG)).getBooleanValue();
			// recreate gradient instruction and output
			boolean opt = !ec.getProgram().containsFunctionProgramBlock(namespace, gradientsFunc, false);
			FunctionProgramBlock func = ec.getProgram().getFunctionProgramBlock(namespace, gradientsFunc, opt);
			ArrayList<DataIdentifier> inputs = func.getInputParams();
			ArrayList<DataIdentifier> outputs = func.getOutputParams();
			CPOperand[] boundInputs = inputs.stream()
					.map(input -> new CPOperand(input.getName(), input.getValueType(), input.getDataType()))
					.toArray(CPOperand[]::new);
			ArrayList<String> outputNames = outputs.stream().map(DataIdentifier::getName)
					.collect(Collectors.toCollection(ArrayList::new));
			Instruction gradientsInstruction = new FunctionCallCPInstruction(namespace, gradientsFunc,
					opt, boundInputs, func.getInputParamNames(), outputNames, "gradient function");
			DataIdentifier gradientsOutput = outputs.get(0);

			// recreate aggregation instruction and output if needed
			Instruction aggregationInstruction = null;
			DataIdentifier aggregationOutput = null;
			if(_localUpdate && _numBatchesToCompute > 1 | modelAvg) {
				func = ec.getProgram().getFunctionProgramBlock(namespace, aggFunc, opt);
				inputs = func.getInputParams();
				outputs = func.getOutputParams();
				boundInputs = inputs.stream()
						.map(input -> new CPOperand(input.getName(), input.getValueType(), input.getDataType()))
						.toArray(CPOperand[]::new);
				outputNames = outputs.stream().map(DataIdentifier::getName)
						.collect(Collectors.toCollection(ArrayList::new));
				aggregationInstruction = new FunctionCallCPInstruction(namespace, aggFunc,
						opt, boundInputs, func.getInputParamNames(), outputNames, "aggregation function");
				aggregationOutput = outputs.get(0);
			}
			ListObject accGradients = null;
			int currentLocalBatchNumber = _localStartBatchNum;
			// prepare execution context
			ec.setVariable(Statement.PS_MODEL, model);
			for(int batchCounter = 0; batchCounter < _numBatchesToCompute; batchCounter++) {
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
				accGradients = modelAvg ? null :
						ParamservUtils.accrueGradients(accGradients, gradients, false);

				// update the local model with gradients if needed
				// FIXME ensure that with modelAvg we always update the model
				// (current fails due to missing aggregation instruction)
				if(_localUpdate && (batchCounter < _numBatchesToCompute - 1 | modelAvg) ) {
					// Invoke the aggregate function
					aggregationInstruction.processInstruction(ec);
					// Get the new model
					model = ec.getListObject(aggregationOutput.getName());
					// Set new model in execution context
					ec.setVariable(Statement.PS_MODEL, model);
					// clean up gradients and result
					ParamservUtils.cleanupListObject(ec, aggregationOutput.getName());
				}

				// clean up
				ParamservUtils.cleanupData(ec, Statement.PS_FEATURES);
				ParamservUtils.cleanupData(ec, Statement.PS_LABELS);
			}

			// model clean up
			ParamservUtils.cleanupListObject(ec, ec.getVariable(Statement.PS_FED_MODEL_VARID).toString());
			// TODO double check cleanup gradients and models

			// stop timing
			DoubleObject gradientsTime = new DoubleObject(tGradients.stop());
			ParamServStatistics.accGradientComputeTime(gradientsTime.getLongValue());
			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS,
					new Object[]{modelAvg ? model : accGradients, gradientsTime});
		}

		@Override
		public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
			return null;
		}
	}


	/**
	 * This wraps federatedComputeGradientsForNBatches and adds encryption
	 */
	private static class HEComputeGradientsForNBatches extends federatedComputeGradientsForNBatches {
		private static final long serialVersionUID = -3535901512348794852L;
		private final long[] _deferredIds;

		protected HEComputeGradientsForNBatches(long[] deferredIds, long[] inIDs, int numBatchesToCompute, boolean localUpdate, int localStartBatchNum) {
			super(inIDs, numBatchesToCompute, localUpdate, localStartBatchNum);
			_deferredIds = deferredIds;
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data_without_deferred) {
			Timing tTotal = new Timing(true);
			// add features and gradients to data
			Data[] deferred_inputs = Arrays.stream(_deferredIds).mapToObj(id -> ec.getVariable(String.valueOf(id))).toArray(Data[]::new);
			Data[] data = Arrays.copyOf(deferred_inputs, deferred_inputs.length + data_without_deferred.length);
			System.arraycopy(data_without_deferred, 0, data, deferred_inputs.length, data_without_deferred.length);
			FederatedResponse res = super.execute(ec, data);

			if (!res.isSuccessful()) {
				return res;
			}

			// encrypt model with SEAL
			try {
				Timing tEncrypt = DMLScript.STATISTICS ? new Timing(true) : null;

				ListObject model = (ListObject) res.getData()[0];
				ListObject encrypted_model = new ListObject(model);
				IntStream.range(0, model.getLength()).forEach(matrix_idx -> {
					CiphertextMatrix encrypted_matrix = ec.getSealClient().encrypt((MatrixObject) model.getData(matrix_idx));
					encrypted_model.set(matrix_idx, encrypted_matrix);
				});

				// overwrite model with encryption
				res.getData()[0] = encrypted_model;

				if (tEncrypt != null) {
					ParamServStatistics.accHEEncryptionTime((long)tEncrypt.stop());
				}

				// stop timing
				DoubleObject gradientsTime = new DoubleObject(tTotal.stop());
				res.getData()[1] = gradientsTime;
			} catch (Exception e) {
				return new FederatedResponse(FederatedResponse.ResponseType.ERROR, new Object[] { e });
			}
			return res;
		}
	}

	private static class HEComputePartialDecryption extends FederatedUDF {
		private static final long serialVersionUID = -4535098129348794852L;
		private final CiphertextMatrix[] _encrypted_sum;

		protected HEComputePartialDecryption(CiphertextMatrix[] encrypted_sum) {
			super(new long[]{});
			_encrypted_sum = encrypted_sum;
		}

		@Override
		public FederatedResponse execute(ExecutionContext ec, Data... data) {
			Timing tPartialDecrypt = DMLScript.STATISTICS ? new Timing(true) : null;
			PlaintextMatrix[] result = new PlaintextMatrix[_encrypted_sum.length];
			IntStream.range(0, result.length).forEach(i -> {
				result[i] = ec.getSealClient().partiallyDecrypt(_encrypted_sum[i]);
			});
			if (tPartialDecrypt != null) {
				ParamServStatistics.accHEPartialDecryptionTime((long)tPartialDecrypt.stop());
			}
			return new FederatedResponse(FederatedResponse.ResponseType.SUCCESS, result);
		}

		@Override
		public Pair<String, LineageItem> getLineageItem(ExecutionContext ec) {
			return null;
		}
	}


	public PlaintextMatrix[] getPartialDecryption(CiphertextMatrix[] encrypted_sum) {
		Object udf = new HEComputePartialDecryption(encrypted_sum);
		Future<FederatedResponse> udfResponse = _featuresData.executeFederatedOperation(
				new FederatedRequest(RequestType.EXEC_UDF, _featuresData.getVarID(), udf));

		try {
			Object[] responseData = udfResponse.get().getData();
			return (PlaintextMatrix[]) responseData;
		} catch(Exception e) {
			throw new DMLRuntimeException("FederatedLocalPSThread: failed to execute UDF" + e.getMessage());
		}
	}

	// Statistics methods
	protected void accFedPSGradientWeightingTime(Timing time) {
		if (DMLScript.STATISTICS && time != null)
			ParamServStatistics.accFedGradientWeightingTime((long) time.stop());
	}

	@Override
	public String getWorkerName() {
		return String.format("Federated worker_%d", _workerID);
	}

	@Override
	protected void incWorkerNumber() {
		if (DMLScript.STATISTICS)
			ParamServStatistics.incWorkerNumber();
	}

	@Override
	protected void accLocalModelUpdateTime(Timing time) {
		throw new NotImplementedException();
	}

	@Override
	protected void accBatchIndexingTime(Timing time) {
		throw new NotImplementedException();
	}

	@Override
	protected void accGradientComputeTime(Timing time) {
		throw new NotImplementedException();
	}

	public PublicKey getPartialPublicKey() {
		return _partial_public_key;
	}

	public void setPublicKey(PublicKey public_key) {
		Future<FederatedResponse> res = _featuresData.executeFederatedOperation(
				new FederatedRequest(RequestType.EXEC_UDF, _featuresData.getVarID(),
						new SetPublicKeyFederatedWorker(public_key)));

		try {
			FederatedResponse response = res.get();
			if(!response.isSuccessful())
				throw new DMLRuntimeException("FederatedLocalPSThread: SetPublicKey UDF failed");

		}
		catch(Exception e) {
			throw new DMLRuntimeException("FederatedLocalPSThread: failed to execute Public Key Setup UDF" + e.getMessage());
		}
	}
}
