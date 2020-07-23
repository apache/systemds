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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.DataIdentifier;
import org.apache.sysds.parser.Statement;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.FunctionProgramBlock;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.parfor.stat.Timing;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.FunctionCallCPInstruction;
import org.apache.sysds.runtime.instructions.cp.ListObject;
import org.apache.sysds.utils.Statistics;

public abstract class ParamServer 
{
	protected static final Log LOG = LogFactory.getLog(ParamServer.class.getName());
	protected static final boolean ACCRUE_BSP_GRADIENTS = true;
	
	// worker input queues and global model
	protected Map<Integer, BlockingQueue<ListObject>> _modelMap;
	private ListObject _model;

	//aggregation service
	protected ExecutionContext _ec;
	private Statement.PSUpdateType _updateType;
	private FunctionCallCPInstruction _inst;
	private String _outputName;
	private boolean[] _finishedStates;  // Workers' finished states
	private ListObject _accGradients = null;

	protected ParamServer() {}

	protected ParamServer(ListObject model, String aggFunc, Statement.PSUpdateType updateType, ExecutionContext ec, int workerNum) {
		// init worker queues and global model
		_modelMap = new HashMap<>(workerNum);
		IntStream.range(0, workerNum).forEach(i -> {
			// Create a single element blocking queue for workers to receive the broadcasted model
			_modelMap.put(i, new ArrayBlockingQueue<>(1));
		});
		_model = model;
		
		// init aggregation service
		_ec = ec;
		_updateType = updateType;
		_finishedStates = new boolean[workerNum];
		setupAggFunc(_ec, aggFunc);
		
		// broadcast initial model
		broadcastModel(true);
	}

	protected void setupAggFunc(ExecutionContext ec, String aggFunc) {
		String[] cfn = DMLProgram.splitFunctionKey(aggFunc);
		String ns = cfn[0];
		String fname = cfn[1];
		FunctionProgramBlock func = ec.getProgram().getFunctionProgramBlock(ns, fname, false);
		ArrayList<DataIdentifier> inputs = func.getInputParams();
		ArrayList<DataIdentifier> outputs = func.getOutputParams();

		// Check the output of the aggregation function
		if (outputs.size() != 1) {
			throw new DMLRuntimeException(String.format("The output of the '%s' function should provide one list containing the updated model.", aggFunc));
		}
		if (outputs.get(0).getDataType() != DataType.LIST) {
			throw new DMLRuntimeException(String.format("The output of the '%s' function should be type of list.", aggFunc));
		}
		_outputName = outputs.get(0).getName();

		CPOperand[] boundInputs = inputs.stream()
			.map(input -> new CPOperand(input.getName(), input.getValueType(), input.getDataType()))
			.toArray(CPOperand[]::new);
		ArrayList<String> outputNames = outputs.stream().map(DataIdentifier::getName)
			.collect(Collectors.toCollection(ArrayList::new));
		_inst = new FunctionCallCPInstruction(ns, fname, false, boundInputs,
			func.getInputParamNames(), outputNames, "aggregate function");
	}

	public abstract void push(int workerID, ListObject value);

	public abstract ListObject pull(int workerID);

	public ListObject getResult() {
		// All the model updating work has terminated,
		// so we could return directly the result model
		return _model;
	}
	
	protected synchronized void updateGlobalModel(int workerID, ListObject gradients) {
		try {
			if (LOG.isDebugEnabled()) {
				LOG.debug(String.format("Successfully pulled the gradients [size:%d kb] of worker_%d.",
					gradients.getDataSize() / 1024, workerID));
			}

			switch(_updateType) {
				case BSP: {
					setFinishedState(workerID);

					// Accumulate the intermediate gradients
					if( ACCRUE_BSP_GRADIENTS )
						_accGradients = ParamservUtils.accrueGradients(_accGradients, gradients, true);
					else
						updateGlobalModel(gradients);

					if (allFinished()) {
						// Update the global model with accrued gradients
						if( ACCRUE_BSP_GRADIENTS ) {
							updateGlobalModel(_accGradients);
							_accGradients = null;
						}
						
						// Broadcast the updated model
						resetFinishedStates();
						broadcastModel(true);
						if (LOG.isDebugEnabled())
							LOG.debug("Global parameter is broadcasted successfully.");
					}
					break;
				}
				case ASP: {
					updateGlobalModel(gradients);
					broadcastModel(workerID);
					break;
				}
				default:
					throw new DMLRuntimeException("Unsupported update: " + _updateType.name());
			}
		} 
		catch (Exception e) {
			throw new DMLRuntimeException("Aggregation service failed: ", e);
		}
	}

	private void updateGlobalModel(ListObject gradients) {
		Timing tAgg = DMLScript.STATISTICS ? new Timing(true) : null;
		_model = updateLocalModel(_ec, gradients, _model);
		if (DMLScript.STATISTICS)
			Statistics.accPSAggregationTime((long) tAgg.stop());
	}

	/**
	 * A service method for updating model with gradients
	 *
	 * @param ec execution context
	 * @param gradients list of gradients
	 * @param model old model
	 * @return new model
	 */
	protected ListObject updateLocalModel(ExecutionContext ec, ListObject gradients, ListObject model) {
		// Populate the variables table with the gradients and model
		ec.setVariable(Statement.PS_GRADIENTS, gradients);
		ec.setVariable(Statement.PS_MODEL, model);

		// Invoke the aggregate function
		_inst.processInstruction(ec);

		// Get the new model
		ListObject newModel = ec.getListObject(_outputName);

		// Clean up the list according to the data referencing status
		ParamservUtils.cleanupListObject(ec, Statement.PS_MODEL, newModel.getStatus());
		ParamservUtils.cleanupListObject(ec, Statement.PS_GRADIENTS);
		return newModel;
	}
	
	private boolean allFinished() {
		return !ArrayUtils.contains(_finishedStates, false);
	}

	private void resetFinishedStates() {
		Arrays.fill(_finishedStates, false);
	}

	private void setFinishedState(int workerID) {
		_finishedStates[workerID] = true;
	}

	/**
	 * Broadcast the model for all workers
	 */
	private void broadcastModel(boolean par) {
		IntStream stream = IntStream.range(0, _modelMap.size());
		(par ? stream.parallel() : stream).forEach(workerID -> {
			try {
				broadcastModel(workerID);
			} catch (InterruptedException e) {
				throw new DMLRuntimeException("Paramserv func: some error occurred when broadcasting model", e);
			}
		});
	}

	private void broadcastModel(int workerID) throws InterruptedException {
		Timing tBroad = DMLScript.STATISTICS ? new Timing(true) : null;

		//broadcast copy of model to specific worker, cleaned up by worker
		_modelMap.get(workerID).put(ParamservUtils.copyList(_model, false));

		if (DMLScript.STATISTICS)
			Statistics.accPSModelBroadcastTime((long) tBroad.stop());
	}
}
