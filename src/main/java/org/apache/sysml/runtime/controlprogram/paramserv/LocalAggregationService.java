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

package org.apache.sysml.runtime.controlprogram.paramserv;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.parser.DataIdentifier;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.FunctionProgramBlock;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.cp.Data;
import org.apache.sysml.runtime.instructions.cp.FunctionCallCPInstruction;
import org.apache.sysml.runtime.instructions.cp.ListObject;

public class LocalAggregationService extends AggregationService implements Runnable {

	protected static final Log LOG = LogFactory.getLog(LocalAggregationService.class.getName());
	private static final long POLL_FREQUENCY = 1;

	private String _funcNS;
	private String _funcName;
	private int _finished;
	private boolean[] _tables;
	private ArrayList<DataIdentifier> _inputs;
	private ArrayList<DataIdentifier> _outputs;
	private CPOperand[] _boundInputs;
	private ArrayList<String> _inputNames;
	private ArrayList<String> _outputNames;

	public LocalAggregationService(String aggFunc, ExecutionContext ec, ListObject models, ListObject hyperParams,
			ParamServer ps, int workerNum) {
		super(aggFunc, ec, models, hyperParams, ps, workerNum);

		// Populate the variables table with the hyperparams and global parameters
		ParamservUtils.populate(_ec, _hyperParams);
		ParamservUtils.populate(_ec, _models);

		// Fetch the agg function
		String[] keys = DMLProgram.splitFunctionKey(_aggFunc);
		_funcName = keys[0];
		if (keys.length == 2) {
			_funcNS = keys[0];
			_funcName = keys[1];
		}
		FunctionProgramBlock func = _ec.getProgram().getFunctionProgramBlock(_funcNS, _funcName);
		_inputs = func.getInputParams();
		_outputs = func.getOutputParams();
		_boundInputs = _inputs.stream()
				.map(input -> new CPOperand(input.getName(), input.getValueType(), input.getDataType()))
				.toArray(CPOperand[]::new);
		_inputNames = _inputs.stream().map(DataIdentifier::getName).collect(Collectors.toCollection(ArrayList::new));
		_outputNames = _outputs.stream().map(DataIdentifier::getName).collect(Collectors.toCollection(ArrayList::new));
	}

	@Override
	public void run() {

		// Push the global parameters
		pushGlobalParams();

		while (isAlive()) {
			// Pull the gradients from ps
			_finished = 0;
			_tables = new boolean[_workerNum];
			while (_finished < _workerNum) {
				IntStream.range(0, _workerNum).forEach(i -> {
					if (!_tables[i]) {
						//LOG.info(String.format("Try to pull the gradients of worker_%d.", i));
						Data gradients = _ps.pull(ParamServer.GRADIENTS_PREFIX + i);
						if (gradients != null) {
							LOG.info(String.format("Successfully pulled the gradients of worker_%d.", i));
							aggregate((ListObject) gradients);
							_finished++;
							_tables[i] = true;
						}
						try {
							TimeUnit.SECONDS.sleep(POLL_FREQUENCY);
						} catch (InterruptedException e) {
							throw new DMLRuntimeException(
									"Aggregation service: Failed to sleep when polling the gradients.", e);
						}
					}
				});
			}
			// Push the updated model to ps
			pushGlobalParams();
		}
	}

	private void aggregate(ListObject gradients) {

		// Populate the variables table with the gradients
		ParamservUtils.populate(_ec, gradients);

		// Invoke the aggregate function
		FunctionCallCPInstruction inst = new FunctionCallCPInstruction(_funcNS, _funcName, _boundInputs, _inputNames,
				_outputNames, "aggregate function");
		inst.processInstruction(_ec);

		// Get the output
		List<Data> paramsList = _outputs.stream().map(id -> _ec.getVariable(id.getName())).collect(Collectors.toList());
		ListObject output = new ListObject(paramsList, _outputNames);

		// Update the model
		List<Data> newData = _models.getNames().stream().map(output::slice).collect(Collectors.toList());
		_models = new ListObject(newData, _models.getNames());

		// Clean up the local execution context
		//_ec.destroy();
	}

	/**
	 * Push the global parameters to ps
	 */
	private void pushGlobalParams() {
		_ps.push(ParamServer.RESULT_MODEL, _models);
		// Pin the model to avoid being cleaned up
		_models.setStatus(_ec.pinVariables(_models.getNames()));
		IntStream.range(0, _workerNum).forEach(i -> _ps.push(ParamServer.GLOBAL_PREFIX + i, _models));
		LOG.info(String.format("Successfully pushed %d copies of global parameters to ps.", _workerNum));
	}
}
