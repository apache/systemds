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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.parser.DataIdentifier;
import org.apache.sysml.parser.Statement;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.FunctionProgramBlock;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.cp.Data;
import org.apache.sysml.runtime.instructions.cp.FunctionCallCPInstruction;
import org.apache.sysml.runtime.instructions.cp.ListObject;

public class LocalPSWorker extends PSWorker implements Runnable {

	protected static final Log LOG = LogFactory.getLog(LocalPSWorker.class.getName());
	private static final int POLL_FREQUENCY = 1;

	private String _funcNS;
	private String _funcName;
	private ArrayList<DataIdentifier> _inputs;
	private ArrayList<DataIdentifier> _outputs;
	private CPOperand[] _boundInputs;
	private ArrayList<String> _inputNames;
	private ArrayList<String> _outputNames;

	public LocalPSWorker(long workerID, String updFunc, Statement.PSFrequency freq, int epochs, long batchSize,
			ListObject hyperParams, ExecutionContext ec, ParamServer ps) {
		super(workerID, updFunc, freq, epochs, batchSize, hyperParams, ec, ps);

		// Get the update function
		String[] keys = DMLProgram.splitFunctionKey(_updFunc);
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

		ParamservUtils.populate(_ec, _hyperParams, _inputs);

		long dataSize = _features.getNumRows();

		for (int i = 0; i < _epochs; i++) {
			int totalIter = (int) Math.ceil(dataSize / _batchSize);
			for (int j = 0; j < totalIter; j++) {
				// Pull the global parameters from ps
				ListObject globalParams = pollGlobalParams();
				ParamservUtils.populate(_ec, globalParams, _inputs);

				long begin = j * _batchSize + 1;
				long end = begin + _batchSize;
				if (j == totalIter - 1) {
					// If it is the last iteration
					end = dataSize;
				}

				// Get batch features and labels
				MatrixObject bFeatures = ParamservUtils.sliceMatrix(_features, begin, end);
				MatrixObject bLabels = ParamservUtils.sliceMatrix(_labels, begin, end);
				_ec.setVariable(Statement.PS_FEATURES, bFeatures);
				_ec.setVariable(Statement.PS_LABELS, bLabels);

				LOG.info(String.format(
						"Local worker_%d: Got batch data [size:%d kb] of index from %d to %d. [Epoch:%d  Total epoch:%d  Iteration:%d  Total iteration:%d]",
						_workerID, bFeatures.getDataSize() / 1024 + bLabels.getDataSize() / 1024, begin, end, i + 1,
						_epochs, j + 1, totalIter));

				// Invoke the update function
				FunctionCallCPInstruction inst = new FunctionCallCPInstruction(_funcNS, _funcName, _boundInputs,
						_inputNames, _outputNames, "update function");
				inst.processInstruction(_ec);

				// Get the gradients
				List<Data> gradientsList = _outputs.stream().map(id -> _ec.getVariable(id.getName()))
						.collect(Collectors.toList());
				ListObject gradients = new ListObject(gradientsList, _outputNames);

				// Pin the gradients
				gradients.setStatus(_ec.pinVariables(gradients.getNames()));

				// Push the gradients to ps
				pushGradients(gradients);

				ParamservUtils.cleanupListObject(_ec, globalParams);
				ParamservUtils.cleanupListObject(_ec, gradients);
				ParamservUtils.cleanupData(bFeatures);
				ParamservUtils.cleanupData(bLabels);
			}
			LOG.info(String.format("Local worker_%d: Finished %d epoch.", _workerID, i + 1));
		}

		LOG.info(String.format("Local worker_%d: Job finished.", _workerID));
		ParamservUtils.cleanupListObject(_ec, _hyperParams);
		_ec.getVariables().removeAll();
	}

	private void pushGradients(ListObject gradients) {
		_ps.push(ParamServer.GRADIENTS_PREFIX + _workerID, ParamservUtils.copyList(gradients));
		LOG.info(String.format("Local worker_%d: Successfully push the gradients [size:%d kb] to ps.", _workerID,
				gradients.getDataSize() / 1024));
	}

	private ListObject pollGlobalParams() {
		ListObject globalParams = null;
		while (globalParams == null) {
			globalParams = (ListObject) _ps.pull(ParamServer.GLOBAL_PREFIX + _workerID);
			try {
				TimeUnit.SECONDS.sleep(POLL_FREQUENCY);
			} catch (InterruptedException e) {
				throw new DMLRuntimeException(
						String.format("Local worker_%d: Failed to pull the global parameters.", _workerID), e);
			}
		}
		LOG.info(String.format("Local worker_%d: Successfully pull the global parameters [size:%d kb] from ps.",
				_workerID, globalParams.getDataSize() / 1024));
		return globalParams;
	}
}
