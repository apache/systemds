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
		String[] keys = DMLProgram.splitFunctionKey(getUpdFunc());
		_funcName = keys[0];
		if (keys.length == 2) {
			_funcNS = keys[0];
			_funcName = keys[1];
		}
		FunctionProgramBlock func = getEC().getProgram().getFunctionProgramBlock(_funcNS, _funcName);
		_inputs = func.getInputParams();
		_outputs = func.getOutputParams();
		_boundInputs = _inputs.stream()
				.map(input -> new CPOperand(input.getName(), input.getValueType(), input.getDataType()))
				.toArray(CPOperand[]::new);
		_inputNames = _inputs.stream().map(DataIdentifier::getName).collect(Collectors.toCollection(ArrayList::new));
		_outputNames = _outputs.stream().map(DataIdentifier::getName).collect(Collectors.toCollection(ArrayList::new));

		ParamservUtils.populate(_ec, _hyperParams, _inputs);
	}

	@Override
	public void run() {
		int epochs = getEpochs();

		// Load training data
		MatrixObject features = getFeatures();
		MatrixObject labels = getLabels();
		long dataSize = features.getNumRows();

		for (int i = 0; i < epochs; i++) {
			long begin = 1;
			while (begin < dataSize) {
				// Pull the global parameters from ps
				ListObject globalParams = pollGlobalParams();
				ParamservUtils.populate(getEC(), globalParams);

				// Get batch features and labels
				long end = begin + getBatchSize();
				if (end > dataSize) {
					end = dataSize;
				}
				LOG.info(String.format("Local worker_%d: Try to get the batch data of index from %d to %d.", _workerID,
						begin, end));
				MatrixObject bFeatures = ParamservUtils.sliceMatrix(features, begin, end);
				MatrixObject bLabels = ParamservUtils.sliceMatrix(labels, begin, end);
				begin = end + 1;

				getEC().setVariable(Statement.PS_FEATURES, bFeatures);
				getEC().setVariable(Statement.PS_LABELS, bLabels);

				// Invoke the update function
				FunctionCallCPInstruction inst = new FunctionCallCPInstruction(_funcNS, _funcName, _boundInputs,
						_inputNames, _outputNames, "update function");
				inst.processInstruction(getEC());

				// Get the output
				List<Data> gradientsList = _outputs.stream().map(id -> getEC().getVariable(id.getName()))
						.collect(Collectors.toList());
				ListObject gradients = new ListObject(gradientsList, _outputNames);

				// Push the gradients to ps
				pushGradients(gradients);

			}
			LOG.info(String.format("Local worker_%d: Finished %d epoch.", _workerID, i));
		}
		LOG.info(String.format("Local worker_%d: Job finished.", _workerID));
	}

	private void pushGradients(ListObject gradients) {
		// Pin the output to avoiding being cleaned up
		gradients.setStatus(_ec.pinVariables(gradients.getNames()));
		getPS().push(ParamServer.GRADIENTS_PREFIX + _workerID, gradients);
		LOG.info(String.format("Local worker_%d: Successfully push the gradients to ps.", _workerID));
	}

	private ListObject pollGlobalParams() {
		ListObject globalParams = null;
		while (globalParams == null) {
			globalParams = (ListObject) getPS().pull(ParamServer.GLOBAL_PREFIX + _workerID);
			try {
				TimeUnit.SECONDS.sleep(POLL_FREQUENCY);
			} catch (InterruptedException e) {
				throw new DMLRuntimeException(
						String.format("Local worker_%d: Failed to pull the global parameters.", _workerID), e);
			}
		}
		LOG.info(String.format("Local worker_%d: Successfully pull the global parameters from ps.", _workerID));
		return globalParams;
	}
}
