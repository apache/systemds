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
import java.util.stream.Collectors;

import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.parser.DataIdentifier;
import org.apache.sysml.parser.Expression;
import org.apache.sysml.parser.Statement;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.FunctionProgramBlock;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.instructions.cp.FunctionCallCPInstruction;
import org.apache.sysml.runtime.instructions.cp.ListObject;

@SuppressWarnings("unused")
public abstract class PSWorker {

	long _workerID = -1;
	int _epochs;
	long _batchSize;
	MatrixObject _features;
	MatrixObject _labels;
	ExecutionContext _ec;
	ParamServer _ps;
	private String _updFunc;
	private Statement.PSFrequency _freq;
	private MatrixObject _valFeatures;
	private MatrixObject _valLabels;

	DataIdentifier _output;
	FunctionCallCPInstruction _inst;

	public PSWorker(long workerID, String updFunc, Statement.PSFrequency freq, int epochs, long batchSize,
			ListObject hyperParams, ExecutionContext ec, ParamServer ps) {
		this._workerID = workerID;
		this._updFunc = updFunc;
		this._freq = freq;
		this._epochs = epochs;
		this._batchSize = batchSize;
		this._ec = ExecutionContextFactory.createContext(ec.getProgram());
		if (hyperParams != null) {
			this._ec.setVariable(Statement.PS_HYPER_PARAMS, hyperParams);
		}
		this._ps = ps;

		// Get the update function
		String[] keys = DMLProgram.splitFunctionKey(updFunc);
		String funcName = keys[0];
		String funcNS = null;
		if (keys.length == 2) {
			funcNS = keys[0];
			funcName = keys[1];
		}
		FunctionProgramBlock func = ec.getProgram().getFunctionProgramBlock(funcNS, funcName);
		ArrayList<DataIdentifier> inputs = func.getInputParams();
		ArrayList<DataIdentifier> outputs = func.getOutputParams();
		CPOperand[] boundInputs = inputs.stream()
				.map(input -> new CPOperand(input.getName(), input.getValueType(), input.getDataType()))
				.toArray(CPOperand[]::new);
		ArrayList<String> inputNames = inputs.stream().map(DataIdentifier::getName)
				.collect(Collectors.toCollection(ArrayList::new));
		ArrayList<String> outputNames = outputs.stream().map(DataIdentifier::getName)
				.collect(Collectors.toCollection(ArrayList::new));
		_inst = new FunctionCallCPInstruction(funcNS, funcName, boundInputs, inputNames, outputNames,
				"update function");

		// Check the inputs of the update function
		checkInput(inputs, Expression.DataType.MATRIX, Statement.PS_FEATURES);
		checkInput(inputs, Expression.DataType.MATRIX, Statement.PS_LABELS);
		checkInput(inputs, Expression.DataType.LIST, Statement.PS_MODEL);
		if (hyperParams != null) {
			checkInput(inputs, Expression.DataType.LIST, Statement.PS_HYPER_PARAMS);
		}

		// Check the output of the update function
		if (outputs.size() != 1) {
			throw new DMLRuntimeException(
				String.format("The output of the '%s' function should provide one list containing the gradients.", updFunc));
		}
		if (outputs.get(0).getDataType() != Expression.DataType.LIST) {
			throw new DMLRuntimeException(
					String.format("The output of the '%s' function should be type of list.", updFunc));
		}
		_output = outputs.get(0);
	}

	private void checkInput(ArrayList<DataIdentifier> inputs, Expression.DataType dt, String pname) {
		if (inputs.stream().filter(input -> input.getDataType() == dt && pname.equals(input.getName())).count() != 1) {
			throw new DMLRuntimeException(
				String.format("The '%s' function should provide an input of '%s' type named '%s'.", _updFunc, dt, pname));
		}
	}

	public void setFeatures(MatrixObject features) {
		this._features = features;
	}

	public void setLabels(MatrixObject labels) {
		this._labels = labels;
	}

	public void setValFeatures(MatrixObject valFeatures) {
		this._valFeatures = valFeatures;
	}

	public void setValLabels(MatrixObject valLabels) {
		this._valLabels = valLabels;
	}
}
