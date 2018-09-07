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

package org.apache.sysml.runtime.controlprogram;

import java.util.ArrayList;
import java.util.HashMap;

import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.parser.DataIdentifier;
import org.apache.sysml.parser.ExternalFunctionStatement;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.instructions.cp.CPOperand;
import org.apache.sysml.runtime.matrix.data.InputInfo;
import org.apache.sysml.udf.ExternalFunctionInvocationInstruction;
import org.apache.sysml.udf.PackageFunction;

/**
 * CP external function program block, that overcomes the need for 
 * BlockToCell and CellToBlock MR jobs by changing the contract for an external function.
 * If execlocation="CP", the implementation of an external function must read and write
 * matrices as InputInfo.BinaryBlockInputInfo and OutputInfo.BinaryBlockOutputInfo.
 * 
 * Furthermore, it extends ExternalFunctionProgramBlock with a base directory in order
 * to make it parallelizable, even in case of different JVMs. For this purpose every
 * external function must implement a &lt;SET_BASE_DIR&gt; method. 
 * 
 *
 */
public class ExternalFunctionProgramBlockCP extends ExternalFunctionProgramBlock 
{
	/**
	 * Constructor that also provides otherParams that are needed for external
	 * functions. Remaining parameters will just be passed to constructor for
	 * function program block.
	 * 
	 * @param prog runtime program
	 * @param inputParams list of input data identifiers
	 * @param outputParams list of output data identifiers
	 * @param otherParams map of other parameters
	 * @param baseDir base directory
	 */
	public ExternalFunctionProgramBlockCP(Program prog,
			ArrayList<DataIdentifier> inputParams,
			ArrayList<DataIdentifier> outputParams,
			HashMap<String, String> otherParams,
			String baseDir) {

		super(prog, inputParams, outputParams, baseDir); //w/o instruction generation
		
		// copy other params 
		_otherParams = new HashMap<>();
		_otherParams.putAll(otherParams);

		// generate instructions (overwritten)
		createInstructions();
	}

	/**
	 * Method to be invoked to execute instructions for the external function
	 * invocation
	 */
	@Override
	public void execute(ExecutionContext ec) 
	{
		if( _inst.size() != 1 )
			throw new DMLRuntimeException("Invalid number of instructions: "+_inst.size());
		
		// execute package function via ExternalFunctionInvocationInstruction
		try {
			 _inst.get(0).processInstruction(ec);
		}
		catch (Exception e){
			throw new DMLRuntimeException(printBlockErrorLocation() + "Error evaluating external function: "
				+ DMLProgram.constructFunctionKey(_namespace, _functionName), e);
		}
		
		// check return values
		checkOutputParameters(ec.getVariables());
	}

	@Override
	protected void createInstructions() 
	{
		_inst = new ArrayList<>();

		// assemble information provided through keyvalue pairs
		String className = _otherParams.get(ExternalFunctionStatement.CLASS_NAME);
		String configFile = _otherParams.get(ExternalFunctionStatement.CONFIG_FILE);
		
		// class name cannot be null, however, configFile and execLocation can be null
		if (className == null)
			throw new RuntimeException(this.printBlockErrorLocation() + ExternalFunctionStatement.CLASS_NAME + " not provided!");

		// assemble input and output param strings
		CPOperand[] inputs = getOperands(getInputParams());
		CPOperand[] outputs = getOperands(getOutputParams());
		
		// generate instruction
		PackageFunction fun = createFunctionObject(className, configFile);
		ExternalFunctionInvocationInstruction einst = 
			new ExternalFunctionInvocationInstruction(inputs, outputs, fun, _baseDir, InputInfo.BinaryBlockInputInfo);
		verifyFunctionInputsOutputs(fun, inputs, outputs);
		
		_inst.add(einst);
	}

	@Override
	public String printBlockErrorLocation(){
		return "ERROR: Runtime error in external function program block (for CP) generated from external function statement block between lines " + _beginLine + " and " + _endLine + " -- ";
	}
}