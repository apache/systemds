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

package org.apache.sysml.runtime.instructions.cp;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.lops.Lop;
import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.parser.DataIdentifier;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLScriptException;
import org.apache.sysml.runtime.controlprogram.FunctionProgramBlock;
import org.apache.sysml.runtime.controlprogram.LocalVariableMap;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.InstructionUtils;
import org.apache.sysml.runtime.io.IOUtilFunctions;

public class FunctionCallCPInstruction extends CPInstruction {
	private final String _functionName;
	private final String _namespace;
	private final CPOperand[] _boundInputs;
	private final List<String> _boundInputNames;
	private final List<String> _funArgNames;
	private final List<String> _boundOutputNames;

	public FunctionCallCPInstruction(String namespace, String functName, CPOperand[] boundInputs,
			List<String> boundInputNames, List<String> funArgNames, List<String> boundOutputNames, String istr) {
		super(CPType.External, null, functName, istr);
		_functionName = functName;
		_namespace = namespace;
		_boundInputs = boundInputs;
		_boundInputNames = boundInputNames;
		_funArgNames = funArgNames;
		_boundOutputNames = boundOutputNames;
	}

	public String getFunctionName() {
		return _functionName;
	}

	public String getNamespace() {
		return _namespace;
	}
	
	public static FunctionCallCPInstruction parseInstruction(String str) {
		//schema: extfunct, fname, num inputs, num outputs, inputs (name-value pairs), outputs
		String[] parts = InstructionUtils.getInstructionPartsWithValueType ( str );
		String namespace = parts[1];
		String functionName = parts[2];
		int numInputs = Integer.valueOf(parts[3]);
		int numOutputs = Integer.valueOf(parts[4]);
		CPOperand[] boundInputs = new CPOperand[numInputs];
		List<String> boundInputNames = new ArrayList<>();
		List<String> funArgNames = new ArrayList<>();
		List<String> boundOutputNames = new ArrayList<>();
		for (int i = 0; i < numInputs; i++) {
			String[] nameValue = IOUtilFunctions.splitByFirst(parts[5 + i], "=");
			boundInputs[i] = new CPOperand(nameValue[1]);
			funArgNames.add(nameValue[0]);
			boundInputNames.add(boundInputs[i].getName());
		}
		for (int i = 0; i < numOutputs; i++)
			boundOutputNames.add(parts[5 + numInputs + i]);
		return new FunctionCallCPInstruction ( namespace, functionName,
			boundInputs, boundInputNames, funArgNames, boundOutputNames, str );
	}
	
	@Override
	public Instruction preprocessInstruction(ExecutionContext ec) {
		//default pre-process behavior
		Instruction tmp = super.preprocessInstruction(ec);
		//maintain debug state (function call stack) 
		if( DMLScript.ENABLE_DEBUG_MODE )
			ec.handleDebugFunctionEntry((FunctionCallCPInstruction) tmp);
		return tmp;
	}

	@Override
	public void processInstruction(ExecutionContext ec) {
		if( LOG.isTraceEnabled() ){
			LOG.trace("Executing instruction : " + this.toString());
		}
		// get the function program block (stored in the Program object)
		FunctionProgramBlock fpb = ec.getProgram().getFunctionProgramBlock(_namespace, _functionName);
		
		// sanity check number of function parameters
		if( _boundInputs.length < fpb.getInputParams().size() ) {
			throw new DMLRuntimeException("Number of bound input parameters does not match the function signature "
				+ "("+_boundInputs.length+", but "+fpb.getInputParams().size()+" expected)");
		}
		
		// create bindings to formal parameters for given function call
		// These are the bindings passed to the FunctionProgramBlock for function execution 
		LocalVariableMap functionVariables = new LocalVariableMap();
		for( int i=0; i<_boundInputs.length; i++) {
			//error handling non-existing variables
			CPOperand input = _boundInputs[i];
			if( !input.isLiteral() && !ec.containsVariable(input.getName()) ) {
				throw new DMLRuntimeException("Input variable '"+input.getName()+"' not existing on call of " + 
					DMLProgram.constructFunctionKey(_namespace, _functionName) + " (line "+getLineNum()+").");
			}
			//get input matrix/frame/scalar
			String argName = _funArgNames.get(i);
			DataIdentifier currFormalParam = fpb.getInputParam(argName);
			if( currFormalParam == null ) {
				throw new DMLRuntimeException("Non-existing named "
					+ "function argument: '"+argName+"' (line "+getLineNum()+").");
			}
			
			Data value = ec.getVariable(input);
			
			//graceful value type conversion for scalar inputs with wrong type
			if( value.getDataType() == DataType.SCALAR
				&& value.getValueType() != currFormalParam.getValueType() ) 
			{
				value = ScalarObjectFactory.createScalarObject(
					currFormalParam.getValueType(), (ScalarObject)value);
			}
			
			//set input parameter
			functionVariables.put(currFormalParam.getName(), value);
		}
		
		// Pin the input variables so that they do not get deleted 
		// from pb's symbol table at the end of execution of function
		boolean[] pinStatus = ec.pinVariables(_boundInputNames);
		
		// Create a symbol table under a new execution context for the function invocation,
		// and copy the function arguments into the created table. 
		ExecutionContext fn_ec = ExecutionContextFactory.createContext(false, ec.getProgram());
		if (ConfigurationManager.isGPU()) {
			fn_ec.setGPUContexts(ec.getGPUContexts());
			fn_ec.getGPUContext(0).initializeThread();
		}
		fn_ec.setVariables(functionVariables);
		// execute the function block
		try {
			fpb._functionName = this._functionName;
			fpb._namespace = this._namespace;
			fpb.execute(fn_ec);
		}
		catch (DMLScriptException e) {
			throw e;
		}
		catch (Exception e){
			String fname = DMLProgram.constructFunctionKey(_namespace, _functionName);
			throw new DMLRuntimeException("error executing function " + fname, e);
		}
		
		// cleanup all returned variables w/o binding 
		HashSet<String> expectRetVars = new HashSet<>();
		for(DataIdentifier di : fpb.getOutputParams())
			expectRetVars.add(di.getName());
		
		LocalVariableMap retVars = fn_ec.getVariables();
		for( String varName : new ArrayList<>(retVars.keySet()) ) {
			if( expectRetVars.contains(varName) )
				continue;
			//cleanup unexpected return values to avoid leaks
			fn_ec.cleanupDataObject(fn_ec.removeVariable(varName));
		}
		
		// Unpin the pinned variables
		ec.unpinVariables(_boundInputNames, pinStatus);

		// add the updated binding for each return variable to the variables in original symbol table
		// (with robustness for unbound outputs, i.e., function calls without assignment)
		int numOutputs = Math.min(_boundOutputNames.size(), fpb.getOutputParams().size());
		for (int i=0; i< numOutputs; i++) {
			String boundVarName = _boundOutputNames.get(i);
			Data boundValue = retVars.get(fpb.getOutputParams().get(i).getName());
			if (boundValue == null)
				throw new DMLRuntimeException(boundVarName + " was not assigned a return value");

			//cleanup existing data bound to output variable name
			Data exdata = ec.removeVariable(boundVarName);
			if( exdata != boundValue )
				ec.cleanupDataObject(exdata);
			
			//add/replace data in symbol table
			ec.setVariable(boundVarName, boundValue);
		}
	}

	@Override
	public void postprocessInstruction(ExecutionContext ec) {
		//maintain debug state (function call stack) 
		if (DMLScript.ENABLE_DEBUG_MODE )
			ec.handleDebugFunctionExit( this );
		//default post-process behavior
		super.postprocessInstruction(ec);
	}

	@Override
	public void printMe() {
		LOG.debug("ExternalBuiltInFunction: " + this.toString());
	}
	
	public List<String> getBoundOutputParamNames() {
		return _boundOutputNames;
	}

	public String updateInstStringFunctionName(String pattern, String replace)
	{
		//split current instruction
		String[] parts = instString.split(Lop.OPERAND_DELIMITOR);
		if( parts[3].equals(pattern) )
			parts[3] = replace;	
		
		//construct and set modified instruction
		StringBuilder sb = new StringBuilder();
		for( String part : parts ) {
			sb.append(part);
			sb.append(Lop.OPERAND_DELIMITOR);
		}

		return sb.substring( 0, sb.length()-Lop.OPERAND_DELIMITOR.length() );
	}
}
