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

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.lops.Lop;
import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.parser.DataIdentifier;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLScriptException;
import org.apache.sysml.runtime.controlprogram.FunctionProgramBlock;
import org.apache.sysml.runtime.controlprogram.LocalVariableMap;
import org.apache.sysml.runtime.controlprogram.caching.CacheableData;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.InstructionUtils;

public class FunctionCallCPInstruction extends CPInstruction {
	private final String _functionName;
	private final String _namespace;
	private final CPOperand[] _boundInputs;
	private final ArrayList<String> _boundInputNames;
	private final ArrayList<String> _boundOutputNames;

	public FunctionCallCPInstruction(String namespace, String functName, CPOperand[] boundInputs,
		ArrayList<String> boundInputNames, ArrayList<String> boundOutputNames, String istr) {
		super(CPType.External, null, functName, istr);
		_functionName = functName;
		_namespace = namespace;
		_boundInputs = boundInputs;
		_boundInputNames = boundInputNames;
		_boundOutputNames = boundOutputNames;
	}

	public String getFunctionName() {
		return _functionName;
	}

	public String getNamespace() {
		return _namespace;
	}
	
	public static FunctionCallCPInstruction parseInstruction(String str) {
		//schema: extfunct, fname, num inputs, num outputs, inputs, outputs
		String[] parts = InstructionUtils.getInstructionPartsWithValueType ( str );
		String namespace = parts[1];
		String functionName = parts[2];
		int numInputs = Integer.valueOf(parts[3]);
		int numOutputs = Integer.valueOf(parts[4]);
		CPOperand[] boundInputs = new CPOperand[numInputs];
		ArrayList<String> boundInputNames = new ArrayList<>();
		ArrayList<String> boundOutputNames = new ArrayList<>();
		for (int i = 0; i < numInputs; i++) {
			boundInputs[i] = new CPOperand(parts[5 + i]);
			boundInputNames.add(boundInputs[i].getName());
		}
		for (int i = 0; i < numOutputs; i++)
			boundOutputNames.add(parts[5 + numInputs + i]);
		return new FunctionCallCPInstruction ( namespace,
			functionName, boundInputs, boundInputNames, boundOutputNames, str );
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
		for( int i=0; i<fpb.getInputParams().size(); i++) 
		{
			//error handling non-existing variables
			CPOperand input = _boundInputs[i];
			if( !input.isLiteral() && !ec.containsVariable(input.getName()) ) {
				throw new DMLRuntimeException("Input variable '"+input.getName()+"' not existing on call of " + 
					DMLProgram.constructFunctionKey(_namespace, _functionName) + " (line "+getLineNum()+").");
			}
			//get input matrix/frame/scalar
			DataIdentifier currFormalParam = fpb.getInputParams().get(i);
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
		if (DMLScript.USE_ACCELERATOR) {
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
			Data var = fn_ec.removeVariable(varName);
			if( var instanceof CacheableData )
				fn_ec.cleanupCacheableData((CacheableData<?>)var);
		}
		
		// Unpin the pinned variables
		ec.unpinVariables(_boundInputNames, pinStatus);

		// add the updated binding for each return variable to the variables in original symbol table
		for (int i=0; i< fpb.getOutputParams().size(); i++){
			String boundVarName = _boundOutputNames.get(i);
			Data boundValue = retVars.get(fpb.getOutputParams().get(i).getName());
			if (boundValue == null)
				throw new DMLRuntimeException(boundVarName + " was not assigned a return value");

			//cleanup existing data bound to output variable name
			Data exdata = ec.removeVariable(boundVarName);
			if ( exdata != null && exdata instanceof CacheableData && exdata != boundValue ) {
				ec.cleanupCacheableData( (CacheableData<?>)exdata );
			}
			
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

	public ArrayList<String> getBoundInputParamNames() {
		return _boundInputNames;
	}
	
	public ArrayList<String> getBoundOutputParamNames() {
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
