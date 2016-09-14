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
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.lops.Lop;
import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.parser.DataIdentifier;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLScriptException;
import org.apache.sysml.runtime.controlprogram.FunctionProgramBlock;
import org.apache.sysml.runtime.controlprogram.LocalVariableMap;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContextFactory;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.InstructionUtils;


/**
 * 
 */
public class FunctionCallCPInstruction extends CPInstruction 
{	
	private String _functionName;
	private String _namespace;
	
	public String getFunctionName(){
		return _functionName;
	}
	
	public String getNamespace() {
		return _namespace;
	}
	
	// stores both the bound input and output parameters
	private ArrayList<CPOperand> _boundInputParamOperands;
	private ArrayList<String> _boundInputParamNames;
	private ArrayList<String> _boundOutputParamNames;
	
	public FunctionCallCPInstruction(String namespace, String functName, ArrayList<CPOperand> boundInParamOperands, ArrayList<String> boundInParamNames, ArrayList<String> boundOutParamNames, String istr) {
		super(null, functName, istr);
		
		_cptype = CPINSTRUCTION_TYPE.External;
		_functionName = functName;
		_namespace = namespace;
		_boundInputParamOperands = boundInParamOperands;
		_boundInputParamNames = boundInParamNames;
		_boundOutputParamNames = boundOutParamNames;
		
	}
		
	/**
	 * 
	 */
	public static FunctionCallCPInstruction parseInstruction(String str) 
		throws DMLRuntimeException 
	{	
		//schema: extfunct, fname, num inputs, num outputs, inputs, outputs
		String[] parts = InstructionUtils.getInstructionPartsWithValueType ( str );
		String namespace = parts[1];
		String functionName = parts[2];
		int numInputs = Integer.valueOf(parts[3]);
		int numOutputs = Integer.valueOf(parts[4]);
		ArrayList<CPOperand> boundInParamOperands = new ArrayList<CPOperand>();
		ArrayList<String> boundInParamNames = new ArrayList<String>();
		ArrayList<String> boundOutParamNames = new ArrayList<String>();
		for (int i = 0; i < numInputs; i++) {
			CPOperand operand = new CPOperand(parts[5 + i]);
			boundInParamOperands.add(operand);
			boundInParamNames.add(operand.getName());
		}
		for (int i = 0; i < numOutputs; i++) {
			boundOutParamNames.add(parts[5 + numInputs + i]);
		}
		
		return new FunctionCallCPInstruction ( namespace,functionName, 
				boundInParamOperands, boundInParamNames, boundOutParamNames, str );
	}

		
	@Override
	public Instruction preprocessInstruction(ExecutionContext ec)
		throws DMLRuntimeException 
	{
		//default pre-process behavior
		Instruction tmp = super.preprocessInstruction(ec);
		
		//maintain debug state (function call stack) 
		if( DMLScript.ENABLE_DEBUG_MODE ) {
			ec.handleDebugFunctionEntry((FunctionCallCPInstruction) tmp);
		}

		return tmp;
	}

	@Override
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException 
	{		
		if( LOG.isTraceEnabled() ){
			LOG.trace("Executing instruction : " + this.toString());
		}
		
		// get the function program block (stored in the Program object)
		FunctionProgramBlock fpb = ec.getProgram().getFunctionProgramBlock(_namespace, _functionName);
		
		// create bindings to formal parameters for given function call
		// These are the bindings passed to the FunctionProgramBlock for function execution 
		LocalVariableMap functionVariables = new LocalVariableMap();		
		for( int i=0; i<fpb.getInputParams().size(); i++) 
		{				
			DataIdentifier currFormalParam = fpb.getInputParams().get(i);
			String currFormalParamName = currFormalParam.getName();
			Data currFormalParamValue = null; 
			ValueType valType = fpb.getInputParams().get(i).getValueType();
				
			// CASE (a): default values, if call w/ less params than signature (scalars only)
			if( i > _boundInputParamNames.size() )
			{	
				String defaultVal = fpb.getInputParams().get(i).getDefaultValue();
				currFormalParamValue = ec.getScalarInput(defaultVal, valType, false);
			}
			// CASE (b) literals or symbol table entries
			else {
				CPOperand operand = _boundInputParamOperands.get(i);
				String varname = operand.getName();
				//error handling non-existing variables
				if( !operand.isLiteral() && ec.containsVariable(varname) ) {
					throw new DMLRuntimeException("Input variable '"+varname+"' not existing on call of " + 
							DMLProgram.constructFunctionKey(_namespace, _functionName) + " (line "+getLineNum()+").");
				}
				//get input matrix/frame/scalar
				currFormalParamValue = (operand.getDataType()!=DataType.SCALAR) ? ec.getVariable(varname) : 
					ec.getScalarInput(varname, operand.getValueType(), operand.isLiteral());
			}
			
			functionVariables.put(currFormalParamName,currFormalParamValue);						
		}
		
		// Pin the input variables so that they do not get deleted 
		// from pb's symbol table at the end of execution of function
	    HashMap<String,Boolean> pinStatus = ec.pinVariables(_boundInputParamNames);
		
		// Create a symbol table under a new execution context for the function invocation,
		// and copy the function arguments into the created table. 
		ExecutionContext fn_ec = ExecutionContextFactory.createContext(false, ec.getProgram());
		fn_ec.setVariables(functionVariables);
		
		// execute the function block
		try {
			fpb.execute(fn_ec);
		}
		catch (DMLScriptException e) {
			throw e;
		}
		catch (Exception e){
			String fname = DMLProgram.constructFunctionKey(_namespace, _functionName);
			throw new DMLRuntimeException("error executing function " + fname, e);
		}
		
		LocalVariableMap retVars = fn_ec.getVariables();  
		
		// cleanup all returned variables w/o binding 
		Collection<String> retVarnames = new LinkedList<String>(retVars.keySet());
		HashSet<String> probeVars = new HashSet<String>();
		for(DataIdentifier di : fpb.getOutputParams())
			probeVars.add(di.getName());
		for( String var : retVarnames ) {
			if( !probeVars.contains(var) ) //cleanup candidate
			{
				Data dat = fn_ec.removeVariable(var);
				if( dat != null && dat instanceof MatrixObject )
					fn_ec.cleanupMatrixObject((MatrixObject)dat);
			}
		}
		
		// Unpin the pinned variables
		ec.unpinVariables(_boundInputParamNames, pinStatus);
		
		// add the updated binding for each return variable to the variables in original symbol table
		for (int i=0; i< fpb.getOutputParams().size(); i++){
		
			String boundVarName = _boundOutputParamNames.get(i); 
			Data boundValue = retVars.get(fpb.getOutputParams().get(i).getName());
			if (boundValue == null)
				throw new DMLRuntimeException(boundVarName + " was not assigned a return value");

			//cleanup existing data bound to output variable name
			Data exdata = ec.removeVariable(boundVarName);
			if ( exdata != null && exdata instanceof MatrixObject && exdata != boundValue ) {
				ec.cleanupMatrixObject( (MatrixObject)exdata );
			}
			
			//add/replace data in symbol table
			if( boundValue instanceof MatrixObject )
				((MatrixObject) boundValue).setVarName(boundVarName);
			ec.setVariable(boundVarName, boundValue);
		}
	}

	@Override
	public void postprocessInstruction(ExecutionContext ec)
		throws DMLRuntimeException 
	{
		//maintain debug state (function call stack) 
		if (DMLScript.ENABLE_DEBUG_MODE ) {
			ec.handleDebugFunctionExit( this );
		}
		
		//default post-process behavior
		super.postprocessInstruction(ec);
	}

	@Override
	public void printMe() {
		LOG.debug("ExternalBuiltInFunction: " + this.toString());
	}

	public String getGraphString() {
		return "ExtBuiltinFunc: " + _functionName;
	}
	
	public ArrayList<String> getBoundInputParamNames()
	{
		return _boundInputParamNames;
	}
	
	public ArrayList<String> getBoundOutputParamNames()
	{
		return _boundOutputParamNames;
	}
	
	/**
	 * 
	 * @param fname
	 */
	public void setFunctionName(String fname)
	{
		//update instruction string
		String oldfname = _functionName;
		instString = updateInstStringFunctionName(oldfname, fname);
		
		//set attribute
		_functionName = fname;
		instOpcode = fname;
	}

	/**
	 * 
	 * @param pattern
	 * @param replace
	 */
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
