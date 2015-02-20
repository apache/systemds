/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.instructions.cp;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;

import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.parser.DataIdentifier;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLScriptException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.FunctionProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.LocalVariableMap;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.util.UtilFunctions;


/**
 * 
 */
public class FunctionCallCPInstruction extends CPInstruction 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private String _functionName;
	private String _namespace;
	
	//private CPOperand _output;
	
	public String getFunctionName(){
		return _functionName;
	}
	
	public String getNamespace() {
		return _namespace;
	}
	
	// stores both the bound input and output parameters
	private ArrayList<String> _boundInputParamNames;
	private ArrayList<String> _boundOutputParamNames;
	
	public FunctionCallCPInstruction(String namespace, String functName, ArrayList<String> boundInParamNames, ArrayList<String> boundOutParamNames, String istr) {
		super(null, functName, istr);
		
		_cptype = CPINSTRUCTION_TYPE.External;
		_functionName = functName;
		_namespace = namespace;
		_boundInputParamNames = boundInParamNames;
		_boundOutputParamNames = boundOutParamNames;
		
	}
		
	/**
	 * Instruction format extFunct:::[FUNCTION NAME]:::[num input params]:::[num output params]:::[list of delimited input params ]:::[list of delimited ouput params]
	 * These are the "bound names" for the inputs / outputs.  For example, out1 = foo(in1, in2) yields
	 * extFunct:::foo:::2:::1:::in1:::in2:::out1
	 * 
	 */
	public static Instruction parseInstruction(String str) throws DMLRuntimeException, DMLUnsupportedOperationException {
		
		String[] parts = InstructionUtils.getInstructionPartsWithValueType ( str );
		String namespace = parts[1];
		String functionName = parts[2];
		int numInputs = Integer.valueOf(parts[3]);
		int numOutputs = Integer.valueOf(parts[4]);
		ArrayList<String> boundInParamNames = new ArrayList<String>();
		ArrayList<String> boundOutParamNames = new ArrayList<String>();
		
		int FIRST_PARAM_INDEX = 5;
		for (int i = 0; i < numInputs; i++) {
			boundInParamNames.add(parts[FIRST_PARAM_INDEX + i]);
		}
		for (int i = 0; i < numOutputs; i++) {
			boundOutParamNames.add(parts[FIRST_PARAM_INDEX + numInputs + i]);
		}
		
		return new FunctionCallCPInstruction ( namespace,functionName, boundInParamNames, boundOutParamNames, str );
	}

	@Override
	public void processInstruction(ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{		
		LOG.trace("Executing instruction : " + this.toString());
		// get the function program block (stored in the Program object)
		FunctionProgramBlock fpb = ec.getProgram().getFunctionProgramBlock(this._namespace, this._functionName);
		
		// create bindings to formal parameters for given function call
		// These are the bindings passed to the FunctionProgramBlock for function execution 
		LocalVariableMap functionVariables = new LocalVariableMap();
		
		for (int i=0; i<fpb.getInputParams().size();i++) {
			
			// for each formal parameter:  create a [formalParamName, variable] binding based on the function call, 
			// and place each binding into functionVariables. 4 cases for corresponding functionCall param:
			//  (1) is a variable 
			//			-- will be DataIdentifier expression (but not FunctionCallIdentifer) 
			//			-- look in _variables for the Data value 
			//			-- create [formalParamName, ScalarObject] binding
			//  (2) is a constant value -- will be ConstIdentifier expression -- create [formalParamName, ScalarObject] binding where ScalarObject is constant value
			//  (3) is default value -- will be NO expression --  create [formalParamName, ScalarObject] where ScalarObject is default value
			//	(4) is an expression [NOT SUPPORTED YET]
			
			DataIdentifier currFormalParam = fpb.getInputParams().get(i);
			String currFormalParamName = currFormalParam.getName();
			Data currFormalParamValue = null; 
			ValueType valType = fpb.getInputParams().get(i).getValueType();
				
			// CASE (3): using default value (scalars only)
			if (i > this._boundInputParamNames.size() || (ec.getVariable(this._boundInputParamNames.get(i)) == null)){
				
				if (valType == ValueType.BOOLEAN){
					boolean defaultVal = (i > this._boundInputParamNames.size()) ? Boolean.valueOf(fpb.getInputParams().get(i).getDefaultValue()).booleanValue() : Boolean.valueOf(this._boundInputParamNames.get(i)).booleanValue();
					currFormalParamValue = new BooleanObject(defaultVal);
				}
				else if (valType == ValueType.DOUBLE){
					double defaultVal = (i > this._boundInputParamNames.size()) ? Double.valueOf(fpb.getInputParams().get(i).getDefaultValue()).doubleValue() : Double.valueOf(this._boundInputParamNames.get(i)).doubleValue();
					currFormalParamValue = new DoubleObject(defaultVal);
				}
				else if (valType == ValueType.INT){ //via safe case for robustness
					long defaultVal = (i > this._boundInputParamNames.size()) ? UtilFunctions.parseToInt(fpb.getInputParams().get(i).getDefaultValue()) : UtilFunctions.parseToInt(this._boundInputParamNames.get(i));
					currFormalParamValue = new IntObject(defaultVal);
				}
				else if (valType == ValueType.STRING){
					String defaultVal = (i > this._boundInputParamNames.size()) ? fpb.getInputParams().get(i).getDefaultValue() : this._boundInputParamNames.get(i);
					currFormalParamValue = new StringObject(defaultVal);
				}
				else{
					throw new DMLUnsupportedOperationException(currFormalParamValue + " has inapporpriate value type");
				}
			}
			else {
				currFormalParamValue = ec.getVariable(this._boundInputParamNames.get(i));
			}
				
			functionVariables.put(currFormalParamName,currFormalParamValue);	
					
		}
		
		// Pin the input variables so that they do not get deleted 
		// from pb's symbol table at the end of execution of function
	    HashMap<String,Boolean> pinStatus = ec.pinVariables(this._boundInputParamNames);
		
		// Create a symbol table under a new execution context for the function invocation,
		// and copy the function arguments into the created table. 
		ExecutionContext fn_ec = new ExecutionContext(false, ec.getProgram());
		fn_ec.setVariables(functionVariables);
		
		// execute the function block
		try {
			fpb.execute(fn_ec);
		}
		catch (DMLScriptException e) {
			throw e;
		}
		catch (Exception e){
			String fname = this._namespace + "::" + this._functionName;
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
				throw new DMLUnsupportedOperationException(boundVarName + " was not assigned a return value");

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

	//
	//public String getOutputVariableName() {
	//	return null; //output.getName();
	//}

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
		_opcode = fname;
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
