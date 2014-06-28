/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.parser;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Vector;

import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.FunctionOp.FunctionType;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;


public class FunctionStatementBlock extends StatementBlock 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	/**
	 *  TODO: DRB:  This needs to be changed to reflect:
	 *  
	 *    1)  Default values for variables -- need to add R styled check here to make sure that once vars with 
	 *    default values start, they keep going to the right
	 *    
	 *    2)  The other parameters for External Functions
	 * @throws IOException 
	 */
	public VariableSet validate(DMLProgram dmlProg, VariableSet ids, HashMap<String,ConstIdentifier> constVars) 
		throws LanguageException, ParseException, IOException 
	{
		
		if (_statements.size() > 1){
			throw new LanguageException(this.printBlockErrorLocation() + "FunctionStatementBlock should have only 1 statement (FunctionStatement)");
		}
		FunctionStatement fstmt = (FunctionStatement) _statements.get(0);
			
		// handle DML-bodied functions
		if (!(fstmt instanceof ExternalFunctionStatement)){
					
			// perform validate for function body
			this._dmlProg = dmlProg;
			for(StatementBlock sb : fstmt.getBody())
			{
				ids = sb.validate(dmlProg, ids, constVars);
				constVars = sb.getConstOut();
			}
			if (fstmt.getBody().size() > 0)
				_constVarsIn.putAll(fstmt.getBody().get(0).getConstIn());
			
			if (fstmt.getBody().size() > 1)
				_constVarsOut.putAll(fstmt.getBody().get(fstmt.getBody().size()-1).getConstOut());
			
//			for each return value, check variable is defined and validate the return type
			// 	if returnValue type known incorrect, then throw exception
			Vector<DataIdentifier> returnValues = fstmt.getOutputParams();
			for (DataIdentifier returnValue : returnValues){
				DataIdentifier curr = ids.getVariable(returnValue.getName());
				if (curr == null){
					LOG.error(this.printBlockErrorLocation() + "for function " + fstmt.getName() + ", return variable " + returnValue.getName() + " must be defined in function ");
					throw new LanguageException(this.printBlockErrorLocation() + "for function " + fstmt.getName() + ", return variable " + returnValue.getName() + " must be defined in function ");
				}
				
				if (curr.getDataType() == DataType.UNKNOWN){
					LOG.error(curr.printWarningLocation() + "for function " + fstmt.getName() + ", return variable " + curr.getName() + " data type of " + curr.getDataType() + " may not match data type in function signature of " + returnValue.getDataType());
				}
				
				if (curr.getValueType() == ValueType.UNKNOWN){
					LOG.error(curr.printWarningLocation() + "for function " + fstmt.getName() + ", return variable " + curr.getName() + " data type of " + curr.getValueType() + " may not match data type in function signature of " + returnValue.getValueType());
				}
				
				if (curr.getDataType() != DataType.UNKNOWN && !curr.getDataType().equals(returnValue.getDataType()) ){
					LOG.error(curr.printErrorLocation() + "for function " + fstmt.getName() + ", return variable " + curr.getName() + " data type of " + curr.getDataType() + " does not match data type in function signature of " + returnValue.getDataType());
					throw new LanguageException(curr.printErrorLocation() + "for function " + fstmt.getName() + ", return variable " + curr.getName() + " data type of " + curr.getDataType() + " does not match data type in function signature of " + returnValue.getDataType());
				}
				
				if (curr.getValueType() != ValueType.UNKNOWN && !curr.getValueType().equals(returnValue.getValueType())){
					
					// attempt to convert value type: handle conversion from scalar DOUBLE or INT
					if (curr.getDataType() == DataType.SCALAR && returnValue.getDataType() == DataType.SCALAR){ 
						if (returnValue.getValueType() == ValueType.DOUBLE){
							if (curr.getValueType() == ValueType.INT){
								IntIdentifier currIntValue = (IntIdentifier)constVars.get(curr.getName());
								if (currIntValue != null){
									DoubleIdentifier currDoubleValue = new DoubleIdentifier(currIntValue.getValue());
									constVars.put(curr.getName(), currDoubleValue);
								}
								LOG.warn(curr.printWarningLocation() + "for function " + fstmt.getName() + ", return variable " + curr.getName() + " value type of " + curr.getValueType() + " does not match value type in function signature of " + returnValue.getValueType() + " but was safely cast");
								curr.setValueType(ValueType.DOUBLE);
								ids.addVariable(curr.getName(), curr);
							}
							else {
								// THROW EXCEPTION -- CANNOT CONVERT
								LOG.error(curr.printErrorLocation() + "for function " + fstmt.getName() + ", return variable " + curr.getName() + " value type of " + curr.getValueType() + " does not match value type in function signature of " + returnValue.getValueType() + " and cannot safely cast value");
								throw new LanguageException(curr.printErrorLocation() + "for function " + fstmt.getName() + ", return variable " + curr.getName() + " value type of " + curr.getValueType() + " does not match value type in function signature of " + returnValue.getValueType() + " and cannot safely cast value");
							}
						}
						if (returnValue.getValueType() == ValueType.INT){
							// THROW EXCEPTION -- CANNOT CONVERT
							LOG.error(curr.printErrorLocation() + "for function " + fstmt.getName() + ", return variable " + curr.getName() + " value type of " + curr.getValueType() + " does not match value type in function signature of " + returnValue.getValueType() + " and cannot safely cast " + curr.getValueType() + " as " + returnValue.getValueType());
							throw new LanguageException(curr.printErrorLocation() + "for function " + fstmt.getName() + ", return variable " + curr.getName() + " value type of " + curr.getValueType() + " does not match value type in function signature of " + returnValue.getValueType() + " and cannot safely cast " + curr.getValueType() + " as " + returnValue.getValueType());
							
						} 
					}	
					else {
						LOG.error(curr.printErrorLocation() + "for function " + fstmt.getName() + ", return variable " + curr.getName() + " value type of " + curr.getValueType() + " does not match value type in function signature of " + returnValue.getValueType() + " and cannot safely cast double as int");
						throw new LanguageException(curr.printErrorLocation() + "for function " + fstmt.getName() + ", return variable " + curr.getName() + " value type of " + curr.getValueType() + " does not match value type in function signature of " + returnValue.getValueType() + " and cannot safely cast " + curr.getValueType() + " as " + returnValue.getValueType());
					}
				}
				
			}
		}
		else {
			//validate specified attributes and attribute values
			ExternalFunctionStatement efstmt = (ExternalFunctionStatement) fstmt;
			efstmt.validateParameters();
			
			//validate child statements
			this._dmlProg = dmlProg;
			for(StatementBlock sb : efstmt.getBody()) 
			{
				ids = sb.validate(dmlProg, ids, constVars);
				constVars = sb.getConstOut();
			}
		}
		
		

		return ids;
	}

	public FunctionType getFunctionOpType()
	{
		FunctionType ret = FunctionType.UNKNOWN;
		
		FunctionStatement fstmt = (FunctionStatement) _statements.get(0);
		if (fstmt instanceof ExternalFunctionStatement) 
		{
			ExternalFunctionStatement efstmt = (ExternalFunctionStatement) fstmt;
			String execType = efstmt.getOtherParams().get(ExternalFunctionStatement.EXEC_TYPE);
			if( execType!=null ){
				if(execType.equals(ExternalFunctionStatement.IN_MEMORY))
					ret = FunctionType.EXTERNAL_MEM;
				else
					ret = FunctionType.EXTERNAL_FILE;
			}
		}
		else
		{
			ret = FunctionType.DML; 
		}
		
		return ret;
	}
	
	public VariableSet initializeforwardLV(VariableSet activeInPassed) throws LanguageException {
		
		FunctionStatement fstmt = (FunctionStatement)_statements.get(0);
		if (_statements.size() > 1){
			throw new LanguageException(this.printBlockErrorLocation() + "FunctionStatementBlock should have only 1 statement (while statement)");
		}
		
		_read = new VariableSet();
		_gen = new VariableSet();
				
		VariableSet current = new VariableSet();
		current.addVariables(activeInPassed);
		
		for (int  i = 0; i < fstmt.getBody().size(); i++){
			
			StatementBlock sb = fstmt.getBody().get(i);
			current = sb.initializeforwardLV(current);	
			
			// for each generated variable in this block, check variable not killed
			// in prior statement block in while stmt blody
			for (String varName : sb._gen.getVariableNames()){
				
				// IF the variable is NOT set in the while loop PRIOR to this stmt block, 
				// THEN needs to be generated
				if (!_kill.getVariableNames().contains(varName)){
					_gen.addVariable(varName, sb._gen.getVariable(varName));	
				}
			}
			
			_read.addVariables(sb._read);
			_updated.addVariables(sb._updated);
		
			// only add kill variables for statement blocks guaranteed to execute
			if (!(sb instanceof WhileStatementBlock) && !(sb instanceof ForStatementBlock) ){
				_kill.addVariables(sb._kill);
			}	
		}
		
		// activeOut includes variables from passed live in and updated in the while body
		_liveOut = new VariableSet();
		_liveOut.addVariables(current);
		_liveOut.addVariables(_updated);
		return _liveOut;
	}

	public VariableSet initializebackwardLV(VariableSet loPassed) throws LanguageException{
		
		FunctionStatement wstmt = (FunctionStatement)_statements.get(0);
			
		VariableSet lo = new VariableSet();
		lo.addVariables(loPassed);
		
		// calls analyze for each statement block in while stmt body
		int numBlocks = wstmt.getBody().size();
		for (int i = numBlocks - 1; i >= 0; i--){
			lo = wstmt.getBody().get(i).analyze(lo);
		}	
		
		VariableSet loReturn = new VariableSet();
		loReturn.addVariables(lo);
		return loReturn;
	
	}
	
	
	public ArrayList<Hop> get_hops() throws HopsException {
		
		if (_hops != null && _hops.size() > 0){
			throw new HopsException(this.printBlockErrorLocation() + "there should be no HOPs associated with the FunctionStatementBlock");
		}
		
		return _hops;
	}
	
	
	public VariableSet analyze(VariableSet loPassed) throws LanguageException{
		throw new LanguageException(this.printBlockErrorLocation() + "Both liveIn and liveOut variables need to be specified for liveness analysis for FunctionStatementBlock");	
	}
	
	
	public VariableSet analyze(VariableSet liPassed, VariableSet loPassed) throws LanguageException{
 		
		VariableSet candidateLO = new VariableSet();
		candidateLO.addVariables(loPassed);
		candidateLO.addVariables(_gen);
		
		VariableSet origLiveOut = new VariableSet();
		origLiveOut.addVariables(_liveOut);
		
		_liveOut = new VariableSet();
	 	for (String name : candidateLO.getVariableNames()){
	 		if (origLiveOut.containsVariable(name)){
	 			_liveOut.addVariable(name, candidateLO.getVariable(name));
	 		}
	 	}
	 	
		initializebackwardLV(_liveOut);
		
		// Cannot remove kill variables
		_liveIn = new VariableSet();
		_liveIn.addVariables(liPassed);
		
		/*
		for(String key : _gen.getVariableNames()){
			if (_liveIn.containsVariable(key) == false){
				throw new LanguageException(this.getStatement(0).printErrorLocation() + "function " + ((FunctionStatement)this.getStatement(0)).getName() + " requires variable " + key + " to be passed as formal parameter");
			}
		}
		*/
		VariableSet liveInReturn = new VariableSet();
		liveInReturn.addVariables(_liveIn);
		
		return liveInReturn;
	}
}