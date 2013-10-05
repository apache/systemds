/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.parser;

import java.util.HashMap;

public class CVStatement extends Statement 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	private FunctionCallIdentifier 	_inputs 				= null;
	
	private HashMap<String,String>	_partitionParams		= null; 
	private FunctionCallIdentifier 	_partitionOutputs 		= null;
	
	private FunctionCallIdentifier 	_trainFunctionCall 		= null;
	private FunctionCallIdentifier 	_trainFunctionOutputs 	= null;
	
	private FunctionCallIdentifier 	_testFunctionCall		= null;
	private FunctionCallIdentifier 	_testFunctionOutputs 	= null;
	
	private Identifier 	_aggFunctionCall 		= null;
	private FunctionCallIdentifier 	_aggFunctionOutputs 	= null;
	
	public Statement rewriteStatement(String prefix) throws LanguageException{
		LOG.error(this.printErrorLocation() + "should not call rewriteStatement for CVStatement");
		throw new LanguageException(this.printErrorLocation() + "should not call rewriteStatement for CVStatement");
	}
	
	@Override
	public String toString() {
		StringBuffer sb = new StringBuffer();
		sb.append("crossval " 	+_inputs.toString());
		sb.append("partition " 	+ _partitionParams.toString() 	+ " as " + _partitionOutputs.toString());
		sb.append("train " 		+ _trainFunctionCall.toString() + " as " + _trainFunctionOutputs.toString()); 
		sb.append("test " 		+ _testFunctionCall.toString() 	+ " as " + _testFunctionOutputs.toString()); 
		sb.append("aggregate "  + _aggFunctionCall.toString()	+ " as " + _aggFunctionOutputs.toString());
		sb.append("\n");
		return sb.toString();
	}

	public CVStatement() {
	
	}

	@Override
	public boolean controlStatement() {
		return false;
	}

	@Override
	public VariableSet initializebackwardLV(VariableSet lo) {
		return null;
	}

	@Override
	public void initializeforwardLV(VariableSet activeIn) {
	}

	@Override
	public VariableSet variablesRead() {
		VariableSet set = new VariableSet();
		for (Expression input : _inputs.getParamExpressions()){
			set.addVariables(input.variablesRead());
		}
		return set;
	}

	@Override
	// only update 
	public VariableSet variablesUpdated() {
		VariableSet set = new VariableSet();
		for (Expression output : _aggFunctionOutputs.getParamExpressions())
			set.addVariables(output.variablesUpdated());
		return set;
	}

	public FunctionCallIdentifier get_inputs() {
		return _inputs;
	}

	public void set_inputs(FunctionCallIdentifier inputs) {
		_inputs = inputs;
	}

	public HashMap<String, String> get_partitionParams() {
		return _partitionParams;
	}

	public void set_partitionParams(HashMap<String, String> partitionParams) {
		_partitionParams = partitionParams;
	}

	public FunctionCallIdentifier get_partitionOutputs() {
		return _partitionOutputs;
	}

	public void set_partitionOutputs(FunctionCallIdentifier partitionOutputs) {
		_partitionOutputs = partitionOutputs;
	}

	public FunctionCallIdentifier get_trainFunctionCall() {
		return _trainFunctionCall;
	}

	public void set_trainFunctionCall(FunctionCallIdentifier trainFunctionCall) {
		_trainFunctionCall = trainFunctionCall;
	}

	public FunctionCallIdentifier get_trainFunctionOutputs() {
		return _trainFunctionOutputs;
	}

	public void set_trainFunctionOutputs(FunctionCallIdentifier trainFunctionOutputs) {
		_trainFunctionOutputs = trainFunctionOutputs;
	}

	public FunctionCallIdentifier get_testFunctionCall() {
		return _testFunctionCall;
	}

	public void set_testFunctionCall(FunctionCallIdentifier testFunctionCall) {
		_testFunctionCall = testFunctionCall;
	}

	public FunctionCallIdentifier get_testFunctionOutputs() {
		return _testFunctionOutputs;
	}

	public void set_testFunctionOutputs(FunctionCallIdentifier testFunctionOutputs) {
		_testFunctionOutputs = testFunctionOutputs;
	}

	public Identifier get_aggFunctionCall() {
		return _aggFunctionCall;
	}

	public void set_aggFunctionCall(Identifier aggFunctionCall) {
		_aggFunctionCall = aggFunctionCall;
	}

	public FunctionCallIdentifier get_aggFunctionOutputs() {
		return _aggFunctionOutputs;
	}

	public void set_aggFunctionOutputs(FunctionCallIdentifier aggFunctionOutputs) {
		_aggFunctionOutputs = aggFunctionOutputs;
	}

} // end class