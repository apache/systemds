/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.parser;


public abstract class LiveVariableAnalysis 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	VariableSet _read;
	VariableSet _updated;
	VariableSet _gen;
	VariableSet _kill;
	VariableSet _liveIn;
	VariableSet _liveOut;
	boolean _initialized = false;
	
	VariableSet _warnSet;	// variables that may not be initialized
							// applicable for control blocks
		
	public VariableSet variablesRead(){
		return _read;
	}
	
	public VariableSet variablesUpdated() {
		return _updated;
	}
	
	public VariableSet getWarn(){
		return _warnSet;
	}
	
	public VariableSet liveIn() {
		return _liveIn;
	}
	
	public VariableSet liveOut() {
		return _liveOut;
	}
	
	public VariableSet getKill(){
		return _kill;
	}
	
	public void setLiveOut(VariableSet lo){
		_liveOut = lo;
	}
	
	public void setLiveIn(VariableSet li){
		_liveIn = li;
	}
	
	public abstract VariableSet initializeforwardLV(VariableSet activeIn) throws LanguageException;
	public abstract VariableSet initializebackwardLV(VariableSet loPassed) throws LanguageException;
	public abstract VariableSet analyze(VariableSet loPassed) throws LanguageException;
	
	
	public void updateLiveVariablesIn(VariableSet liveIn){
		 updateLiveVariables(_liveIn,liveIn);
	}
	
	public void updateLiveVariablesOut(VariableSet liveOut){
		 updateLiveVariables(_liveOut,liveOut);
	}
	
	private void updateLiveVariables(VariableSet origVars, VariableSet newVars){
		for (String var : newVars.getVariables().keySet()){
			if (origVars.containsVariable(var)){
				DataIdentifier varId = newVars.getVariable(var);
				if (varId != null){
					origVars.addVariable(var, varId);
				}
			}
		}
	}
}
