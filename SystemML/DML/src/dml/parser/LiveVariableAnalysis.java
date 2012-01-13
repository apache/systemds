package dml.parser;

import dml.utils.LanguageException;

public abstract class LiveVariableAnalysis {
	
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

	public static void throwUndefinedVar ( String varName, String statement ) throws LanguageException {
		throw new LanguageException("Undefined Variable (" + varName + ") used in statement " + statement + ".",
				LanguageException.LanguageErrorCodes.INVALID_PARAMETERS);
	}
}
