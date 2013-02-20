package com.ibm.bi.dml.parser;

import com.ibm.bi.dml.utils.LanguageException;

 
public class PathStatement extends Statement{
	
	// Set of file system paths to packages
	String _pathValue;
	
	public PathStatement(){}
	
	public void setPathValue(String pathSet){
		_pathValue = pathSet;
	}
	
	public String getPathValue(){
		return _pathValue;
	}
	
	public Statement rewriteStatement(String prefix) throws LanguageException{
		return this;	
	}
		
	public void initializeforwardLV(VariableSet activeIn){}
	
	public VariableSet initializebackwardLV(VariableSet lo){
		return lo;
	}
	
	@Override
	public VariableSet variablesRead() {
		return null;
	}

	@Override
	public VariableSet variablesUpdated() {
	  	return null;
	}

	@Override
	public boolean controlStatement() {
		return false;
	}

	public String toString(){
		 StringBuffer sb = new StringBuffer();
		 sb.append(Statement.SETWD + "(");
		 sb.append(_pathValue);
		 sb.append(")");
		 sb.append(";");
		 return sb.toString(); 
	}
	 
}
