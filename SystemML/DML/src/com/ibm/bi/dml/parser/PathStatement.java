package com.ibm.bi.dml.parser;

import java.util.ArrayList;

import com.ibm.bi.dml.utils.LanguageException;

 
public class PathStatement extends Statement{
	
	// Set of file system paths to packages
	ArrayList<String> _pathSet;
	
	public PathStatement(){
		_pathSet = new ArrayList<String>();
	}
	
	/**
	 * setPackagePaths: sets the paths where packages can be located
	 * @param pathSet colon-delimited set of paths to packages
	 */
	public void addPackagePaths(String pathSet){
		
		String[] paths = pathSet.split(":");
		for (String path : paths)
			_pathSet.add(path);
	}
	
	public ArrayList<String> getPackagePaths(){
		return _pathSet;
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
		 sb.append(Statement.DMLPATH + " ");
		
		 for (int i=0; i < _pathSet.size(); i++ ){
			 sb.append(_pathSet.get(i));
			 if (i < _pathSet.size()-1) sb.append(":");
		 }
		 sb.append(";");
		 return sb.toString(); 
	}
	 
}
