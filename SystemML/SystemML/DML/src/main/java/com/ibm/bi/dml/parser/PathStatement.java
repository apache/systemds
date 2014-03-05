/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.parser;


 
public class PathStatement extends Statement
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	// Set of file system paths to packages
	private String _pathValue;
	
	public PathStatement(String pv){
		_pathValue = pv;
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
