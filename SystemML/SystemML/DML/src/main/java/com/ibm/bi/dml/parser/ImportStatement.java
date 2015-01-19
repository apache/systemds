/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.parser;

import java.io.File;


 
public class ImportStatement extends Statement
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	/*
	// indicates the imported list of function names 
	private ArrayList<String> _importVarList;
	
	// Indicates if the "import all" option (import moudle (*)) is being used
	private boolean _importAll;
	*/
	
	// Stores the namespace methods for 
	private String _namespace;
	
	// Stores the filepath with the DML file location
	private String _filePath;
	
	// store "complete path" -- DML working dir + filepath
	private String _completePath;
	

	public ImportStatement(){
		_namespace 	  	= null;
		_filePath 		= null;
		_completePath 	= null;
		
	}
	
	public void setNamespace(String passed){
		_namespace = passed;
	}
	
	public String getNamespace(){
		return _namespace;
	}
	
	public String getFilePath() {
		return _filePath;
	}
	
	public void setFilePath(String filePath) {
		_filePath = filePath;
	}

	public String getCompletePath() {
		return _completePath;
	}
	
	public void setCompletePath(String filePath) {
		_completePath = filePath;
	}
	
	/**
	 * verify: verifies that the module actually exists and sets the complete 
	 * path for the import statement (package root specified in working directory concatenated 
	 * with package path 
	 *  
	 * @param packagePaths specifies directories to search for DML packages
	 * 
	 * @return true if the module exists in one of the specified packagePaths
	 */
	public boolean verify(String workingDir){
		
		if (!workingDir.endsWith("/"))
			workingDir += "/";
			
		String path = workingDir + this._filePath;
		File completePathFile = new File(path);
		if (completePathFile.exists()){
			_completePath = path;
			return true;
		}	
		_completePath = null;
		return false;
	} // end method
	
	public Statement rewriteStatement(String prefix) throws LanguageException{
		throw new LanguageException(this.printErrorLocation() + "rewriting for inlining not supported for ImportStatement");
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
		 StringBuilder sb = new StringBuilder();
		 sb.append(Statement.SOURCE + "(");
		 sb.append(this._filePath + ")");
		 
		 if (this._namespace != null){
			 sb.append(" AS " + this._namespace);
		 }
		 sb.append(";");
		 return sb.toString(); 
	}
	 
}
