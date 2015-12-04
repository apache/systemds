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

package org.apache.sysml.parser;


 
public class ImportStatement extends Statement
{
		
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
	
	
	public Statement rewriteStatement(String prefix) throws LanguageException{
		LOG.error(this.printErrorLocation() + "rewriting for inlining not supported for ImportStatement");
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
