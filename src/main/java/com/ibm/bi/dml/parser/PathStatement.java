/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.parser;


 
public class PathStatement extends Statement
{
		
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
		 StringBuilder sb = new StringBuilder();
		 sb.append(Statement.SETWD + "(");
		 sb.append(_pathValue);
		 sb.append(")");
		 sb.append(";");
		 return sb.toString(); 
	}
	 
}
