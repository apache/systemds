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

import java.util.ArrayList;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.debug.DMLBreakpointManager;


public class AssignmentStatement extends Statement
{
		
	private ArrayList<DataIdentifier> _targetList;
	private Expression _source;
	 
	// rewrites statement to support function inlining (creates deep copy)
	public Statement rewriteStatement(String prefix) throws LanguageException{
				
		// rewrite target (deep copy)
		DataIdentifier newTarget = (DataIdentifier)_targetList.get(0).rewriteExpression(prefix);
		
		// rewrite source (deep copy)
		Expression newSource = _source.rewriteExpression(prefix);
		
		// create rewritten assignment statement (deep copy)
		AssignmentStatement retVal = new AssignmentStatement(newTarget, newSource,this.getBeginLine(), 
											this.getBeginColumn(), this.getEndLine(), this.getEndColumn());
		
		return retVal;
	}
	
	
	public AssignmentStatement(DataIdentifier t, Expression s) {

		_targetList = new ArrayList<DataIdentifier>();
		_targetList.add(t);
		_source = s;
	}
	
	
	public AssignmentStatement(DataIdentifier t, Expression s, int beginLine, int beginCol, int endLine, int endCol) 
		throws LanguageException
	{	
		_targetList = new ArrayList<DataIdentifier>();
		_targetList.add(t);
		_source = s;
	
		setBeginLine(beginLine);
		setBeginColumn(beginCol);
		setEndLine(endLine);
		setEndColumn(endCol);
		
	}
	
	public DataIdentifier getTarget(){
		return _targetList.get(0);
	}
	
	public ArrayList<DataIdentifier> getTargetList()
	{
		return _targetList;
	}

	public Expression getSource(){
		return _source;
	}
	public void setSource(Expression s){
		_source = s;
	}
	
	@Override
	public boolean controlStatement() {
		// ensure that breakpoints end up in own statement block 
		if (DMLScript.ENABLE_DEBUG_MODE) {
			DMLBreakpointManager.insertBreakpoint(_source.getBeginLine());
			return true;
		}

		// for now, ensure that function call ends up in different statement block
		if (_source instanceof FunctionCallIdentifier)
			return true;
		
		return false;
	}
	
	public void initializeforwardLV(VariableSet activeIn){
		//do nothing
	}
	
	public VariableSet initializebackwardLV(VariableSet lo){
		return lo;
	}
	
	public VariableSet variablesRead() {
		VariableSet result = new VariableSet();
		
		// add variables read by source expression
		result.addVariables(_source.variablesRead());
		
		// for LHS IndexedIdentifier, add variables for indexing expressions
		for (int i=0; i<_targetList.size(); i++){
			if (_targetList.get(i) instanceof IndexedIdentifier) {
				IndexedIdentifier target = (IndexedIdentifier) _targetList.get(i);
				result.addVariables(target.variablesRead());
			}
		}		
		return result;
	}
	
	public  VariableSet variablesUpdated() {
		VariableSet result =  new VariableSet();
		
		// add target to updated list
		for (DataIdentifier target : _targetList)
			result.addVariable(target.getName(), target);
		return result;
	}
	
	public String toString(){
		StringBuilder sb = new StringBuilder();
		for (int i=0; i< _targetList.size(); i++){
			sb.append(_targetList.get(i).toString());
		}
		sb.append(" = ");
		sb.append(_source.toString());
		sb.append(";");
		
		return sb.toString();
	}
}
