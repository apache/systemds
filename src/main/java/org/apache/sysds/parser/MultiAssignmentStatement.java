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

package org.apache.sysds.parser;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MultiAssignmentStatement extends Statement
{
	private ArrayList<DataIdentifier> _targetList;
	private Expression _source;
	
	// rewrites statement to support function inlining (creates deep copy) 
	@Override
	public Statement rewriteStatement(String prefix) {
		ArrayList<DataIdentifier> newTargetList = new ArrayList<>();
		
		// rewrite targetList (deep copy)
		for (DataIdentifier target : _targetList){
			DataIdentifier newTarget = (DataIdentifier) target.rewriteExpression(prefix);
			newTargetList.add(newTarget);
		}
		
		// rewrite source (deep copy)
		Expression newSource = _source.rewriteExpression(prefix);
		
		// create rewritten assignment statement (deep copy)
		MultiAssignmentStatement retVal = new MultiAssignmentStatement(newTargetList, newSource);
		retVal.setParseInfo(this);
		
		return retVal;
	}
	
	public MultiAssignmentStatement(ArrayList<DataIdentifier> tList, Expression s){
		_targetList = tList;
		_source = s;
	}

	public ArrayList<DataIdentifier> getTargetList(){
		return _targetList;
	}
	
	public void setTargetList(List<DataIdentifier> diList) {
		_targetList.clear();
		_targetList.addAll(diList);
	}
	
	public Expression getSource(){
		return _source;
	}
	
	@Override
	// conservative assignment to separate statement block; will merge later if possible
	public boolean controlStatement() {
		return true;
	}
	
	@Override
	public void initializeforwardLV(VariableSet activeIn) { }
	
	@Override
	public VariableSet initializebackwardLV(VariableSet lo) { return lo; }
	
	@Override
	public VariableSet variablesRead() {
		VariableSet result = new VariableSet();
		// add variables read by source expression
		result.addVariables(_source.variablesRead());
		// for any IndexedIdentifier on LHS, add variables for indexing expressions
		for (int i=0; i<_targetList.size(); i++){
			if (_targetList.get(i) instanceof IndexedIdentifier) {
				IndexedIdentifier target = (IndexedIdentifier) _targetList.get(i);
				result.addVariables(target.variablesRead());
			}
		}
		return result;
	}
	
	@Override
	public  VariableSet variablesUpdated() {
	
		VariableSet result =  new VariableSet();
		
		// add target to updated list
		for (DataIdentifier target : _targetList){
			result.addVariable(target.getName(), target);
		}
		return result;
	}
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append(Arrays.toString(_targetList.toArray()));
		sb.append(" = ");
		sb.append(_source.toString());
		sb.append(";");
		return sb.toString();
	}

	public void setSource(FunctionCallIdentifier s) {
		_source = s;
	}
}
