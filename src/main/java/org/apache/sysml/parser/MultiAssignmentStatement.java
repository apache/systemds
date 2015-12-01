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

package org.apache.sysml.parser;

import java.util.ArrayList;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.debug.DMLBreakpointManager;



public class MultiAssignmentStatement extends Statement
{
		
	private ArrayList<DataIdentifier> _targetList;
	private Expression _source;
		
	// rewrites statement to support function inlining (creates deep copy) 
	public Statement rewriteStatement(String prefix) throws LanguageException{
				
		ArrayList<DataIdentifier> newTargetList = new ArrayList<DataIdentifier>();
		
		// rewrite targetList (deep copy)
		for (DataIdentifier target : _targetList){
			DataIdentifier newTarget = (DataIdentifier) target.rewriteExpression(prefix);
			newTargetList.add(newTarget);
		}
		
		// rewrite source (deep copy)
		Expression newSource = _source.rewriteExpression(prefix);
		
		// create rewritten assignment statement (deep copy)
		MultiAssignmentStatement retVal = new MultiAssignmentStatement(newTargetList, newSource);
		retVal.setBeginLine(this.getBeginLine());
		retVal.setBeginColumn(this.getBeginColumn());
		retVal.setEndLine(this.getEndLine());
		retVal.setEndColumn(this.getEndColumn());
		
		return retVal;
	}
	
	public MultiAssignmentStatement(ArrayList<DataIdentifier> tList, Expression s){
		_targetList = tList;
		_source = s;
	}
	
	// NOTE: f is not used -- however, error is thrown "methods have same erasure" if not included in signature
	public MultiAssignmentStatement(ArrayList<ArrayList<Expression>> exprListList, Expression s, int f){
		
		_source = s;
		
		_targetList = new ArrayList<DataIdentifier>();
		for (ArrayList<Expression> exprList : exprListList){
			Expression expr = exprList.get(0);
			if( expr instanceof IndexedIdentifier )
				_targetList.add((IndexedIdentifier)expr);
			else
				_targetList.add(new DataIdentifier(expr.toString()));
		}
	}
	
	public ArrayList<DataIdentifier> getTargetList(){
		return _targetList;
	}
	
	public Expression getSource(){
		return _source;
	}
	
	@Override
	// conservative assignment to separate statement block; will merge later if possible
	public boolean controlStatement() {
		// ensure that breakpoints end up in own statement block 
		if (DMLScript.ENABLE_DEBUG_MODE) {
			DMLBreakpointManager.insertBreakpoint(_source.getBeginLine());
		}
		return true;
	}
	
	public void initializeforwardLV(VariableSet activeIn){}
	public VariableSet initializebackwardLV(VariableSet lo){
		return lo;
	}
	
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
	
	public  VariableSet variablesUpdated() {
	
		VariableSet result =  new VariableSet();
		
		// add target to updated list
		for (DataIdentifier target : _targetList){
			result.addVariable(target.getName(), target);
		}
		return result;
	}
	
	@Override
	public String toString()
	{
		StringBuilder sb = new StringBuilder();
		sb.append("[");
		
		for( int i=0; i< _targetList.size(); i++ )
		{
			sb.append(_targetList.get(i).toString());
			if (i < _targetList.size() - 1)
				sb.append(",");
		}
		sb.append("] = ");
		sb.append(_source.toString());
		sb.append(";");
		
		return sb.toString();
	}

	public void setSource(FunctionCallIdentifier s) {
		_source = s;
		
	}
}
