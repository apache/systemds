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

import org.antlr.v4.runtime.ParserRuleContext;
import org.apache.sysds.common.Builtins;

public class AssignmentStatement extends Statement
{
	private ArrayList<DataIdentifier> _targetList;
	private Expression _source;
	private boolean _isAccum; //+=
	 
	// rewrites statement to support function inlining (creates deep copy)
	@Override
	public Statement rewriteStatement(String prefix) {
		// rewrite target (deep copy)
		DataIdentifier newTarget = (DataIdentifier) _targetList.get(0).rewriteExpression(prefix);
		// rewrite source (deep copy)
		Expression newSource = _source.rewriteExpression(prefix);
		// create rewritten assignment statement (deep copy)
		AssignmentStatement retVal = new AssignmentStatement(newTarget, newSource, this);
		return retVal;
	}
	
	public AssignmentStatement(DataIdentifier di, Expression exp) {
		_targetList = new ArrayList<>();
		_targetList.add(di);
		_source = exp;
	}
	
	public AssignmentStatement(DataIdentifier di, Expression exp, ParseInfo parseInfo) {
		this(di, exp);
		setParseInfo(parseInfo);
	}

	public AssignmentStatement(ParserRuleContext ctx, DataIdentifier di, Expression exp) {
		this(di, exp);
		setCtxValues(ctx);
	}

	public AssignmentStatement(ParserRuleContext ctx, DataIdentifier di, Expression exp, String filename) {
		this(ctx, di, exp);
		setFilename(filename);
	}

	public DataIdentifier getTarget(){
		return _targetList.get(0);
	}
	
	public ArrayList<DataIdentifier> getTargetList() {
		return _targetList;
	}

	public void setTarget(DataIdentifier di) {
		_targetList.set(0, di);
	}
	
	public Expression getSource(){
		return _source;
	}
	
	public void setSource(Expression s){
		_source = s;
	}
	
	public boolean isAccumulator() {
		return _isAccum;
	}
	
	public void setAccumulator(boolean flag) {
		_isAccum = flag;
	}
	
	@Override
	public boolean controlStatement() {
		// for now, ensure that function call ends up in different statement block
		if (_source instanceof FunctionCallIdentifier)
			return true;
		if (_source.toString().contains(Builtins.TIME.toString()))
			return true;
		
		return false;
	}

	@Override
	public void initializeforwardLV(VariableSet activeIn){
		//do nothing
	}
	
	@Override
	public VariableSet initializebackwardLV(VariableSet lo){
		return lo;
	}
	
	@Override
	public VariableSet variablesRead() {
		VariableSet result = new VariableSet();
		// add variables read by source expression
		result.addVariables(_source.variablesRead());
		// for left indexing or accumulators add targets as well
		for (DataIdentifier target : _targetList)
			if (target instanceof IndexedIdentifier || _isAccum )
				result.addVariables(target.variablesRead());
		return result;
	}
	
	@Override
	public  VariableSet variablesUpdated() {
		VariableSet result =  new VariableSet();
		// add target to updated list
		for (DataIdentifier target : _targetList)
			if (target != null)
				result.addVariable(target.getName(), target);
		return result;
	}
	
	@Override
	public String toString(){
		StringBuilder sb = new StringBuilder();
		for (int i=0; i< _targetList.size(); i++)
			sb.append(_targetList.get(i));
		sb.append(_isAccum ? " += " : " = ");
		if (_source instanceof StringIdentifier) {
			sb.append("\"");
			sb.append(_source.toString());
			sb.append("\"");
		} else {
			sb.append(_source.toString());
		}
		sb.append(";");
		return sb.toString();
	}
}
