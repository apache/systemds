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



public class WhileStatement extends Statement
{
	private ConditionalPredicate _predicate;
	private ArrayList<StatementBlock> _body;
	
	@Override
	public Statement rewriteStatement(String prefix) {
		throw new LanguageException(this.printErrorLocation() + "should not call rewriteStatement for WhileStatement");
	}
	
	public WhileStatement() {
		_predicate = null;
		_body = new ArrayList<>();
	}
	
	public void setPredicate(ConditionalPredicate pred){
		_predicate = pred;
	}
		
	public void addStatementBlock(StatementBlock sb){
		_body.add(sb);
	}
	
	public ConditionalPredicate getConditionalPredicate(){
		return _predicate;
	}
	
	public ArrayList<StatementBlock> getBody(){
		return _body;
	}
	
	public void setBody(ArrayList<StatementBlock> body){
		_body = body;
	}
	
	@Override
	public boolean controlStatement() {
		return true;
	}
	
	public void mergeStatementBlocks(){
		_body = StatementBlock.mergeStatementBlocks(_body);
	}
	
	@Override
	public String toString(){
		StringBuilder sb = new StringBuilder();
		sb.append("while ( ");
		sb.append(_predicate.toString());
		sb.append(") { \n");
		for (StatementBlock block : _body){
			sb.append(block.toString());
		}
		sb.append("}\n");
		return sb.toString();
	}

	@Override
	public void initializeforwardLV(VariableSet activeIn) {
		throw new LanguageException(this.printErrorLocation() + "should never call initializeforwardLV for WhileStatement");
	}
	
	@Override
	public VariableSet initializebackwardLV(VariableSet lo) {
		throw new LanguageException(this.printErrorLocation() + "should never call initializeforwardLV for WhileStatement");
	}
	
	@Override
	public VariableSet variablesRead() {
		LOG.warn(this.printWarningLocation() + "should not call variablesRead from WhileStatement ");
		return new VariableSet();
	}

	@Override
	public VariableSet variablesUpdated() {
		LOG.warn(this.printWarningLocation() + "should not call variablesRead from WhileStatement ");
		return new VariableSet();
	}
}
