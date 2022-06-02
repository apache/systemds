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
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.recompile.Recompiler;
import org.apache.sysds.lops.Lop;


public class WhileStatementBlock extends StatementBlock 
{
	
	private Hop _predicateHops;
	private Lop _predicateLops = null;
	private boolean _requiresPredicateRecompile = false;
	
	@Override
	public VariableSet validate(DMLProgram dmlProg, VariableSet ids, HashMap<String,ConstIdentifier> constVars, boolean conditional) 
	{
		if (_statements.size() > 1){
			raiseValidateError("WhileStatementBlock should have only 1 statement (while statement)", conditional);
		}
		
		WhileStatement wstmt = (WhileStatement) _statements.get(0);
		ConditionalPredicate predicate = wstmt.getConditionalPredicate();
		
		// Record original size information before loop for ALL variables 
		// Will compare size / type info for these after loop completes
		// Replace variables with changed size with unknown value 
		VariableSet origVarsBeforeBody = new VariableSet();
		for (String key : ids.getVariableNames()){
			DataIdentifier origId = ids.getVariable(key);
			DataIdentifier copyId = new DataIdentifier(origId);
			origVarsBeforeBody.addVariable(key, copyId);
		}
		
		//////////////////////////////////////////////////////////////////////////////
		// FIRST PASS: process the predicate / statement blocks in the body of the for statement
		///////////////////////////////////////////////////////////////////////////////
		
		//remove updated vars from constants
		for( String var : _updated.getVariableNames() )
			if( constVars.containsKey( var ) )
				constVars.remove( var );
		
		// process the statement blocks in the body of the while statement
		predicate.getPredicate().validateExpression(ids.getVariables(), constVars, conditional);
		ArrayList<StatementBlock> body = wstmt.getBody();
		
		_dmlProg = dmlProg;
		for(StatementBlock sb : body)
		{
			//always conditional
			ids = sb.validate(dmlProg, ids, constVars, true);
			constVars = sb.getConstOut();
		}
				
		if (!body.isEmpty()) {
			_constVarsIn.putAll(body.get(0).getConstIn());
			_constVarsOut.putAll(body.get(body.size()-1).getConstOut());
		}
		
		// for each updated variable 
		boolean revalidationRequired = false;
		for (String key : _updated.getVariableNames())
		{	
			DataIdentifier startVersion = origVarsBeforeBody.getVariable(key);
			DataIdentifier endVersion   = ids.getVariable(key);
			
			if (startVersion != null && endVersion != null)
			{	
				//handle data type change (reject) 
				if (!startVersion.getOutput().getDataType().equals(endVersion.getOutput().getDataType())){
					raiseValidateError("WhileStatementBlock has unsupported conditional data type change of variable '"+key+"' in loop body.", conditional);
				}
				
				//handle size change
				long startVersionDim1 	= (startVersion instanceof IndexedIdentifier)   ? ((IndexedIdentifier)startVersion).getOrigDim1() : startVersion.getDim1(); 
				long endVersionDim1		= (endVersion instanceof IndexedIdentifier) ? ((IndexedIdentifier)endVersion).getOrigDim1() : endVersion.getDim1(); 
				long startVersionDim2 	= (startVersion instanceof IndexedIdentifier)   ? ((IndexedIdentifier)startVersion).getOrigDim2() : startVersion.getDim2(); 
				long endVersionDim2		= (endVersion instanceof IndexedIdentifier) ? ((IndexedIdentifier)endVersion).getOrigDim2() : endVersion.getDim2(); 
				
				boolean sizeUnchanged = ((startVersionDim1 == endVersionDim1) &&
						                 (startVersionDim2 == endVersionDim2) );
				
				//handle sparsity change
				//NOTE: nnz not propagated via validate, and hence, we conservatively assume that nnz have been changed.
				//long startVersionNNZ 	= startVersion.getNnz();
				//long endVersionNNZ    = endVersion.getNnz(); 
				//boolean nnzUnchanged  = (startVersionNNZ == endVersionNNZ);
				boolean nnzUnchanged = false;
				
				// IF size has changed -- 
				if (!sizeUnchanged || !nnzUnchanged){
					revalidationRequired = true;
					DataIdentifier recVersion = new DataIdentifier(endVersion);
					if(!sizeUnchanged)
						recVersion.setDimensions(-1, -1);
					if(!nnzUnchanged)
						recVersion.setNnz(-1);
					origVarsBeforeBody.addVariable(key, recVersion);
				}
			}
		}
		
			
		// revalidation is required -- size was updated for at least 1 variable
		if (revalidationRequired)
		{
			// update ids to the reconciled values
			ids = origVarsBeforeBody;
		
			//////////////////////////////////////////////////////////////////////////////
			// SECOND PASS: process the predicate / statement blocks in the body of the for statement
			///////////////////////////////////////////////////////////////////////////////
		
			//remove updated vars from constants
			for( String var : _updated.getVariableNames() )
				if( constVars.containsKey( var ) )
					constVars.remove( var );
			
			// process the statement blocks in the body of the while statement
			predicate.getPredicate().validateExpression(ids.getVariables(), constVars, conditional);
			body = wstmt.getBody();
			
			_dmlProg = dmlProg;
			for(StatementBlock sb : body)
			{
				//always conditional
				ids = sb.validate(dmlProg, ids, constVars, true);
				constVars = sb.getConstOut();
			}
					
			if (!body.isEmpty()) {
				_constVarsIn.putAll(body.get(0).getConstIn());
				_constVarsOut.putAll(body.get(body.size()-1).getConstOut());
			}		
		}
		
		return ids;
	}

	@Override
	public VariableSet initializeforwardLV(VariableSet activeInPassed) {
		
		WhileStatement wstmt = (WhileStatement)_statements.get(0);
		if (_statements.size() > 1){
			throw new LanguageException(_statements.get(0).printErrorLocation() + "WhileStatementBlock should have only 1 statement (while statement)");
		}
		
		_read = new VariableSet();
		_read.addVariables(wstmt.getConditionalPredicate().variablesRead());
		_updated.addVariables(wstmt.getConditionalPredicate().variablesUpdated());
		
		_gen = new VariableSet();
		_gen.addVariables(wstmt.getConditionalPredicate().variablesRead());
				
		VariableSet current = new VariableSet();
		current.addVariables(activeInPassed);
		
		for( StatementBlock sb : wstmt.getBody() )
		{
			current = sb.initializeforwardLV(current);	
			
			// for each generated variable in this block, check variable not killed
			// in prior statement block in while stmt blody
			for (String varName : sb._gen.getVariableNames()){
				
				// IF the variable is NOT set in the while loop PRIOR to this stmt block, 
				// THEN needs to be generated
				if (!_kill.getVariableNames().contains(varName)){
					_gen.addVariable(varName, sb._gen.getVariable(varName));	
				}
			}
			
			_read.addVariables(sb._read);
			_updated.addVariables(sb._updated);
		
			// only add kill variables for statement blocks guaranteed to execute
			if (!(sb instanceof WhileStatementBlock) && !(sb instanceof ForStatementBlock) ){
				_kill.addVariables(sb._kill);
			}	
		}
		
		// set preliminary "warn" set -- variables that if used later may cause runtime error
		// if the loop is not executed
		// warnSet = (updated MINUS (updatedIfBody INTERSECT updatedElseBody)) MINUS current
		for (String varName : _updated.getVariableNames()){
			if (!activeInPassed.containsVariable(varName)) {
				_warnSet.addVariable(varName, _updated.getVariable(varName));
			}
		}
		
		// activeOut includes variables from passed live in and updated in the while body
		_liveOut = new VariableSet();
		_liveOut.addVariables(current);
		_liveOut.addVariables(_updated);
		return _liveOut;
	}

	@Override
	public VariableSet initializebackwardLV(VariableSet loPassed) {
		
		WhileStatement wstmt = (WhileStatement)_statements.get(0);
			
		VariableSet lo = new VariableSet();
		lo.addVariables(loPassed);
		
		// calls analyze for each statement block in while stmt body
		int numBlocks = wstmt.getBody().size();
		for (int i = numBlocks - 1; i >= 0; i--){
			lo = wstmt.getBody().get(i).analyze(lo);
		}	
		
		VariableSet loReturn = new VariableSet();
		loReturn.addVariables(lo);
		return loReturn;
	
	}
	
	public void setPredicateHops(Hop hops) {
		_predicateHops = hops;
	}
	
	public Hop getPredicateHops(){
		return _predicateHops;
	}
	
	public Lop getPredicateLops() {
		return _predicateLops;
	}

	public void setPredicateLops(Lop predicateLops) {
		_predicateLops = predicateLops;
	}
	
	public ArrayList<String> getInputstoSB() {
		// By calling getInputstoSB on all the child statement blocks,
		// we remove the variables only read in the while predicate but
		// never used in the body from the input list.
		HashSet<String> inputs = new HashSet<>();
		WhileStatement fstmt = (WhileStatement)_statements.get(0);
		for (StatementBlock sb : fstmt.getBody())
			inputs.addAll(sb.getInputstoSB());
		// Hashset ensures no duplicates in the variable list
		return new ArrayList<>(inputs);
	}
	
	@Override
	public VariableSet analyze(VariableSet loPassed) {
		VariableSet predVars = new VariableSet();
		predVars.addVariables(((WhileStatement)_statements.get(0)).getConditionalPredicate().variablesRead());
		predVars.addVariables(((WhileStatement)_statements.get(0)).getConditionalPredicate().variablesUpdated());
		
		VariableSet candidateLO = new VariableSet();
		candidateLO.addVariables(loPassed);
		candidateLO.addVariables(_gen);
		candidateLO.addVariables(predVars);
		
		VariableSet origLiveOut = new VariableSet();
		origLiveOut.addVariables(_liveOut);
		origLiveOut.addVariables(predVars);
		origLiveOut.addVariables(_gen);
		
		_liveOut = new VariableSet();
	 	for (String name : candidateLO.getVariableNames()){
	 		if (origLiveOut.containsVariable(name)){
	 			_liveOut.addVariable(name, candidateLO.getVariable(name));
	 		}
	 	}
	 	
		initializebackwardLV(_liveOut);
		
		// set final warnSet: remove variables NOT in live out
		VariableSet finalWarnSet = new VariableSet();
		for (String varName : _warnSet.getVariableNames()){
			if (_liveOut.containsVariable(varName)){
				finalWarnSet.addVariable(varName,_warnSet.getVariable(varName));
			}
		}
		_warnSet = finalWarnSet;
		
		// for now just print the warn set
		for (String varName : _warnSet.getVariableNames()){
			LOG.warn(_warnSet.getVariable(varName).printWarningLocation() + "Initialization of " + varName + " depends on while execution");
		}
		
		// Cannot remove kill variables
		_liveIn = new VariableSet();
		_liveIn.addVariables(_liveOut);
		_liveIn.addVariables(_gen);
		
		VariableSet liveInReturn = new VariableSet();
		liveInReturn.addVariables(_liveIn);
		
		return liveInReturn;
	}

	@Override
	public void updateRepetitionEstimates(double repetitions){
		this.repetitions = repetitions * DEFAULT_LOOP_REPETITIONS;
		if ( getPredicateHops() != null )
			getPredicateHops().updateRepetitionEstimates(this.repetitions);
		for(Statement statement : getStatements()) {
			List<StatementBlock> children = ((WhileStatement)statement).getBody();
			for ( StatementBlock stmBlock : children ){
				stmBlock.updateRepetitionEstimates(this.repetitions);
			}
		}
	}
	
	/////////
	// materialized hops recompilation flags
	////
	
	public boolean updatePredicateRecompilationFlag() {
		return (_requiresPredicateRecompile = 
			ConfigurationManager.isDynamicRecompilation() 
			&& Recompiler.requiresRecompilation(getPredicateHops()));
	}
	
	public boolean requiresPredicateRecompilation() {
		return _requiresPredicateRecompile;
	}
}
