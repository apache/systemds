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
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.recompile.Recompiler;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.runtime.instructions.cp.BooleanObject;
import org.apache.sysds.runtime.instructions.cp.DoubleObject;


public class ForStatementBlock extends StatementBlock 
{
	
	protected Hop _fromHops        = null;
	protected Hop _toHops          = null;
	protected Hop _incrementHops   = null;
	
	protected Lop _fromLops        = null;
	protected Lop _toLops          = null;
	protected Lop _incrementLops   = null;

	protected boolean _requiresFromRecompile      = false;
	protected boolean _requiresToRecompile        = false;
	protected boolean _requiresIncrementRecompile = false;
	
	
	public IterablePredicate getIterPredicate(){
		return ((ForStatement)_statements.get(0)).getIterablePredicate();
	}

	@Override
	public VariableSet validate(DMLProgram dmlProg, VariableSet ids, HashMap<String,ConstIdentifier> constVars, boolean conditional) 
	{
		if (_statements.size() > 1){
			raiseValidateError("ForStatementBlock should have only 1 statement (for statement)", conditional);
		}
		ForStatement fs = (ForStatement) _statements.get(0);
		IterablePredicate predicate = fs.getIterablePredicate();
		
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
		
		predicate.validateExpression(ids.getVariables(), constVars, conditional);
		ArrayList<StatementBlock> body = fs.getBody();
		
		//perform constant propagation for ( from, to, incr )
		//(e.g., useful for reducing false positives in parfor dependency analysis)
		if( !conditional )
			performConstantPropagation(constVars);
		
		//validate body
		_dmlProg = dmlProg;
		for(StatementBlock sb : body) {
			ids = sb.validate(dmlProg, ids, constVars, true);
			constVars = sb.getConstOut();
		}
		
		if (!body.isEmpty()){
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
					raiseValidateError("ForStatementBlock has unsupported conditional data type change of variable '"+key+"' in loop body.", conditional);
				}
				
				//handle size change
				long startVersionDim1 = (startVersion instanceof IndexedIdentifier)   ? ((IndexedIdentifier)startVersion).getOrigDim1() : startVersion.getDim1(); 
				long endVersionDim1 = (endVersion instanceof IndexedIdentifier) ? ((IndexedIdentifier)endVersion).getOrigDim1() : endVersion.getDim1(); 
				long startVersionDim2 = (startVersion instanceof IndexedIdentifier)   ? ((IndexedIdentifier)startVersion).getOrigDim2() : startVersion.getDim2(); 
				long endVersionDim2 = (endVersion instanceof IndexedIdentifier) ? ((IndexedIdentifier)endVersion).getOrigDim2() : endVersion.getDim2(); 
				boolean sizeUnchanged = ((startVersionDim1 == endVersionDim1) && (startVersionDim2 == endVersionDim2) );
				
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
		if (revalidationRequired){
		
			// update ids to the reconciled values
			ids = origVarsBeforeBody;
			
			//////////////////////////////////////////////////////////////////////////////
			// SECOND PASS: process the predicate / statement blocks in the body of the for statement
			///////////////////////////////////////////////////////////////////////////////
			
			//remove updated vars from constants
			for( String var : _updated.getVariableNames() )
				if( constVars.containsKey( var ) )
					constVars.remove( var );
			
			//perform constant propagation for ( from, to, incr )
			//(e.g., useful for reducing false positives in parfor dependency analysis)
			if( !conditional )
				performConstantPropagation(constVars);
			
			predicate.validateExpression(ids.getVariables(), constVars, conditional);
			body = fs.getBody();
			
			//validate body
			_dmlProg = dmlProg;
			for(StatementBlock sb : body) {
				ids = sb.validate(dmlProg, ids, constVars, true);
				constVars = sb.getConstOut();
			}
			if (!body.isEmpty()){
				_constVarsIn.putAll(body.get(0).getConstIn());
				_constVarsOut.putAll(body.get(body.size()-1).getConstOut());
			}
		}
		
		return ids;
	}
	
	@Override
	public VariableSet initializeforwardLV(VariableSet activeInPassed) {
		
		ForStatement fstmt = (ForStatement)_statements.get(0);
		if (_statements.size() > 1){
			throw new LanguageException(_statements.get(0).printErrorLocation() + "ForStatementBlock should have only 1 statement (for statement)");
		}
		
		_read = new VariableSet();
		_read.addVariables(fstmt.getIterablePredicate().variablesRead());
		_updated.addVariables(fstmt.getIterablePredicate().variablesUpdated());
		
		_gen = new VariableSet();
		_gen.addVariables(fstmt.getIterablePredicate().variablesRead());

		// add the iterVar from iterable predicate to kill set 
		_kill.addVariables(fstmt.getIterablePredicate().variablesUpdated());
		
		VariableSet current = new VariableSet();
		current.addVariables(activeInPassed);
		current.addVariables(_updated);
		
		
		for( StatementBlock sb : fstmt.getBody())
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
		
		ForStatement fstmt = (ForStatement)_statements.get(0);
			
		VariableSet lo = new VariableSet();
		lo.addVariables(loPassed);
		
		// calls analyze for each statement block in while stmt body
		int numBlocks = fstmt.getBody().size();
		for (int i = numBlocks - 1; i >= 0; i--){
			lo = fstmt.getBody().get(i).analyze(lo);
		}	
		
		VariableSet loReturn = new VariableSet();
		loReturn.addVariables(lo);
		return loReturn;
	
	}
	
	public void setFromHops(Hop hops) { _fromHops = hops; }
	public void setToHops(Hop hops) { _toHops = hops; }
	public void setIncrementHops(Hop hops) { _incrementHops = hops; }
	
	public Hop getFromHops()      { return _fromHops; }
	public Hop getToHops()        { return _toHops; }
	public Hop getIncrementHops() { return _incrementHops; }

	public void setFromLops(Lop lops) { 
		_fromLops = lops; 
	}
	public void setToLops(Lop lops) { _toLops = lops; }
	public void setIncrementLops(Lop lops) { _incrementLops = lops; }
	
	public Lop getFromLops()      { return _fromLops; }
	public Lop getToLops()        { return _toLops; }
	public Lop getIncrementLops() { return _incrementLops; }

	public ArrayList<String> getInputstoSB() {
		// By calling getInputstoSB on all the child statement blocks,
		// we remove the variables only read in the for predicate but
		// never used in the body from the input list.
		HashSet<String> inputs = new HashSet<>();
		ForStatement fstmt = (ForStatement)_statements.get(0);
		for (StatementBlock sb : fstmt.getBody())
			inputs.addAll(sb.getInputstoSB());
		// Hashset ensures no duplicates in the variable list
		return new ArrayList<>(inputs);
	}

	@Override
	public VariableSet analyze(VariableSet loPassed) {
 		
		VariableSet predVars = new VariableSet();
		IterablePredicate ip = ((ForStatement)_statements.get(0)).getIterablePredicate(); 
		
		predVars.addVariables(ip.variablesRead());
		predVars.addVariables(ip.variablesUpdated());
		
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
		for (String varName : _warnSet.getVariableNames()) {
			if( !ip.getIterVar().getName().equals( varName)  )
				LOG.warn(_warnSet.getVariable(varName).printWarningLocation() + "Initialization of " + varName + " depends on for execution");
		}
		
		// Cannot remove kill variables
		_liveIn = new VariableSet();
		_liveIn.addVariables(_liveOut);
		_liveIn.addVariables(_gen);
		
		VariableSet liveInReturn = new VariableSet();
		liveInReturn.addVariables(_liveIn);
		
		return liveInReturn;
	}
	

	public void performConstantPropagation(HashMap<String, ConstIdentifier> currConstVars) {
		IterablePredicate ip = getIterPredicate();
		
		// handle replacement in from expression
		Expression replacementExpr = replaceConstantVar(ip.getFromExpr(), currConstVars); 
		if (replacementExpr != null)
			ip.setFromExpr(replacementExpr);
		
		// handle replacement in to expression
		replacementExpr = replaceConstantVar(ip.getToExpr(), currConstVars);  
		if (replacementExpr != null)
			ip.setToExpr(replacementExpr);
		
		// handle replacement in increment expression
		replacementExpr = replaceConstantVar(ip.getIncrementExpr(), currConstVars);
		if (replacementExpr != null)
			ip.setIncrementExpr(replacementExpr);
	}
	
	private static Expression replaceConstantVar(Expression expr, HashMap<String, ConstIdentifier> currConstVars)
	{
		Expression ret = null;
		
		if (expr instanceof DataIdentifier && !(expr instanceof IndexedIdentifier)) 
		{	
			// check if the DataIdentifier variable is a ConstIdentifier
			String identifierName = ((DataIdentifier) expr).getName();
			if (currConstVars.containsKey(identifierName)) {
				ConstIdentifier constValue = currConstVars.get(identifierName);
				// AUTO CASTING (using runtime operations for consistency)
				switch (constValue.getValueType()) {
				case FP64:
					ret = new IntIdentifier(new DoubleObject(((DoubleIdentifier) constValue).getValue()).getLongValue(),
							expr);
					break;
				case INT64:
					ret = new IntIdentifier((IntIdentifier) constValue, expr);
					break;
				case BITSET:
					ret = new IntIdentifier(
							new BooleanObject(((BooleanIdentifier) constValue).getValue()).getLongValue(), expr);
					break;
				default:
					// do nothing
				}
			}
		}
		else
		{
			//do nothing, cannot replace full expression
			ret = expr;
		}
		
		return ret;
	}
	
	/////////
	// materialized hops recompilation flags
	////
	
	public boolean updatePredicateRecompilationFlags() {
		if( ConfigurationManager.isDynamicRecompilation() ) {
			_requiresFromRecompile = Recompiler.requiresRecompilation(getFromHops());
			_requiresToRecompile = Recompiler.requiresRecompilation(getToHops());
			_requiresIncrementRecompile = Recompiler.requiresRecompilation(getIncrementHops());
		}
		return requiresPredicateRecompilation();
	}
	
	public boolean requiresPredicateRecompilation() {
		return (_requiresFromRecompile || _requiresToRecompile || _requiresIncrementRecompile);
	}
	
	public boolean requiresFromRecompilation() {
		return _requiresFromRecompile;
	}
	
	public boolean requiresToRecompilation() {
		return _requiresToRecompile;
	}
	
	public boolean requiresIncrementRecompilation() {
		return _requiresIncrementRecompile;
	}

	public int getEstimateReps(){
		if(_fromHops != null && _toHops != null && _incrementHops != null){
			List<Hop> fr = _fromHops.getInput();
			List<Hop> to = _toHops.getInput();
			List<Hop> in = _incrementHops.getInput();
			if (fr.size() == 1 && to.size() == 1 && in.size()== 1 && 
				fr.get(0) instanceof LiteralOp && to.get(0) instanceof LiteralOp && in.get(0) instanceof LiteralOp){
				long f = ((LiteralOp)fr.get(0)).getLongValue();
				long t = ((LiteralOp)to.get(0)).getLongValue();
				long i = ((LiteralOp)in.get(0)).getLongValue();
				return (int)((t-f)/i) + 1;
			}
		}
		
		return (int) DEFAULT_LOOP_REPETITIONS;
	}

	@Override
	public void updateRepetitionEstimates(double repetitions){
		this.repetitions = repetitions * getEstimateReps();
		if ( _fromHops != null )
			_fromHops.updateRepetitionEstimates(this.repetitions);
		if ( _toHops != null )
			_toHops.updateRepetitionEstimates(this.repetitions);
		if ( _incrementHops != null )
			_incrementHops.updateRepetitionEstimates(this.repetitions);
		for(Statement statement : getStatements()) {
			List<StatementBlock> children = ((ForStatement) statement).getBody();
			for ( StatementBlock stmBlock : children ){
				stmBlock.updateRepetitionEstimates(this.repetitions);
			}
		}
	}
}
