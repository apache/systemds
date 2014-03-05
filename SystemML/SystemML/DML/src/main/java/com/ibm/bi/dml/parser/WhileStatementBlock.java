/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.parser;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.compile.Recompiler;


public class WhileStatementBlock extends StatementBlock 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private Hop _predicateHops;
	private Lop _predicateLops = null;
	private boolean _requiresPredicateRecompile = false;
	
	
	public VariableSet validate(DMLProgram dmlProg, VariableSet ids, HashMap<String,ConstIdentifier> constVars) 
		throws LanguageException, ParseException, IOException 
	{	
		if (_statements.size() > 1){
			throw new LanguageException(_statements.get(0).printErrorLocation() + "WhileStatementBlock should have only 1 statement (while statement)");
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
		predicate.getPredicate().validateExpression(ids.getVariables(), constVars);
		ArrayList<StatementBlock> body = wstmt.getBody();
		
		_dmlProg = dmlProg;
		for(StatementBlock sb : body)
		{
			ids = sb.validate(dmlProg, ids, constVars);
			constVars = sb.getConstOut();
		}
				
		if (body.size() > 0) {
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
					String error = printErrorLocation() + "WhileStatementBlock has unsupported conditional data type change of variable '"+key+"' in loop body.";
					LOG.error(error); 
					throw new LanguageException(error);
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
			predicate.getPredicate().validateExpression(ids.getVariables(), constVars);
			body = wstmt.getBody();
			
			_dmlProg = dmlProg;
			for(StatementBlock sb : body)
			{
				ids = sb.validate(dmlProg, ids, constVars);
				constVars = sb.getConstOut();
			}
					
			if (body.size() > 0) {
				_constVarsIn.putAll(body.get(0).getConstIn());
				_constVarsOut.putAll(body.get(body.size()-1).getConstOut());
			}		
		}
		
		return ids;
	}


	public VariableSet initializeforwardLV(VariableSet activeInPassed) throws LanguageException {
		
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
		
		for (int  i = 0; i < wstmt.getBody().size(); i++){
			
			StatementBlock sb = wstmt.getBody().get(i);
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

	public VariableSet initializebackwardLV(VariableSet loPassed) throws LanguageException{
		
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
	
	public ArrayList<Hop> get_hops() throws HopsException {
		
		if (_hops != null && _hops.size() > 0){
			throw new HopsException(this._statements.get(0).printErrorLocation() + "there should be no HOPs associated with the WhileStatementBlock");
		}
		
		return _hops;
	}
	
	public Hop getPredicateHops(){
		return _predicateHops;
	}
	
	public Lop get_predicateLops() {
		return _predicateLops;
	}

	public void set_predicateLops(Lop predicateLops) {
		_predicateLops = predicateLops;
	}
	
	public VariableSet analyze(VariableSet loPassed) throws LanguageException{
	 		
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
			LOG.warn(   "***** WARNING: Initialization of " + varName + " on line " + _warnSet.getVariable(varName).getBeginLine() + " depends on while execution");
		}
		
		// Cannot remove kill variables
		_liveIn = new VariableSet();
		_liveIn.addVariables(_liveOut);
		_liveIn.addVariables(_gen);
		
		VariableSet liveInReturn = new VariableSet();
		liveInReturn.addVariables(_liveIn);
		
		return liveInReturn;
	}
	
	/////////
	// materialized hops recompilation flags
	////
	
	public void updatePredicateRecompilationFlag() 
		throws HopsException
	{
		_requiresPredicateRecompile =  OptimizerUtils.ALLOW_DYN_RECOMPILATION 
			                           && DMLScript.rtplatform == RUNTIME_PLATFORM.HYBRID	
			                           && Recompiler.requiresRecompilation(getPredicateHops());
	}
	
	public boolean requiresPredicateRecompilation()
	{
		return _requiresPredicateRecompile;
	}
}