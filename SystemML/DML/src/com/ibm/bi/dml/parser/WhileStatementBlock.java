package com.ibm.bi.dml.parser;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import com.ibm.bi.dml.hops.Hops;
import com.ibm.bi.dml.lops.Lops;
import com.ibm.bi.dml.utils.HopsException;
import com.ibm.bi.dml.utils.LanguageException;


public class WhileStatementBlock extends StatementBlock {
	
	private Hops _predicateHops;
	private Lops _predicateLops = null;
	
	public VariableSet validate(DMLProgram dmlProg, VariableSet ids, HashMap<String,ConstIdentifier> constVars) throws LanguageException, ParseException, IOException {
		
		if (_statements.size() > 1)
			throw new LanguageException(_statements.get(0).printErrorLocation() + "WhileStatementBlock should have only 1 statement (while statement)");
		
		WhileStatement wstmt = (WhileStatement) _statements.get(0);
		ConditionalPredicate predicate = wstmt.getConditionalPredicate();
		
		// merge function calls if possible
		wstmt.setBody(StatementBlock.mergeFunctionCalls(wstmt.getBody(), dmlProg));
		
		// process the statement blocks in the body of the while statement
		predicate.getPredicate().validateExpression(ids.getVariables(), constVars);
		ArrayList<StatementBlock> body = wstmt.getBody();
		this._dmlProg = dmlProg;
		for(StatementBlock sb : body)
		{
			ids = sb.validate(dmlProg, ids, constVars);
			constVars = sb.getConstOut();
		}
		
		if (body.size() > 0) {
			_constVarsIn.putAll(body.get(0).getConstIn());
			_constVarsOut.putAll(body.get(body.size()-1).getConstOut());
		}
		return ids;
	}


	public VariableSet initializeforwardLV(VariableSet activeInPassed) throws LanguageException {
		
		WhileStatement wstmt = (WhileStatement)_statements.get(0);
		if (_statements.size() > 1)
			throw new LanguageException(_statements.get(0).printErrorLocation() + "WhileStatementBlock should have only 1 statement (while statement)");
		
		
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
			if (!(sb instanceof WhileStatementBlock) && !(sb instanceof IfStatementBlock) && !(sb instanceof ForStatementBlock) ){
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
	
	public void set_predicate_hops(Hops hops) {
		_predicateHops = hops;
	}
	
	public ArrayList<Hops> get_hops() throws HopsException {
		
		if (_hops != null && _hops.size() > 0){
			throw new HopsException(this._statements.get(0).printErrorLocation() + "there should be no HOPs associated with the WhileStatementBlock");
		}
		
		return _hops;
	}
	
	public Hops getPredicateHops(){
		return _predicateHops;
	}
	
	public Lops get_predicateLops() {
		return _predicateLops;
	}

	public void set_predicateLops(Lops predicateLops) {
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
			System.out.println(   "***** WARNING: Initialization of " + varName + " on line " + _warnSet.getVariable(varName).getBeginLine() + " depends on while execution");
		}
		
		// Cannot remove kill variables
		_liveIn = new VariableSet();
		_liveIn.addVariables(_liveOut);
		_liveIn.addVariables(_gen);
		
		VariableSet liveInReturn = new VariableSet();
		liveInReturn.addVariables(_liveIn);
		
		return liveInReturn;
	}
}