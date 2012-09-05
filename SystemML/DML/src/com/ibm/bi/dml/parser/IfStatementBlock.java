package com.ibm.bi.dml.parser;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import com.ibm.bi.dml.hops.Hops;
import com.ibm.bi.dml.lops.Lops;
import com.ibm.bi.dml.utils.HopsException;
import com.ibm.bi.dml.utils.LanguageException;


public class IfStatementBlock extends StatementBlock {
	
	private Hops _predicateHops;
	private Lops _predicateLops = null;
		
	public VariableSet validate(DMLProgram dmlProg, VariableSet ids, HashMap<String,ConstIdentifier> constVars) throws LanguageException, ParseException, IOException {
		
		if (_statements.size() > 1)
			throw new LanguageException("IfStatementBlock should only have 1 statement (IfStatement)");
		
		IfStatement ifstmt = (IfStatement) _statements.get(0);
		
		// merge function calls if possible
		ifstmt.setIfBody(StatementBlock.mergeFunctionCalls(ifstmt.getIfBody(), dmlProg));
		ifstmt.setElseBody(StatementBlock.mergeFunctionCalls(ifstmt.getElseBody(), dmlProg));
		
		ConditionalPredicate predicate = ifstmt.getConditionalPredicate();
		predicate.getPredicate().validateExpression(ids.getVariables(), constVars);
		
		HashMap<String,ConstIdentifier> constVarsIfCopy = new HashMap<String,ConstIdentifier> ();
		HashMap<String,ConstIdentifier> constVarsElseCopy = new HashMap<String,ConstIdentifier> ();
		for (String varName : constVars.keySet()){
			constVarsIfCopy.put(varName, constVars.get(varName));
			constVarsElseCopy.put(varName, constVars.get(varName));
		}
		
		VariableSet idsIfCopy = new VariableSet();
		VariableSet idsElseCopy = new VariableSet();
		for (String varName : ids.getVariableNames()){
			idsIfCopy.addVariable(varName, ids.getVariable(varName));
			idsElseCopy.addVariable(varName, ids.getVariable(varName));
		}
		
		// handle if stmt body
		this._dmlProg = dmlProg;
		ArrayList<StatementBlock> ifBody = ifstmt.getIfBody();
		for(StatementBlock sb : ifBody){
			idsIfCopy = sb.validate(dmlProg, idsIfCopy, constVarsIfCopy);
			constVarsIfCopy = sb.getConstOut();
		}
		
		// handle else stmt body
		ArrayList<StatementBlock> elseBody = ifstmt.getElseBody();
		for(StatementBlock sb : elseBody){
			idsElseCopy = sb.validate(dmlProg,idsElseCopy, constVarsElseCopy);
			constVarsElseCopy = sb.getConstOut();
		}
		
		// need to reconcile both idsIfCopy and idsElseCopy 
		HashMap<String,ConstIdentifier> recConstVars = new HashMap<String,ConstIdentifier>();
		recConstVars.putAll(constVarsIfCopy);
		recConstVars.putAll(constVarsElseCopy);
		
		VariableSet allIdVars = new VariableSet();
		allIdVars.addVariables(idsIfCopy);
		allIdVars.addVariables(idsElseCopy);
		
		_constVarsIn.putAll(constVars);
		_constVarsOut.putAll(recConstVars);
		
		return allIdVars;
	}
	
	public VariableSet initializeforwardLV(VariableSet activeInPassed) throws LanguageException {
		
		IfStatement ifstmt = (IfStatement)_statements.get(0);
		if (_statements.size() > 1)
			throw new LanguageException("IfStatementBlock should have only 1 statement (if statement)");
		
		_read = new VariableSet();
		_gen = new VariableSet();
		_kill = new VariableSet();
		_warnSet = new VariableSet();
		
		///////////////////////////////////////////////////////////////////////
		// HANDLE PREDICATE
		///////////////////////////////////////////////////////////////////////
		_read.addVariables(ifstmt.getConditionalPredicate().variablesRead());
		_updated.addVariables(ifstmt.getConditionalPredicate().variablesUpdated());
		_gen.addVariables(ifstmt.getConditionalPredicate().variablesRead());
		
		///////////////////////////////////////////////////////////////////////
		//  IF STATEMENT
		///////////////////////////////////////////////////////////////////////
		
		// initialize forward for each statement block in if body
		VariableSet ifCurrent = new VariableSet();
		ifCurrent.addVariables(activeInPassed);
		VariableSet genIfBody = new VariableSet();
		VariableSet killIfBody = new VariableSet();
		VariableSet updatedIfBody = new VariableSet();
		VariableSet readIfBody = new VariableSet();
		
		for (StatementBlock sb : ifstmt.getIfBody()){
				
			ifCurrent = sb.initializeforwardLV(ifCurrent);
				
			// for each generated variable in this block, check variable not killed
			// (assigned value) in prior statement block in ifstmt blody
			for (String varName : sb._gen.getVariableNames()){
				
				// IF the variable is NOT set in the while loop PRIOR to this stmt block, 
				// THEN needs to be generated
				if (!killIfBody.getVariableNames().contains(varName)){
					genIfBody.addVariable(varName, sb._gen.getVariable(varName));	
				}
			}
				
			readIfBody.addVariables(sb._read);
			updatedIfBody.addVariables(sb._updated);
			
			// only add kill variables for statement blocks guaranteed to execute
			if (!(sb instanceof WhileStatementBlock) && !(sb instanceof IfStatementBlock) && !(sb instanceof ForStatementBlock) ){
				killIfBody.addVariables(sb._kill);
			}	
		}
			
		///////////////////////////////////////////////////////////////////////
		//  ELSE STATEMENT
		///////////////////////////////////////////////////////////////////////
		
		// initialize forward for each statement block in if body
		VariableSet elseCurrent = new VariableSet();
		elseCurrent.addVariables(activeInPassed);
		VariableSet genElseBody = new VariableSet();
		VariableSet killElseBody = new VariableSet();
		VariableSet updatedElseBody = new VariableSet();
		VariableSet readElseBody = new VariableSet();
		
		// initialize forward for each statement block in else body
		for (StatementBlock sb : ifstmt.getElseBody()){
			
			elseCurrent = sb.initializeforwardLV(elseCurrent);
			
			// for each generated variable in this block, check variable not killed
			// (assigned value) in prior statement block in ifstmt blody
			for (String varName : sb._gen.getVariableNames()){
				
				// IF the variable is NOT set in the while loop PRIOR to this stmt block, 
				// THEN needs to be generated
				if (!killElseBody.getVariableNames().contains(varName)){
					genElseBody.addVariable(varName, sb._gen.getVariable(varName));	
				}
			}
				
			readElseBody.addVariables(sb._read);
			updatedElseBody.addVariables(sb._updated);
			
			// only add kill variables for statement blocks guaranteed to execute
			if (!(sb instanceof WhileStatementBlock) && !(sb instanceof IfStatementBlock) && !(sb instanceof ForStatementBlock) ){
				killElseBody.addVariables(sb._kill);
			}
		}

		///////////////////////////////////////////////////////////////////////
		// PERFORM RECONCILIATION
		///////////////////////////////////////////////////////////////////////
		
		// "conservative" read -- union of read sets for if and else path	
		_read.addVariables(readIfBody);
		_read.addVariables(readElseBody);
		
		// "conservative" update -- union of updated 
		_updated.addVariables(updatedIfBody);
		_updated.addVariables(updatedElseBody);

		// "conservative" gen -- union of gen
		_gen.addVariables(genIfBody);
		_gen.addVariables(genElseBody);
		
		// "conservative" kill -- kill set is intersection of if-kill and else-kill
		for ( String varName : killIfBody.getVariableNames()){
			if (killElseBody.containsVariable(varName)){
				_kill.addVariable(varName, killIfBody.getVariable(varName));
			}
		}

		// set preliminary "warn" set -- variables that if used later may cause runtime error
		// if the loop is not executed
		// warnSet = (updated MINUS (updatedIfBody INTERSECT updatedElseBody)) MINUS current
		for (String varName : _updated.getVariableNames()){
			if (!((updatedIfBody.containsVariable(varName) && updatedElseBody.containsVariable(varName))
					|| activeInPassed.containsVariable(varName))) {
				_warnSet.addVariable(varName, _updated.getVariable(varName));
			}
		}
		
		
		// set activeOut to (if body current UNION else body current) UNION updated
		_liveOut = new VariableSet();
		_liveOut.addVariables(ifCurrent);
		_liveOut.addVariables(elseCurrent);
		_liveOut.addVariables(_updated);
		return _liveOut;
	}

	public VariableSet initializebackwardLV(VariableSet loPassed) throws LanguageException{
		
		IfStatement ifstmt = (IfStatement)_statements.get(0);
		if (_statements.size() > 1)
			throw new LanguageException("IfStatementBlock should have only 1 statement (if statement)");
		
		VariableSet currentLiveOutIf = new VariableSet();
		currentLiveOutIf.addVariables(loPassed);
		VariableSet currentLiveOutElse = new VariableSet();
		currentLiveOutElse.addVariables(loPassed);
			
		int numBlocks = ifstmt.getIfBody().size();
		for (int i = numBlocks - 1; i >= 0; i--){
			currentLiveOutIf = ifstmt.getIfBody().get(i).analyze(currentLiveOutIf);
		}
		
		numBlocks = ifstmt.getElseBody().size();
		for (int i = numBlocks - 1; i >= 0; i--){
			currentLiveOutElse = ifstmt.getElseBody().get(i).analyze(currentLiveOutElse);
		}
		
		// Any variable defined in either if-body or else-body is available for later use
		VariableSet bothPathsLiveOut = new VariableSet();
		bothPathsLiveOut.addVariables(currentLiveOutIf);
		bothPathsLiveOut.addVariables(currentLiveOutElse);
		
		return bothPathsLiveOut;
	
	}
	
	public void set_predicate_hops(Hops hops) {
		_predicateHops = hops;
	}
	
	public ArrayList<Hops> get_hops() throws HopsException{
	
		if (_hops != null && _hops.size() > 0){
			throw new HopsException("error there should be no HOPs in IfStatementBlock");
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
	 	
		VariableSet predVars = ((IfStatement)_statements.get(0)).getConditionalPredicate().variablesRead();
		predVars.addVariables(((IfStatement)_statements.get(0)).getConditionalPredicate().variablesUpdated());
		
	 	VariableSet candidateLO = new VariableSet();
	 	candidateLO.addVariables(_gen);
	 	candidateLO.addVariables(loPassed);
	 	
	 	VariableSet origLiveOut = new VariableSet();
	 	origLiveOut.addVariables(_liveOut);
	 	
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
			System.out.println(_warnSet.getVariable(varName).printWarningLocation() + "Initialization of " + varName + " depends on if-else execution");
		}
		
		_liveIn = new VariableSet();
		_liveIn.addVariables(_liveOut);
		_liveIn.removeVariables(_kill);
		_liveIn.addVariables(_gen);
		
		VariableSet liveInReturn = new VariableSet();
		liveInReturn.addVariables(_liveIn);
		return liveInReturn;
	}
	
}
