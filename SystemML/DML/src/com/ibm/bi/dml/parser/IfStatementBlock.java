/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.parser;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.parser.Expression.DataType;


public class IfStatementBlock extends StatementBlock 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	private Hop _predicateHops;
	private Lop _predicateLops = null;
		
	public VariableSet validate(DMLProgram dmlProg, VariableSet ids, HashMap<String,ConstIdentifier> constVars) throws LanguageException, ParseException, IOException {
		
		if (_statements.size() > 1){
			throw new LanguageException(_statements.get(0).printErrorLocation() + "IfStatementBlock should only have 1 statement (IfStatement)");
		}
		IfStatement ifstmt = (IfStatement) _statements.get(0);
			
		ConditionalPredicate predicate = ifstmt.getConditionalPredicate();
		predicate.getPredicate().validateExpression(ids.getVariables(), constVars);
			
		HashMap<String,ConstIdentifier> constVarsIfCopy = new HashMap<String,ConstIdentifier> ();
		HashMap<String,ConstIdentifier> constVarsElseCopy = new HashMap<String,ConstIdentifier> ();
		HashMap<String,ConstIdentifier> constVarsOrigCopy = new HashMap<String,ConstIdentifier> ();
		
		for (String varName : constVars.keySet()){
			constVarsIfCopy.put(varName, constVars.get(varName));
			constVarsElseCopy.put(varName, constVars.get(varName));
			constVarsOrigCopy.put(varName, constVars.get(varName));
		}
		
		VariableSet idsIfCopy 	= new VariableSet();
		VariableSet idsElseCopy = new VariableSet();
		VariableSet	idsOrigCopy = new VariableSet();

		for (String varName : ids.getVariableNames()){
			idsIfCopy.addVariable(varName, ids.getVariable(varName));
			idsElseCopy.addVariable(varName, ids.getVariable(varName));
			idsOrigCopy.addVariable(varName, ids.getVariable(varName));
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
		
		// handle constant variable propogation -- (IF UNION ELSE) MINUS updated vars
		
		//////////////////////////////////////////////////////////////////////////////////
		// handle constant variables 
		// 1) (IF UNION ELSE) MINUS updated const vars
		// 2) reconcile updated const vars
		// 		a) IF updated const variables have same value and datatype in both if / else branch, THEN set updated size to updated size
		//		b) ELSE leave out of reconciled set
		/////////////////////////////////////////////////////////////////////////////////
		
		HashMap<String,ConstIdentifier> recConstVars = new HashMap<String,ConstIdentifier>();
		
		// STEP 1:  (IF UNION ELSE) MINUS updated vars
		for (String varName : constVarsIfCopy.keySet()){
			if (!this._updated.containsVariable(varName))
				recConstVars.put(varName,constVarsIfCopy.get(varName));
		}
		for (String varName : constVarsElseCopy.keySet()){
			if (!this._updated.containsVariable(varName))
				recConstVars.put(varName,constVarsElseCopy.get(varName));
		}
		
		// STEP 2: check that updated const values have in both if / else branches 
		//		a) same data type, 
		//		b) same value type (SCALAR),
		//		c) same value
		for (String updatedVar : this._updated.getVariableNames()){
			DataIdentifier ifVersion 	= idsIfCopy.getVariable(updatedVar);
			DataIdentifier elseVersion  = idsElseCopy.getVariable(updatedVar);
			if (ifVersion != null && elseVersion != null 
					&& ifVersion.getOutput().getDataType().equals(DataType.SCALAR) 
					&& elseVersion.getOutput().getDataType().equals(DataType.SCALAR) 
					&& ifVersion.getOutput().getValueType().equals(elseVersion.getOutput().getValueType()))
			{
				ConstIdentifier ifConstVersion   = constVarsIfCopy.get(updatedVar);
				ConstIdentifier elseConstVersion = constVarsElseCopy.get(updatedVar);
				// IntIdentifier
				if (ifConstVersion != null && elseConstVersion != null && ifConstVersion instanceof IntIdentifier && elseConstVersion instanceof IntIdentifier){
					if ( ((IntIdentifier)ifConstVersion).getValue() == ((IntIdentifier) elseConstVersion).getValue() )
						recConstVars.put(updatedVar, ifConstVersion);
				}
				// DoubleIdentifier
				else if (ifConstVersion != null && elseConstVersion != null && ifConstVersion instanceof DoubleIdentifier && elseConstVersion instanceof DoubleIdentifier){
					if ( ((DoubleIdentifier)ifConstVersion).getValue() == ((DoubleIdentifier) elseConstVersion).getValue() )
						recConstVars.put(updatedVar, ifConstVersion);
				}
				// Boolean 
				else if (ifConstVersion != null && elseConstVersion != null && ifConstVersion instanceof BooleanIdentifier && elseConstVersion instanceof BooleanIdentifier){
					if ( ((BooleanIdentifier)ifConstVersion).getValue() == ((BooleanIdentifier) elseConstVersion).getValue() )
						recConstVars.put(updatedVar, ifConstVersion);
				}
				
				// String
				else if (ifConstVersion != null && elseConstVersion != null && ifConstVersion instanceof StringIdentifier && elseConstVersion instanceof StringIdentifier){
					if ( ((BooleanIdentifier)ifConstVersion).getValue() == ((BooleanIdentifier) elseConstVersion).getValue() )
						recConstVars.put(updatedVar, ifConstVersion);
				}
				
			}
					
			
		}
		
		//////////////////////////////////////////////////////////////////////////////////
		// handle DataIdentifier variables 
		// 1) (IF UNION ELSE) MINUS updated vars
		// 2) reconcile size updated variables
		// 		a) IF updated variables have same size in both if / else branch, THEN set updated size to updated size
		//		b) ELSE  set size updated to (-1,-1)
		// 3) add updated vars to reconciled set
		/////////////////////////////////////////////////////////////////////////////////
		
		// STEP 1:  (IF UNION ELSE) MINUS updated vars
		VariableSet recVars = new VariableSet();
	
		for (String varName : idsIfCopy.getVariableNames()){
			if (!this._updated.containsVariable(varName))
				recVars.addVariable(varName,idsIfCopy.getVariable(varName));
		}
		for (String varName : idsElseCopy.getVariableNames()){
			if (!this._updated.containsVariable(varName))
				recVars.addVariable(varName,idsElseCopy.getVariable(varName));
		}
		
		// STEP 2: reconcile size of updated variables
		for (String updatedVar : this._updated.getVariableNames()){
			DataIdentifier ifVersion 	= idsIfCopy.getVariable(updatedVar);
			DataIdentifier elseVersion  = idsElseCopy.getVariable(updatedVar);
			
			if (ifVersion != null && elseVersion != null) {
				long updatedDim1 = -1, updatedDim2 = -1;
				 
				long ifVersionDim1 		= (ifVersion instanceof IndexedIdentifier)   ? ((IndexedIdentifier)ifVersion).getOrigDim1() : ifVersion.getDim1(); 
				long elseVersionDim1	= (elseVersion instanceof IndexedIdentifier) ? ((IndexedIdentifier)elseVersion).getOrigDim1() : elseVersion.getDim1(); 
				
				long ifVersionDim2 		= (ifVersion instanceof IndexedIdentifier)   ? ((IndexedIdentifier)ifVersion).getOrigDim2() : ifVersion.getDim2(); 
				long elseVersionDim2	= (elseVersion instanceof IndexedIdentifier) ? ((IndexedIdentifier)elseVersion).getOrigDim2() : elseVersion.getDim2(); 
				
				if (ifVersionDim1 == elseVersionDim1){
					updatedDim1 = ifVersionDim1;
				}
				if (ifVersionDim2 == elseVersionDim2){
					updatedDim2 = ifVersionDim2;
				}
				
				// add reconsiled version (deep copy of ifVersion, cast as DataIdentifier)
				DataIdentifier recVersion = new DataIdentifier(ifVersion);
				recVersion.setDimensions(updatedDim1, updatedDim2);
				recVars.addVariable(updatedVar, recVersion);
			}
			else {
				// CASE: defined only if branch
				if (ifVersion != null){
					// add reconciled version (deep copy of ifVersion, cast as DataIdentifier)
					DataIdentifier recVersion = new DataIdentifier(ifVersion);
					recVersion.setDimensions(-1, -1);
					recVars.addVariable(updatedVar, recVersion);
				}
				// CASE: defined only else branch
				else if (elseVersion != null){
					// add reconciled version (deep copy of elseVersion, cast as DataIdentifier)
					DataIdentifier recVersion = new DataIdentifier(elseVersion);
					recVersion.setDimensions(-1, -1);
					recVars.addVariable(updatedVar, recVersion);
				}
				// CASE: updated, but not in either if or else branch
				else {
					// add reconciled version (deep copy of elseVersion, cast as DataIdentifier)
					DataIdentifier recVersion = new DataIdentifier(_updated.getVariable(updatedVar));
					recVersion.setDimensions(-1, -1);
					recVars.addVariable(updatedVar, recVersion);
				}
			}
		}
		
		// propogate updated variables
		VariableSet allIdVars = new VariableSet();
		allIdVars.addVariables(recVars);
		
		_constVarsIn.putAll(constVars);
		_constVarsOut.putAll(recConstVars);
		
		return allIdVars;
	}
	
	public VariableSet initializeforwardLV(VariableSet activeInPassed) throws LanguageException {
		
		IfStatement ifstmt = (IfStatement)_statements.get(0);
		if (_statements.size() > 1){
			throw new LanguageException(ifstmt.printErrorLocation() + "IfStatementBlock should have only 1 statement (if statement)");
		}
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
		if (_statements.size() > 1){
			throw new LanguageException(ifstmt.printErrorLocation() + "IfStatementBlock should have only 1 statement (if statement)");
		}
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
	
	public void set_predicate_hops(Hop hops) {
		_predicateHops = hops;
	}
	
	public ArrayList<Hop> get_hops() throws HopsException{
	
		if (_hops != null && _hops.size() > 0){
			throw new HopsException(this.printBlockErrorLocation() + "error there should be no HOPs in IfStatementBlock");
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
	 	
		VariableSet predVars = ((IfStatement)_statements.get(0)).getConditionalPredicate().variablesRead();
		predVars.addVariables(((IfStatement)_statements.get(0)).getConditionalPredicate().variablesUpdated());
		
	 	VariableSet candidateLO = new VariableSet();
	 	//candidateLO.addVariables(_gen);
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
			LOG.warn(_warnSet.getVariable(varName).printWarningLocation() + "Initialization of " + varName + " depends on if-else execution");
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
