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
import java.util.Map.Entry;

import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.recompile.Recompiler;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.common.Types.DataType;


public class IfStatementBlock extends StatementBlock 
{
	private Hop _predicateHops;
	private Lop _predicateLops = null;
	private boolean _requiresPredicateRecompile = false;
	
	@Override
	public VariableSet validate(DMLProgram dmlProg, VariableSet ids, HashMap<String,ConstIdentifier> constVars, boolean conditional) 
	{
		if (_statements.size() > 1){
			raiseValidateError("IfStatementBlock should only have 1 statement (IfStatement)", conditional);
		}
		
		IfStatement ifstmt = (IfStatement) _statements.get(0);
		
		//validate conditional predicate (incl constant propagation)
		Expression pred = ifstmt.getConditionalPredicate().getPredicate();
		pred.validateExpression(ids.getVariables(), constVars, conditional);
		if( pred instanceof DataIdentifier && constVars.containsKey( ((DataIdentifier)pred).getName()) && !conditional ) {
			ifstmt.getConditionalPredicate().setPredicate(constVars.get(((DataIdentifier)pred).getName()));
		}
		
		HashMap<String,ConstIdentifier> constVarsIfCopy = new HashMap<>(constVars);
		HashMap<String,ConstIdentifier> constVarsElseCopy = new HashMap<> (constVars);
		
		VariableSet idsIfCopy   = new VariableSet(ids);
		VariableSet idsElseCopy = new VariableSet(ids);
		VariableSet idsOrigCopy = new VariableSet(ids);

		// handle if stmt body
		_dmlProg = dmlProg;
		ArrayList<StatementBlock> ifBody = ifstmt.getIfBody();
		for(StatementBlock sb : ifBody){ //conditional exec
			idsIfCopy = sb.validate(dmlProg, idsIfCopy, constVarsIfCopy, true);
			constVarsIfCopy = sb.getConstOut();
		}
		
		// handle else stmt body
		ArrayList<StatementBlock> elseBody = ifstmt.getElseBody();
		for(StatementBlock sb : elseBody){ //conditional exec
			idsElseCopy = sb.validate(dmlProg,idsElseCopy, constVarsElseCopy, true);
			constVarsElseCopy = sb.getConstOut();
		}
		
		
		/////////////////////////////////////////////////////////////////////////////////
		//  check data type and value type are same for updated variables in both 
		//	if statement and else statement
		//  (reject conditional data type change)
		/////////////////////////////////////////////////////////////////////////////////
		for (String updatedVar : this._updated.getVariableNames()){
			DataIdentifier origVersion  = idsOrigCopy.getVariable(updatedVar);
			DataIdentifier ifVersion 	= idsIfCopy.getVariable(updatedVar);
			DataIdentifier elseVersion  = idsElseCopy.getVariable(updatedVar);
			
			//data type handling: reject conditional data type change
			if( ifVersion != null && elseVersion != null ) //both branches exist
			{
				if (!ifVersion.getOutput().getDataType().equals(elseVersion.getOutput().getDataType())){
					raiseValidateError("IfStatementBlock has unsupported conditional data type change of variable '"+updatedVar+"' in if/else branch.", conditional);
				}	
			}
			else if( origVersion !=null ) //only if branch exists
			{
				if (!ifVersion.getOutput().getDataType().equals(origVersion.getOutput().getDataType())){
					raiseValidateError("IfStatementBlock has unsupported conditional data type change of variable '"+updatedVar+"' in if branch.", conditional);
				}
			}
			
			//value type handling		
			if (ifVersion != null && elseVersion != null && !ifVersion.getOutput().getValueType().equals(elseVersion.getOutput().getValueType())){
				LOG.warn(elseVersion.printWarningLocation() + "Variable " + elseVersion.getName() + " defined with different value type in if and else clause.");
			}
		}
		
		// handle constant variable propagation -- (IF UNION ELSE) MINUS updated vars
		
		//////////////////////////////////////////////////////////////////////////////////
		// handle constant variables 
		// 1) (IF UNION ELSE) MINUS updated const vars
		// 2) reconcile updated const vars
		// 		a) IF updated const variables have same value and datatype in both if / else branch, THEN set updated size to updated size
		//		b) ELSE leave out of reconciled set
		/////////////////////////////////////////////////////////////////////////////////
		
		HashMap<String,ConstIdentifier> recConstVars = new HashMap<>();
		
		// STEP 1:  (IF UNION ELSE) MINUS updated vars
		for (Entry<String,ConstIdentifier> e : constVarsIfCopy.entrySet() ){
			String varName = e.getKey();
			if (!_updated.containsVariable(varName))
				recConstVars.put(varName, e.getValue());
		}
		for (Entry<String,ConstIdentifier> e : constVarsElseCopy.entrySet() ){
			String varName = e.getKey();
			if (!_updated.containsVariable(varName))
				recConstVars.put(varName, e.getValue());
		}
		
		
		// STEP 2: check that updated const values have in both if / else branches 
		//		a) same data type, 
		//		b) same value type (SCALAR),
		//		c) same value
		for (String updatedVar : _updated.getVariableNames()){
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
					if ( ((StringIdentifier)ifConstVersion).getValue().equals(((StringIdentifier) elseConstVersion).getValue()) )
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
			if (!_updated.containsVariable(varName))
				recVars.addVariable(varName,idsIfCopy.getVariable(varName));
		}
		for (String varName : idsElseCopy.getVariableNames()){
			if (!_updated.containsVariable(varName))
				recVars.addVariable(varName,idsElseCopy.getVariable(varName));
		}
		
		// STEP 2: reconcile size of updated variables
		for (String updatedVar : _updated.getVariableNames()){
			DataIdentifier ifVersion 	= idsIfCopy.getVariable(updatedVar);
			DataIdentifier elseVersion  = idsElseCopy.getVariable(updatedVar);
			DataIdentifier origVersion = idsOrigCopy.getVariable(updatedVar);
			
			if (ifVersion != null && elseVersion != null) {
				long updatedDim1 = -1, updatedDim2 = -1;
				long updatedNnz = -1; 
				
				long ifVersionDim1 		= (ifVersion instanceof IndexedIdentifier)   ? ((IndexedIdentifier)ifVersion).getOrigDim1() : ifVersion.getDim1(); 
				long elseVersionDim1	= (elseVersion instanceof IndexedIdentifier) ? ((IndexedIdentifier)elseVersion).getOrigDim1() : elseVersion.getDim1(); 
				
				long ifVersionDim2 		= (ifVersion instanceof IndexedIdentifier)   ? ((IndexedIdentifier)ifVersion).getOrigDim2() : ifVersion.getDim2(); 
				long elseVersionDim2	= (elseVersion instanceof IndexedIdentifier) ? ((IndexedIdentifier)elseVersion).getOrigDim2() : elseVersion.getDim2(); 
				
				if( ifVersionDim1 == elseVersionDim1 ){
					updatedDim1 = ifVersionDim1;
				}
				if( ifVersionDim2 == elseVersionDim2 ){
					updatedDim2 = ifVersionDim2;
				}
				
				
				//NOTE: nnz not propagated via validate, and hence, we conservatively assume that nnz have been changed.
				//if( ifVersion.getNnz() == elseVersion.getNnz() ){
				//	updatedNnz = ifVersion.getNnz();
				//}
				
				// add reconsiled version (deep copy of ifVersion, cast as DataIdentifier)
				DataIdentifier recVersion = new DataIdentifier(ifVersion);
				recVersion.setDimensions(updatedDim1, updatedDim2);
				recVersion.setNnz(updatedNnz);
				recVars.addVariable(updatedVar, recVersion);
			}
			else {
				// CASE: defined only if branch
				DataIdentifier recVersion = null;
				if (ifVersion != null){
					// add reconciled version (deep copy of ifVersion, cast as DataIdentifier)
					recVersion = new DataIdentifier(ifVersion);
					recVars.addVariable(updatedVar, recVersion);
				}
				// CASE: defined only else branch
				else if (elseVersion != null){
					// add reconciled version (deep copy of elseVersion, cast as DataIdentifier)
					recVersion = new DataIdentifier(elseVersion);
					recVars.addVariable(updatedVar, recVersion);
				}
				// CASE: updated, but not in either if or else branch
				else {
					// add reconciled version (deep copy of elseVersion, cast as DataIdentifier)
					recVersion = new DataIdentifier(_updated.getVariable(updatedVar));
					recVars.addVariable(updatedVar, recVersion);
				}
				
				
				long updatedDim1 = -1, updatedDim2 = -1;
				long updatedNnz = -1; 
				
				if( origVersion != null ) {
					long origVersionDim1 = (origVersion instanceof IndexedIdentifier)   ? ((IndexedIdentifier)origVersion).getOrigDim1() : origVersion.getDim1(); 
					long recVersionDim1	 = recVersion.getDim1(); //always DataIdentifier (see above)
					long origVersionDim2 = (origVersion instanceof IndexedIdentifier)   ? ((IndexedIdentifier)origVersion).getOrigDim2() : origVersion.getDim2(); 
					long recVersionDim2	 = recVersion.getDim2(); //always DataIdentifier (see above) 
					
					if( origVersionDim1 == recVersionDim1 ){
						updatedDim1 = origVersionDim1;
					}
					if( origVersionDim2 == recVersionDim2 ){
						updatedDim2 = origVersionDim2;
					}
					//NOTE: nnz not propagated via validate, and hence, we conservatively assume that nnz have been changed.
					//if( origVersion.getNnz() == recVersion.getNnz() ){
					//	updatedNnz = recVersion.getNnz();
					//}
				}
				
				recVersion.setDimensions(updatedDim1, updatedDim2);
				recVersion.setNnz(updatedNnz);
			}
		}
		
		// propagate updated variables
		VariableSet allIdVars = new VariableSet(recVars);
		
		_constVarsIn.putAll(constVars);
		_constVarsOut.putAll(recConstVars);
		
		return allIdVars;
	}
	
	@Override
	public VariableSet initializeforwardLV(VariableSet activeInPassed) 
	{
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
		VariableSet ifCurrent = new VariableSet(activeInPassed);
		VariableSet genIfBody = new VariableSet();
		VariableSet killIfBody = new VariableSet();
		VariableSet updatedIfBody = new VariableSet();
		VariableSet readIfBody = new VariableSet();
		
		for (StatementBlock sb : ifstmt.getIfBody())
		{		
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
			if (!(sb instanceof WhileStatementBlock) && !(sb instanceof ForStatementBlock) ){
				killIfBody.addVariables(sb._kill);
			}	
		}
			
		///////////////////////////////////////////////////////////////////////
		//  ELSE STATEMENT
		///////////////////////////////////////////////////////////////////////
		
		// initialize forward for each statement block in if body
		VariableSet elseCurrent = new VariableSet(activeInPassed);
		VariableSet genElseBody = new VariableSet();
		VariableSet killElseBody = new VariableSet();
		VariableSet updatedElseBody = new VariableSet();
		VariableSet readElseBody = new VariableSet();
		
		// initialize forward for each statement block in else body
		for (StatementBlock sb : ifstmt.getElseBody())
		{	
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
			if (!(sb instanceof WhileStatementBlock) && !(sb instanceof ForStatementBlock) ){
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

	@Override
	public VariableSet initializebackwardLV(VariableSet loPassed) 
	{
		IfStatement ifstmt = (IfStatement)_statements.get(0);
		if (_statements.size() > 1){
			throw new LanguageException(ifstmt.printErrorLocation() + "IfStatementBlock should have only 1 statement (if statement)");
		}
		
		VariableSet currentLiveOutIf = new VariableSet(loPassed);
		VariableSet currentLiveOutElse = new VariableSet(loPassed);
			
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

	@Override
	public VariableSet analyze(VariableSet loPassed) {
	 	
		VariableSet predVars = ((IfStatement)_statements.get(0)).getConditionalPredicate().variablesRead();
		predVars.addVariables(((IfStatement)_statements.get(0)).getConditionalPredicate().variablesUpdated());
		
	 	VariableSet candidateLO = new VariableSet(loPassed);
	 	VariableSet origLiveOut = new VariableSet(_liveOut);
	 	
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
		
		//data flow equation: liveout = gen \cup (liveout - kill)
		_liveIn = new VariableSet();
		_liveIn.addVariables(_liveOut);
		_liveIn.removeVariables(_kill);
		_liveIn.addVariables(_gen);
		
		VariableSet liveInReturn = new VariableSet();
		liveInReturn.addVariables(_liveIn);
		return liveInReturn;
	}

	@Override
	public void updateRepetitionEstimates(double repetitions){
		this.repetitions = repetitions;
		getPredicateHops().updateRepetitionEstimates(this.repetitions);
		for ( Statement statement : getStatements() ){
			IfStatement ifStatement = (IfStatement) statement;
			double blockLevelReps = repetitions / 2;
			for ( StatementBlock ifBodySB : ifStatement.getIfBody() )
				ifBodySB.updateRepetitionEstimates(blockLevelReps);
			for ( StatementBlock elseBodySB : ifStatement.getElseBody() )
				elseBodySB.updateRepetitionEstimates(blockLevelReps);
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
