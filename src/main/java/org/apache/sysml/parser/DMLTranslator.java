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

package org.apache.sysml.parser;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.hops.AggBinaryOp;
import org.apache.sysml.hops.AggUnaryOp;
import org.apache.sysml.hops.BinaryOp;
import org.apache.sysml.hops.DataGenOp;
import org.apache.sysml.hops.DataOp;
import org.apache.sysml.hops.FunctionOp;
import org.apache.sysml.hops.FunctionOp.FunctionType;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.Hop.AggOp;
import org.apache.sysml.hops.Hop.DataGenMethod;
import org.apache.sysml.hops.Hop.DataOpTypes;
import org.apache.sysml.hops.Hop.Direction;
import org.apache.sysml.hops.Hop.OpOp2;
import org.apache.sysml.hops.Hop.OpOp3;
import org.apache.sysml.hops.Hop.ParamBuiltinOp;
import org.apache.sysml.hops.Hop.ReOrgOp;
import org.apache.sysml.hops.HopsException;
import org.apache.sysml.hops.IndexingOp;
import org.apache.sysml.hops.LeftIndexingOp;
import org.apache.sysml.hops.LiteralOp;
import org.apache.sysml.hops.MemoTable;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.hops.ParameterizedBuiltinOp;
import org.apache.sysml.hops.ReorgOp;
import org.apache.sysml.hops.TernaryOp;
import org.apache.sysml.hops.UnaryOp;
import org.apache.sysml.hops.ipa.InterProceduralAnalysis;
import org.apache.sysml.hops.rewrite.ProgramRewriter;
import org.apache.sysml.hops.recompile.Recompiler;
import org.apache.sysml.lops.Lop;
import org.apache.sysml.lops.LopsException;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.Expression.FormatType;
import org.apache.sysml.parser.Expression.ParameterizedBuiltinFunctionOp;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.parser.PrintStatement.PRINTTYPE;
import org.apache.sysml.runtime.DMLRuntimeException;


public class DMLTranslator 
{
	private static final Log LOG = LogFactory.getLog(DMLTranslator.class.getName());
	private DMLProgram _dmlProg = null;
	
	public DMLTranslator(DMLProgram dmlp) 
		throws DMLRuntimeException 
	{
		_dmlProg = dmlp;
		
		//setup default size for unknown dimensions
		OptimizerUtils.resetDefaultSize();
		//reinit rewriter according to opt level flags
		Recompiler.reinitRecompiler(); 
	}
	
	/**
	 * Validate parse tree
	 * 
	 * @throws LanguageException
	 * @throws IOException 
	 */
	public void validateParseTree(DMLProgram dmlp) 
		throws LanguageException, ParseException, IOException 
	{
		//STEP1: Pre-processing steps for validate - e.g., prepare read-after-write meta data
		boolean fWriteRead = prepareReadAfterWrite(dmlp, new HashMap<String, DataIdentifier>());
		
		//STEP2: Actual Validate
		// handle functions in namespaces (current program has default namespace)
		for (String namespaceKey : dmlp.getNamespaces().keySet()){
		
			// for each function defined in the namespace
			for (String fname :  dmlp.getFunctionStatementBlocks(namespaceKey).keySet()) {
				FunctionStatementBlock fblock = dmlp.getFunctionStatementBlock(namespaceKey,fname);
			
				HashMap<String, ConstIdentifier> constVars = new HashMap<String, ConstIdentifier>();
				VariableSet vs = new VariableSet();
			
				// add the input variables for the function to input variable list
				FunctionStatement fstmt = (FunctionStatement)fblock.getStatement(0);
				if (fblock.getNumStatements() > 1){
					LOG.error(fstmt.printErrorLocation() + "FunctionStatementBlock can only have 1 FunctionStatement");
					throw new LanguageException(fstmt.printErrorLocation() + "FunctionStatementBlock can only have 1 FunctionStatement");
				}
			
				for (DataIdentifier currVar : fstmt.getInputParams()) {	
					
					if (currVar.getDataType() == DataType.SCALAR){
						currVar.setDimensions(0, 0);
					}
					
					vs.addVariable(currVar.getName(), currVar);
				}
				fblock.validate(dmlp, vs, constVars, false);
			} 
		
		}	
		
		// handle regular blocks -- "main" program
		VariableSet vs = new VariableSet();
		HashMap<String, ConstIdentifier> constVars = new HashMap<String, ConstIdentifier>();
		for (int i = 0; i < dmlp.getNumStatementBlocks(); i++) {
			StatementBlock sb = dmlp.getStatementBlock(i);
			vs = sb.validate(dmlp, vs, constVars, fWriteRead);
			constVars = sb.getConstOut();
		}

		//STEP3: Post-processing steps after validate - e.g., prepare read-after-write meta data
		if( fWriteRead ) 
		{
			//propagate size and datatypes into read
			prepareReadAfterWrite(dmlp, new HashMap<String, DataIdentifier>());
		
			//re-validate main program for datatype propagation
			vs = new VariableSet();
			constVars = new HashMap<String, ConstIdentifier>();
			for (int i = 0; i < dmlp.getNumStatementBlocks(); i++) {
				StatementBlock sb = dmlp.getStatementBlock(i);
				vs = sb.validate(dmlp, vs, constVars, fWriteRead);
				constVars = sb.getConstOut();
			}	
		}
		
		return;
	}

	public void liveVariableAnalysis(DMLProgram dmlp) throws LanguageException {
	
		// for each namespace, handle function program blocks -- forward direction
		for (String namespaceKey : dmlp.getNamespaces().keySet()) {
			for (String fname: dmlp.getFunctionStatementBlocks(namespaceKey).keySet()) {
				FunctionStatementBlock fsb = dmlp.getFunctionStatementBlock(namespaceKey, fname);
				FunctionStatement fstmt = (FunctionStatement)fsb.getStatement(0);
				
				// perform function inlining
				fstmt.setBody(StatementBlock.mergeFunctionCalls(fstmt.getBody(), dmlp));
				
				VariableSet activeIn = new VariableSet();
				for (DataIdentifier id : fstmt.getInputParams()){
					activeIn.addVariable(id.getName(), id); 
				}
				fsb.initializeforwardLV(activeIn);
			}
		}
		
		// for each namespace, handle function program blocks -- backward direction
		for (String namespaceKey : dmlp.getNamespaces().keySet()) {	
			for (String fname: dmlp.getFunctionStatementBlocks(namespaceKey).keySet()) {
				
				// add output variables to liveout / activeout set
				FunctionStatementBlock fsb = dmlp.getFunctionStatementBlock(namespaceKey, fname);
				VariableSet currentLiveOut = new VariableSet();
				VariableSet currentLiveIn = new VariableSet();
				FunctionStatement fstmt = (FunctionStatement)fsb.getStatement(0);
				
				for (DataIdentifier id : fstmt.getInputParams())
					currentLiveIn.addVariable(id.getName(), id);
				
				for (DataIdentifier id : fstmt.getOutputParams())
					currentLiveOut.addVariable(id.getName(), id);
					
				fsb._liveOut = currentLiveOut;
				fsb.analyze(currentLiveIn, currentLiveOut);	
			}
		} 
		
		
		// handle regular program blocks 
		VariableSet currentLiveOut = new VariableSet();
		VariableSet activeIn = new VariableSet();
				
		// handle function inlining
		dmlp.setStatementBlocks(StatementBlock.mergeFunctionCalls(dmlp.getStatementBlocks(), dmlp));
		
		for (int i = 0; i < dmlp.getNumStatementBlocks(); i++) {
			StatementBlock sb = dmlp.getStatementBlock(i);
			activeIn = sb.initializeforwardLV(activeIn);
		}

		if (dmlp.getNumStatementBlocks() > 0){
			StatementBlock lastSb = dmlp.getStatementBlock(dmlp.getNumStatementBlocks() - 1);
			lastSb._liveOut = new VariableSet();
			for (int i = dmlp.getNumStatementBlocks() - 1; i >= 0; i--) {
				StatementBlock sb = dmlp.getStatementBlock(i);
				currentLiveOut = sb.analyze(currentLiveOut);
			}
		}
		return;

	}

	/**
	 * Construct Hops from parse tree
	 * 
	 * @throws ParseException
	 */
	public void constructHops(DMLProgram dmlp) 
		throws ParseException, LanguageException 
	{
		// Step 1: construct hops for all functions
		// for each namespace, handle function program blocks
		for (String namespaceKey : dmlp.getNamespaces().keySet()){		
			for (String fname: dmlp.getFunctionStatementBlocks(namespaceKey).keySet()) {
				FunctionStatementBlock current = dmlp.getFunctionStatementBlock(namespaceKey, fname);
				constructHops(current);
			}
		}
		
		// Step 2: construct hops for main program
		// handle regular program blocks
		for (int i = 0; i < dmlp.getNumStatementBlocks(); i++) {
			StatementBlock current = dmlp.getStatementBlock(i);
			constructHops(current);
		}
	}
		
	/**
	 * 
	 * @param dmlp
	 * @throws ParseException
	 * @throws LanguageException
	 * @throws HopsException
	 */
	public void rewriteHopsDAG(DMLProgram dmlp) 
		throws ParseException, LanguageException, HopsException 
	{
		//apply hop rewrites (static rewrites)
		ProgramRewriter rewriter = new ProgramRewriter(true, false);
		rewriter.rewriteProgramHopDAGs(dmlp);
		resetHopsDAGVisitStatus(dmlp);
		
		//propagate size information from main into functions (but conservatively)
		if( OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS ) {
			InterProceduralAnalysis ipa = new InterProceduralAnalysis();
		    ipa.analyzeProgram(dmlp);
		   	resetHopsDAGVisitStatus(dmlp);
		}
		
		//apply hop rewrites (dynamic rewrites, after IPA)
		ProgramRewriter rewriter2 = new ProgramRewriter(false, true);
		rewriter2.rewriteProgramHopDAGs(dmlp);
		resetHopsDAGVisitStatus(dmlp);
		
		// Compute memory estimates for all the hops. These estimates are used
		// subsequently in various optimizations, e.g. CP vs. MR scheduling and parfor.
		refreshMemEstimates(dmlp);
		resetHopsDAGVisitStatus(dmlp);
	}
	
	
	public void constructLops(DMLProgram dmlp) throws ParseException, LanguageException, HopsException, LopsException {

		// for each namespace, handle function program blocks handle function 
		for (String namespaceKey : dmlp.getNamespaces().keySet()){
			for (String fname: dmlp.getFunctionStatementBlocks(namespaceKey).keySet()) {
				FunctionStatementBlock current = dmlp.getFunctionStatementBlock(namespaceKey, fname);
				constructLops(current);
			}
		}
		
		// handle regular program blocks
		for (int i = 0; i < dmlp.getNumStatementBlocks(); i++) {
			StatementBlock current = dmlp.getStatementBlock(i);
			constructLops(current);
		}
	}
	
	/**
	 * 
	 * @param sb
	 * @throws HopsException
	 * @throws LopsException
	 */
	public void constructLops(StatementBlock sb) 
		throws HopsException, LopsException 
	{	
		if (sb instanceof WhileStatementBlock)
		{
			WhileStatementBlock wsb = (WhileStatementBlock)sb;
			WhileStatement whileStmt = (WhileStatement)wsb.getStatement(0);
			ArrayList<StatementBlock> body = whileStmt.getBody();
				
			if (sb.get_hops() != null && !sb.get_hops().isEmpty()) {
				LOG.error(sb.printBlockErrorLocation() + "WhileStatementBlock should not have hops");
				throw new HopsException(sb.printBlockErrorLocation() + "WhileStatementBlock should not have hops");
			}
			// step through stmt blocks in while stmt body
			for (StatementBlock stmtBlock : body){
				constructLops(stmtBlock);
			}
			
			// handle while stmt predicate
			Lop l = wsb.getPredicateHops().constructLops();
			wsb.set_predicateLops(l);	
			wsb.updatePredicateRecompilationFlag();
		}
		
		else if (sb instanceof IfStatementBlock)
		{
			IfStatementBlock isb = (IfStatementBlock) sb;
			IfStatement ifStmt = (IfStatement)isb.getStatement(0);
			ArrayList<StatementBlock> ifBody = ifStmt.getIfBody();
			ArrayList<StatementBlock> elseBody = ifStmt.getElseBody();
				
			if (sb.get_hops() != null && !sb.get_hops().isEmpty()){
				LOG.error(sb.printBlockErrorLocation() + "IfStatementBlock should not have hops");
				throw new HopsException(sb.printBlockErrorLocation() + "IfStatementBlock should not have hops");
			}
			// step through stmt blocks in if stmt ifBody
			for (StatementBlock stmtBlock : ifBody)
				constructLops(stmtBlock);
			
			// step through stmt blocks in if stmt elseBody
			for (StatementBlock stmtBlock : elseBody)
				constructLops(stmtBlock);
			
			// handle if stmt predicate
			Lop l = isb.getPredicateHops().constructLops();
			isb.set_predicateLops(l);
			isb.updatePredicateRecompilationFlag();
		}
		
		else if (sb instanceof ForStatementBlock) //NOTE: applies to ForStatementBlock and ParForStatementBlock
		{
			ForStatementBlock fsb =  (ForStatementBlock) sb;
			ForStatement fs = (ForStatement)sb.getStatement(0);
			ArrayList<StatementBlock> body = fs.getBody();
						
			if (sb.get_hops() != null && !sb.get_hops().isEmpty() ) {
				LOG.error(sb.printBlockErrorLocation() + "ForStatementBlock should not have hops");
				throw new HopsException(sb.printBlockErrorLocation() + "ForStatementBlock should not have hops");
			}
			// step through stmt blocks in FOR stmt body
			for (StatementBlock stmtBlock : body)
				constructLops(stmtBlock);
			
			// handle for stmt predicate
			if (fsb.getFromHops() != null){
				Lop llobs = fsb.getFromHops().constructLops();
				fsb.setFromLops(llobs);
			}
			if (fsb.getToHops() != null){
				Lop llobs = fsb.getToHops().constructLops();
				fsb.setToLops(llobs);
			}
			if (fsb.getIncrementHops() != null){
				Lop llobs = fsb.getIncrementHops().constructLops();
				fsb.setIncrementLops(llobs);
			}
			fsb.updatePredicateRecompilationFlags();
		}
		else if (sb instanceof FunctionStatementBlock){
			FunctionStatement functStmt = (FunctionStatement)sb.getStatement(0);
			ArrayList<StatementBlock> body = functStmt.getBody();
			
			if (sb.get_hops() != null && !sb.get_hops().isEmpty()) {
				LOG.error(sb.printBlockErrorLocation() + "FunctionStatementBlock should not have hops");
				throw new HopsException(sb.printBlockErrorLocation() + "FunctionStatementBlock should not have hops");
			}
			// step through stmt blocks in while stmt body
			for (StatementBlock stmtBlock : body){
				constructLops(stmtBlock);
			}
		}
		
		// handle default case for regular StatementBlock
		else {
			
			if (sb.get_hops() == null)
				sb.set_hops(new ArrayList<Hop>());
			
			ArrayList<Lop> lops = new ArrayList<Lop>();
			for (Hop hop : sb.get_hops()) {
				lops.add(hop.constructLops());
			}
			sb.setLops(lops);
			sb.updateRecompilationFlag(); 
		}
		
	} // end method
	
		
	public void printLops(DMLProgram dmlp) throws ParseException, LanguageException, HopsException, LopsException {
		if (LOG.isDebugEnabled()){
			// for each namespace, handle function program blocks
			for (String namespaceKey : dmlp.getNamespaces().keySet()){
				for (String fname : dmlp.getFunctionStatementBlocks(namespaceKey).keySet()){
					FunctionStatementBlock fsblock = dmlp.getFunctionStatementBlock(namespaceKey,fname);
					printLops(fsblock);
				}
			}

			for (int i = 0; i < dmlp.getNumStatementBlocks(); i++) {		
				StatementBlock current = dmlp.getStatementBlock(i);
				printLops(current);
			}
		}
	}
			
	public void printLops(StatementBlock current) throws ParseException, HopsException, LopsException {
		if (LOG.isDebugEnabled()){
			ArrayList<Lop> lopsDAG = current.getLops();

			LOG.debug("\n********************** LOPS DAG FOR BLOCK *******************");

			if (current instanceof FunctionStatementBlock) {
				if (current.getNumStatements() > 1)
					LOG.debug("Function statement block has more than 1 stmt");
				FunctionStatement fstmt = (FunctionStatement)current.getStatement(0);
				for (StatementBlock child : fstmt.getBody()){
					printLops(child);
				}
			}

			if (current instanceof WhileStatementBlock) {

				// print predicate lops 
				WhileStatementBlock wstb = (WhileStatementBlock) current; 
				Hop predicateHops = ((WhileStatementBlock) current).getPredicateHops();
				LOG.debug("\n********************** PREDICATE LOPS *******************");
				Lop predicateLops = predicateHops.getLops();
				if (predicateLops == null)
					predicateLops = predicateHops.constructLops();
				predicateLops.printMe();

				if (wstb.getNumStatements() > 1){
					LOG.error(wstb.printBlockErrorLocation() + "WhileStatementBlock has more than 1 statement");
					throw new HopsException(wstb.printBlockErrorLocation() + "WhileStatementBlock has more than 1 statement");
				}
				WhileStatement ws = (WhileStatement)wstb.getStatement(0);

				for (StatementBlock sb : ws.getBody()){
					printLops(sb);
				}
			}

			if (current instanceof IfStatementBlock) {

				// print predicate lops 
				IfStatementBlock istb = (IfStatementBlock) current; 
				Hop predicateHops = ((IfStatementBlock) current).getPredicateHops();
				LOG.debug("\n********************** PREDICATE LOPS *******************");
				Lop predicateLops = predicateHops.getLops();
				if (predicateLops == null)
					predicateLops = predicateHops.constructLops();
				predicateLops.printMe();

				if (istb.getNumStatements() > 1){
					LOG.error(istb.printBlockErrorLocation() + "IfStatmentBlock has more than 1 statement");
					throw new HopsException(istb.printBlockErrorLocation() + "IfStatmentBlock has more than 1 statement");
				}
				IfStatement is = (IfStatement)istb.getStatement(0);

				LOG.debug("\n**** LOPS DAG FOR IF BODY ****");
				for (StatementBlock sb : is.getIfBody()){
					printLops(sb);
				}
				if ( !is.getElseBody().isEmpty() ){
					LOG.debug("\n**** LOPS DAG FOR IF BODY ****");
					for (StatementBlock sb : is.getElseBody()){
						printLops(sb);
					}
				}
			}

			if (current instanceof ForStatementBlock) {

				// print predicate lops 
				ForStatementBlock fsb = (ForStatementBlock) current; 
				LOG.debug("\n********************** PREDICATE LOPS *******************");
				if( fsb.getFromHops() != null ){
					LOG.debug("FROM:");
					Lop llops = fsb.getFromLops();
					if( llops == null )
						llops = fsb.getFromHops().constructLops();
					llops.printMe();
				}
				if( fsb.getToHops() != null ){
					LOG.debug("TO:");
					Lop llops = fsb.getToLops();
					if( llops == null )
						llops = fsb.getToHops().constructLops();
					llops.printMe();
				}
				if( fsb.getIncrementHops() != null ){
					LOG.debug("INCREMENT:");
					Lop llops = fsb.getIncrementLops();
					if( llops == null )
						llops = fsb.getIncrementHops().constructLops();
					llops.printMe();
				}

				if (fsb.getNumStatements() > 1){
					LOG.error(fsb.printBlockErrorLocation() + "ForStatementBlock has more than 1 statement");
					throw new HopsException(fsb.printBlockErrorLocation() + "ForStatementBlock has more than 1 statement");
				}
				ForStatement ws = (ForStatement)fsb.getStatement(0);

				for (StatementBlock sb : ws.getBody()){
					printLops(sb);
				}
			}

			if (lopsDAG != null && !lopsDAG.isEmpty() ) {
				Iterator<Lop> iter = lopsDAG.iterator();
				while (iter.hasNext()) {
					LOG.debug("\n********************** OUTPUT LOPS *******************");
					iter.next().printMe();
				}
			}
		}
	}
	

	public void printHops(DMLProgram dmlp) throws ParseException, LanguageException, HopsException {
		if (LOG.isDebugEnabled()) {
			// for each namespace, handle function program blocks
			for (String namespaceKey : dmlp.getNamespaces().keySet()){
				for (String fname : dmlp.getFunctionStatementBlocks(namespaceKey).keySet()){
					FunctionStatementBlock fsblock = dmlp.getFunctionStatementBlock(namespaceKey,fname);
					printHops(fsblock);
				}
			}

			// hand
			for (int i = 0; i < dmlp.getNumStatementBlocks(); i++) {
				StatementBlock current = dmlp.getStatementBlock(i);
				printHops(current);
			}
		}
	}
	
	public void printHops(StatementBlock current) throws ParseException, HopsException {
		if (LOG.isDebugEnabled()) {
			ArrayList<Hop> hopsDAG = current.get_hops();
			LOG.debug("\n********************** HOPS DAG FOR BLOCK *******************");

			if (current instanceof FunctionStatementBlock) {
				if (current.getNumStatements() > 1)
					LOG.debug("Function statement block has more than 1 stmt");
				FunctionStatement fstmt = (FunctionStatement)current.getStatement(0);
				for (StatementBlock child : fstmt.getBody()){
					printHops(child);
				}
			}

			if (current instanceof WhileStatementBlock) {

				// print predicate hops
				WhileStatementBlock wstb = (WhileStatementBlock) current; 
				Hop predicateHops = wstb.getPredicateHops();
				LOG.debug("\n********************** PREDICATE HOPS *******************");
				predicateHops.printMe();
				
				if (wstb.getNumStatements() > 1)
					LOG.debug("While statement block has more than 1 stmt");
				WhileStatement ws = (WhileStatement)wstb.getStatement(0);

				for (StatementBlock sb : ws.getBody()){
					printHops(sb);
				}
			}

			if (current instanceof IfStatementBlock) {

				// print predicate hops
				IfStatementBlock istb = (IfStatementBlock) current; 
				Hop predicateHops = istb.getPredicateHops();
				LOG.debug("\n********************** PREDICATE HOPS *******************");
				predicateHops.printMe();
				

				if (istb.getNumStatements() > 1)
					LOG.debug("If statement block has more than 1 stmt");
				IfStatement is = (IfStatement)istb.getStatement(0);

				for (StatementBlock sb : is.getIfBody()){
					printHops(sb);
				}

				for (StatementBlock sb : is.getElseBody()){
					printHops(sb);
				}
			}


			if (current instanceof ForStatementBlock) {

				// print predicate hops
				ForStatementBlock fsb = (ForStatementBlock) current; 
				LOG.debug("\n********************** PREDICATE HOPS *******************");
				if (fsb.getFromHops() != null) fsb.getFromHops().printMe();
				if (fsb.getToHops() != null) fsb.getToHops().printMe();
				if (fsb.getIncrementHops() != null) fsb.getIncrementHops().printMe();
				
				if (fsb.getNumStatements() > 1)
					LOG.debug("For statement block has more than 1 stmt");
				ForStatement ws = (ForStatement)fsb.getStatement(0);

				for (StatementBlock sb : ws.getBody()){
					printHops(sb);
				}
			}

			if (hopsDAG != null && !hopsDAG.isEmpty()) {
				// hopsDAG.iterator().next().printMe();
				Iterator<Hop> iter = hopsDAG.iterator();
				while (iter.hasNext()) {
					LOG.debug("\n********************** OUTPUT HOPS *******************");
					iter.next().printMe();
				}
			}
		}
	}

	public void refreshMemEstimates(DMLProgram dmlp) throws ParseException, LanguageException, HopsException {

		// for each namespace, handle function program blocks -- forward direction
		for (String namespaceKey : dmlp.getNamespaces().keySet()){
			for (String fname : dmlp.getFunctionStatementBlocks(namespaceKey).keySet()){
				FunctionStatementBlock fsblock = dmlp.getFunctionStatementBlock(namespaceKey, fname);
				refreshMemEstimates(fsblock);
			}
		}
		
		// handle statement blocks in "main" method
		for (int i = 0; i < dmlp.getNumStatementBlocks(); i++) {
			StatementBlock current = dmlp.getStatementBlock(i);
			refreshMemEstimates(current);
		}
	}
			
	public void refreshMemEstimates(StatementBlock current) throws ParseException, HopsException {
	
		MemoTable memo = new MemoTable();
		ArrayList<Hop> hopsDAG = current.get_hops();
		if (hopsDAG != null && !hopsDAG.isEmpty()) {
			Iterator<Hop> iter = hopsDAG.iterator();
			while (iter.hasNext()) {
				iter.next().refreshMemEstimates(memo);
			}
		}
		
		if (current instanceof FunctionStatementBlock) {
			
			FunctionStatement fstmt = (FunctionStatement)current.getStatement(0);
			for (StatementBlock sb : fstmt.getBody()){
				refreshMemEstimates(sb);
			}
		}
		
		if (current instanceof WhileStatementBlock) {
			// handle predicate
			WhileStatementBlock wstb = (WhileStatementBlock) current;
			wstb.getPredicateHops().refreshMemEstimates(new MemoTable());
		
			if (wstb.getNumStatements() > 1)
				LOG.debug("While statement block has more than 1 stmt");
			WhileStatement ws = (WhileStatement)wstb.getStatement(0);
			
			for (StatementBlock sb : ws.getBody()){
				refreshMemEstimates(sb);
			}
		}
		
		if (current instanceof IfStatementBlock) {
			// handle predicate
			IfStatementBlock istb = (IfStatementBlock) current;
			istb.getPredicateHops().refreshMemEstimates(new MemoTable());
		
			if (istb.getNumStatements() > 1)
				LOG.debug("If statement block has more than 1 stmt");
			IfStatement is = (IfStatement)istb.getStatement(0);
			
			for (StatementBlock sb : is.getIfBody()){
				refreshMemEstimates(sb);
			}
			for (StatementBlock sb : is.getElseBody()){
				refreshMemEstimates(sb);
			}
		}
		
		if (current instanceof ForStatementBlock) {
			// handle predicate
			ForStatementBlock fsb = (ForStatementBlock) current;
			if (fsb.getFromHops() != null) 
				fsb.getFromHops().refreshMemEstimates(new MemoTable());
			if (fsb.getToHops() != null) 
				fsb.getToHops().refreshMemEstimates(new MemoTable());
			if (fsb.getIncrementHops() != null) 
				fsb.getIncrementHops().refreshMemEstimates(new MemoTable());
		
			if (fsb.getNumStatements() > 1)
				LOG.debug("For statement block has more than 1 stmt");
			ForStatement ws = (ForStatement)fsb.getStatement(0);
			
			for (StatementBlock sb : ws.getBody()){
				refreshMemEstimates(sb);
			}
		}
	}
	
	public static void resetHopsDAGVisitStatus(DMLProgram dmlp) throws ParseException, LanguageException, HopsException {

		// for each namespace, handle function program blocks -- forward direction
		for (String namespaceKey : dmlp.getNamespaces().keySet()){
			for (String fname : dmlp.getFunctionStatementBlocks(namespaceKey).keySet()){
				FunctionStatementBlock fsblock = dmlp.getFunctionStatementBlock(namespaceKey, fname);
				resetHopsDAGVisitStatus(fsblock);
			}
		}
		
		// handle statement blocks in "main" method
		for (int i = 0; i < dmlp.getNumStatementBlocks(); i++) {
			StatementBlock current = dmlp.getStatementBlock(i);
			resetHopsDAGVisitStatus(current);
		}
	}
			
	public static void resetHopsDAGVisitStatus(StatementBlock current) throws ParseException, HopsException {
	
		ArrayList<Hop> hopsDAG = current.get_hops();
		if (hopsDAG != null && !hopsDAG.isEmpty() ) {
			Hop.resetVisitStatus(hopsDAG);
		}
		
		if (current instanceof FunctionStatementBlock) {
			
			FunctionStatement fstmt = (FunctionStatement)current.getStatement(0);
			for (StatementBlock sb : fstmt.getBody()){
				resetHopsDAGVisitStatus(sb);
			}
		}
		
		if (current instanceof WhileStatementBlock) {
			// handle predicate
			WhileStatementBlock wstb = (WhileStatementBlock) current;
			wstb.getPredicateHops().resetVisitStatus();
		
			if (wstb.getNumStatements() > 1)
				LOG.debug("While stmt block has more than 1 stmt");
			WhileStatement ws = (WhileStatement)wstb.getStatement(0);
			
			for (StatementBlock sb : ws.getBody()){
				resetHopsDAGVisitStatus(sb);
			}
		}
		
		if (current instanceof IfStatementBlock) {
			// handle predicate
			IfStatementBlock istb = (IfStatementBlock) current;
			istb.getPredicateHops().resetVisitStatus();
		
			if (istb.getNumStatements() > 1)
				LOG.debug("If statement block has more than 1 stmt");
			IfStatement is = (IfStatement)istb.getStatement(0);
			
			for (StatementBlock sb : is.getIfBody()){
				resetHopsDAGVisitStatus(sb);
			}
			for (StatementBlock sb : is.getElseBody()){
				resetHopsDAGVisitStatus(sb);
			}
		}
		
		if (current instanceof ForStatementBlock) {
			// handle predicate
			ForStatementBlock fsb = (ForStatementBlock) current;
			if (fsb.getFromHops() != null) 
				fsb.getFromHops().resetVisitStatus();
			if (fsb.getToHops() != null) 
				fsb.getToHops().resetVisitStatus();
			if (fsb.getIncrementHops() != null) 
				fsb.getIncrementHops().resetVisitStatus();
		
			if (fsb.getNumStatements() > 1)
				LOG.debug("For statment block has more than 1 stmt");
			ForStatement ws = (ForStatement)fsb.getStatement(0);
			
			for (StatementBlock sb : ws.getBody()){
				resetHopsDAGVisitStatus(sb);
			}
		}
	}
	
	public void resetLopsDAGVisitStatus(DMLProgram dmlp) throws HopsException, LanguageException {
		
		// for each namespace, handle function program blocks
		for (String namespaceKey : dmlp.getNamespaces().keySet()){
			for (String fname : dmlp.getFunctionStatementBlocks(namespaceKey).keySet()){
				FunctionStatementBlock fsblock = dmlp.getFunctionStatementBlock(namespaceKey, fname);
				resetLopsDAGVisitStatus(fsblock);
			}
		}
		
		for (int i = 0; i < dmlp.getNumStatementBlocks(); i++) {
			StatementBlock current = dmlp.getStatementBlock(i);
			resetLopsDAGVisitStatus(current);
		}
	}
	
	public void resetLopsDAGVisitStatus(StatementBlock current) throws HopsException {
		
		ArrayList<Hop> hopsDAG = current.get_hops();

		if (hopsDAG != null && !hopsDAG.isEmpty() ) {
			Iterator<Hop> iter = hopsDAG.iterator();
			while (iter.hasNext()){
				Hop currentHop = iter.next();
				currentHop.getLops().resetVisitStatus();
			}
		}
		
		if (current instanceof FunctionStatementBlock) {
			FunctionStatementBlock fsb = (FunctionStatementBlock) current;
			FunctionStatement fs = (FunctionStatement)fsb.getStatement(0);
			
			for (StatementBlock sb : fs.getBody()){
				resetLopsDAGVisitStatus(sb);
			}
		}
		
		
		if (current instanceof WhileStatementBlock) {
			WhileStatementBlock wstb = (WhileStatementBlock) current;
			wstb.get_predicateLops().resetVisitStatus();
			if (wstb.getNumStatements() > 1)
				LOG.debug("While statement block has more than 1 stmt");
			WhileStatement ws = (WhileStatement)wstb.getStatement(0);
			
			for (StatementBlock sb : ws.getBody()){
				resetLopsDAGVisitStatus(sb);
			}
		}
		
		if (current instanceof IfStatementBlock) {
			IfStatementBlock istb = (IfStatementBlock) current;
			istb.get_predicateLops().resetVisitStatus();
			if (istb.getNumStatements() > 1)
				LOG.debug("If statement block has more than 1 stmt");
			IfStatement is = (IfStatement)istb.getStatement(0);
			
			for (StatementBlock sb : is.getIfBody()){
				resetLopsDAGVisitStatus(sb);
			}
			
			for (StatementBlock sb : is.getElseBody()){
				resetLopsDAGVisitStatus(sb);
			}
		}
		
		if (current instanceof ForStatementBlock) {
			ForStatementBlock fsb = (ForStatementBlock) current;
			
			if (fsb.getFromLops() != null) 
				fsb.getFromLops().resetVisitStatus();
			if (fsb.getToLops() != null) 
				fsb.getToLops().resetVisitStatus();
			if (fsb.getIncrementLops() != null) 
				fsb.getIncrementLops().resetVisitStatus();
			
			if (fsb.getNumStatements() > 1)
				LOG.debug("For statement block has more than 1 stmt");
			ForStatement ws = (ForStatement)fsb.getStatement(0);
			
			for (StatementBlock sb : ws.getBody()){
				resetLopsDAGVisitStatus(sb);
			}
		}
	}


	public void constructHops(StatementBlock sb) 
		throws ParseException, LanguageException {

		if (sb instanceof WhileStatementBlock) {
			constructHopsForWhileControlBlock((WhileStatementBlock) sb);
			return;
		}

		if (sb instanceof IfStatementBlock) {
			constructHopsForIfControlBlock((IfStatementBlock) sb);
			return;
		}
		
		if (sb instanceof ForStatementBlock) { //NOTE: applies to ForStatementBlock and ParForStatementBlock
			constructHopsForForControlBlock((ForStatementBlock) sb);
			return;
		}
		
		if (sb instanceof FunctionStatementBlock) {
			constructHopsForFunctionControlBlock((FunctionStatementBlock) sb);
			return;
		}
		
		
		HashMap<String, Hop> ids = new HashMap<String, Hop>();
		ArrayList<Hop> output = new ArrayList<Hop>();

		VariableSet liveIn 	= sb.liveIn();
		VariableSet liveOut = sb.liveOut();
		VariableSet	updated = sb._updated;
		VariableSet gen 	= sb._gen;
		VariableSet updatedLiveOut = new VariableSet();

		// handle liveout variables that are updated --> target identifiers for Assignment
		HashMap<String, Integer> liveOutToTemp = new HashMap<String, Integer>();
		for (int i = 0; i < sb.getNumStatements(); i++) {
			Statement current = sb.getStatement(i);
			
			if (current instanceof AssignmentStatement) {
				AssignmentStatement as = (AssignmentStatement) current;
				DataIdentifier target = as.getTarget();
				if (liveOut.containsVariable(target.getName())) {
					liveOutToTemp.put(target.getName(), Integer.valueOf(i));
				}
			}
			if (current instanceof MultiAssignmentStatement) {
				MultiAssignmentStatement mas = (MultiAssignmentStatement) current;
				
				for (DataIdentifier target : mas.getTargetList()){
					if (liveOut.containsVariable(target.getName())) {
						liveOutToTemp.put(target.getName(), Integer.valueOf(i));
					}
				}
			}
		}

		// only create transient read operations for variables either updated or read-before-update 
		//	(i.e., from LV analysis, updated and gen sets)
		if ( !liveIn.getVariables().values().isEmpty() ) {
			
			for (String varName : liveIn.getVariables().keySet()) {

				if (updated.containsVariable(varName) || gen.containsVariable(varName)){
				
					DataIdentifier var = liveIn.getVariables().get(varName);
					long actualDim1 = (var instanceof IndexedIdentifier) ? ((IndexedIdentifier)var).getOrigDim1() : var.getDim1();
					long actualDim2 = (var instanceof IndexedIdentifier) ? ((IndexedIdentifier)var).getOrigDim2() : var.getDim2();
					DataOp read = new DataOp(var.getName(), var.getDataType(), var.getValueType(), DataOpTypes.TRANSIENTREAD, null, actualDim1, actualDim2, var.getNnz(), var.getRowsInBlock(), var.getColumnsInBlock());
					read.setAllPositions(var.getBeginLine(), var.getBeginColumn(), var.getEndLine(), var.getEndColumn());
					ids.put(varName, read);
				}
			}
		}

	
		for( int i = 0; i < sb.getNumStatements(); i++ ) {
			Statement current = sb.getStatement(i);

			if (current instanceof OutputStatement) {
				OutputStatement os = (OutputStatement) current;

				DataExpression source = os.getSource();
				DataIdentifier target = os.getIdentifier();

				//error handling unsupported indexing expression in write statement
				if( target instanceof IndexedIdentifier ) {
					throw new LanguageException(source.printErrorLocation()+": Unsupported indexing expression in write statement. " +
							                    "Please, assign the right indexing result to a variable and write this variable.");
				}
				
				DataOp ae = (DataOp)processExpression(source, target, ids);
				String formatName = os.getExprParam(DataExpression.FORMAT_TYPE).toString();
				ae.setInputFormatType(Expression.convertFormatType(formatName));

				if (ae.getDataType() == DataType.SCALAR ) {
					ae.setOutputParams(ae.getDim1(), ae.getDim2(), ae.getNnz(), ae.getUpdateInPlace(), -1, -1);
				}
				else {
					switch(ae.getInputFormatType()) {
					case TEXT:
					case MM:
					case CSV:
						// write output in textcell format
						ae.setOutputParams(ae.getDim1(), ae.getDim2(), ae.getNnz(), ae.getUpdateInPlace(), -1, -1);
						break;
						
					case BINARY:
						// write output in binary block format
					    ae.setOutputParams(ae.getDim1(), ae.getDim2(), ae.getNnz(), ae.getUpdateInPlace(), ConfigurationManager.getBlocksize(), ConfigurationManager.getBlocksize());
					    break;
						
						default:
							throw new LanguageException("Unrecognized file format: " + ae.getInputFormatType());
					}
				}
				
				output.add(ae);
				
			}

			if (current instanceof PrintStatement) {
				PrintStatement ps = (PrintStatement) current;
				Expression source = ps.getExpression();
				PRINTTYPE ptype = ps.getType();
				
				DataIdentifier target = createTarget();
				target.setDataType(DataType.SCALAR);
				target.setValueType(ValueType.STRING);
				target.setAllPositions(current.getFilename(), current.getBeginLine(), target.getBeginColumn(), current.getEndLine(),  current.getEndColumn());
				
				Hop ae = processExpression(source, target, ids);
			
				try {
					Hop.OpOp1 op = (ptype == PRINTTYPE.PRINT ? Hop.OpOp1.PRINT : Hop.OpOp1.STOP);
					Hop printHop = new UnaryOp(target.getName(), target.getDataType(), target.getValueType(), op, ae);
					printHop.setAllPositions(current.getBeginLine(), current.getBeginColumn(), current.getEndLine(), current.getEndColumn());
					output.add(printHop);
				} catch ( HopsException e ) {
					throw new LanguageException(e);
				}
			}
	
			if (current instanceof AssignmentStatement) {
	
				AssignmentStatement as = (AssignmentStatement) current;
				DataIdentifier target = as.getTarget();
				Expression source = as.getSource();

			
				// CASE: regular assignment statement -- source is DML expression that is NOT user-defined or external function 
				if (!(source instanceof FunctionCallIdentifier)){
				
					// CASE: target is regular data identifier
					if (!(target instanceof IndexedIdentifier)) {
					
						Hop ae = processExpression(source, target, ids);
						ids.put(target.getName(), ae);
						target.setProperties(source.getOutput());
						Integer statementId = liveOutToTemp.get(target.getName());
						if ((statementId != null) && (statementId.intValue() == i)) {
							DataOp transientwrite = new DataOp(target.getName(), target.getDataType(), target.getValueType(), ae, DataOpTypes.TRANSIENTWRITE, null);
							transientwrite.setOutputParams(ae.getDim1(), ae.getDim2(), ae.getNnz(), ae.getUpdateInPlace(), ae.getRowsInBlock(), ae.getColsInBlock());
							transientwrite.setAllPositions(target.getBeginLine(), target.getBeginColumn(), target.getEndLine(), target.getEndLine());
							updatedLiveOut.addVariable(target.getName(), target);
							output.add(transientwrite);
						}
					} // end if (!(target instanceof IndexedIdentifier)) {
					
					// CASE: target is indexed identifier (left-hand side indexed expression)
					else {
						Hop ae = processLeftIndexedExpression(source, (IndexedIdentifier)target, ids);
						
						ids.put(target.getName(), ae);
						
						// obtain origDim values BEFORE they are potentially updated during setProperties call
						//	(this is incorrect for LHS Indexing)
						long origDim1 = ((IndexedIdentifier)target).getOrigDim1();
						long origDim2 = ((IndexedIdentifier)target).getOrigDim2();						 
						target.setProperties(source.getOutput());
						((IndexedIdentifier)target).setOriginalDimensions(origDim1, origDim2);
						
						// preserve data type matrix of any index identifier
						// (required for scalar input to left indexing)					
						if( target.getDataType() != DataType.MATRIX ) {
							target.setDataType(DataType.MATRIX);
							target.setValueType(ValueType.DOUBLE);
							target.setBlockDimensions(ConfigurationManager.getBlocksize(), ConfigurationManager.getBlocksize());
						}
						
						Integer statementId = liveOutToTemp.get(target.getName());
						if ((statementId != null) && (statementId.intValue() == i)) {
							DataOp transientwrite = new DataOp(target.getName(), target.getDataType(), target.getValueType(), ae, DataOpTypes.TRANSIENTWRITE, null);
							transientwrite.setOutputParams(origDim1, origDim2, ae.getNnz(), ae.getUpdateInPlace(), ae.getRowsInBlock(), ae.getColsInBlock());
							transientwrite.setAllPositions(target.getBeginLine(), target.getBeginColumn(), target.getEndLine(), target.getEndColumn());
							updatedLiveOut.addVariable(target.getName(), target);
							output.add(transientwrite);
						}
					}
					
					
				}
				else 
				{
					//assignment, function call
					FunctionCallIdentifier fci = (FunctionCallIdentifier) source;
					FunctionStatementBlock fsb = this._dmlProg.getFunctionStatementBlock(fci.getNamespace(),fci.getName());
					
					//error handling missing function
					if (fsb == null){
						String error = source.printErrorLocation() + "function " + fci.getName() + " is undefined in namespace " + fci.getNamespace(); 
						LOG.error(error);
						throw new LanguageException(error);
					}
					
					//error handling unsupported function call in indexing expression
					if( target instanceof IndexedIdentifier ) {
						String fkey = DMLProgram.constructFunctionKey(fci.getNamespace(),fci.getName());
						throw new LanguageException("Unsupported function call to '"+fkey+"' in left indexing expression. " +
								                    "Please, assign the function output to a variable.");
					}
					
					ArrayList<Hop> finputs = new ArrayList<Hop>();
					for (ParameterExpression paramName : fci.getParamExprs()){
						Hop in = processExpression(paramName.getExpr(), null, ids);						
						finputs.add(in);
					}

					//create function op
					FunctionType ftype = fsb.getFunctionOpType();
					FunctionOp fcall = new FunctionOp(ftype, fci.getNamespace(), fci.getName(), finputs, new String[]{target.getName()});
					output.add(fcall);
					
					//TODO function output dataops (phase 3)
					//DataOp trFoutput = new DataOp(target.getName(), target.getDataType(), target.getValueType(), fcall, DataOpTypes.FUNCTIONOUTPUT, null);
					//DataOp twFoutput = new DataOp(target.getName(), target.getDataType(), target.getValueType(), trFoutput, DataOpTypes.TRANSIENTWRITE, null);					
				}
			}

			else if (current instanceof MultiAssignmentStatement) {
				//multi-assignment, by definition a function call
				MultiAssignmentStatement mas = (MultiAssignmentStatement) current;
				Expression source = mas.getSource();
				
				if ( source instanceof FunctionCallIdentifier ) {
					FunctionCallIdentifier fci = (FunctionCallIdentifier) source;
					FunctionStatementBlock fsb = this._dmlProg.getFunctionStatementBlock(fci.getNamespace(),fci.getName());
					FunctionStatement fstmt = (FunctionStatement)fsb.getStatement(0);
					if (fstmt == null){
						LOG.error(source.printErrorLocation() + "function " + fci.getName() + " is undefined in namespace " + fci.getNamespace());
						throw new LanguageException(source.printErrorLocation() + "function " + fci.getName() + " is undefined in namespace " + fci.getNamespace());
					}
					
					ArrayList<Hop> finputs = new ArrayList<Hop>();
					for (ParameterExpression paramName : fci.getParamExprs()){
						Hop in = processExpression(paramName.getExpr(), null, ids);						
						finputs.add(in);
					}

					//create function op
					String[] foutputs = new String[mas.getTargetList().size()]; 
					int count = 0;
					for ( DataIdentifier paramName : mas.getTargetList() ){
						foutputs[count++]=paramName.getName();
					}
					
					FunctionType ftype = fsb.getFunctionOpType();
					FunctionOp fcall = new FunctionOp(ftype, fci.getNamespace(), fci.getName(), finputs, foutputs);
					output.add(fcall);
					
					//TODO function output dataops (phase 3)
					/*for ( DataIdentifier paramName : mas.getTargetList() ){
						DataOp twFoutput = new DataOp(paramName.getName(), paramName.getDataType(), paramName.getValueType(), fcall, DataOpTypes.TRANSIENTWRITE, null);
						output.add(twFoutput);
					}*/
				}
				else if ( source instanceof BuiltinFunctionExpression && ((BuiltinFunctionExpression)source).multipleReturns() ) {
					// construct input hops
					Hop fcall = processMultipleReturnBuiltinFunctionExpression((BuiltinFunctionExpression)source, mas.getTargetList(), ids);
					output.add(fcall);					
				}
				else if ( source instanceof ParameterizedBuiltinFunctionExpression && ((ParameterizedBuiltinFunctionExpression)source).multipleReturns() ) {
					// construct input hops
					Hop fcall = processMultipleReturnParameterizedBuiltinFunctionExpression((ParameterizedBuiltinFunctionExpression)source, mas.getTargetList(), ids);
					output.add(fcall);					
				}
				else
					throw new LanguageException("Class \"" + source.getClass() + "\" is not supported in Multiple Assignment statements");
			}
			
		}
		sb.updateLiveVariablesOut(updatedLiveOut);
		sb.set_hops(output);

	}
	
	public void constructHopsForIfControlBlock(IfStatementBlock sb) throws ParseException, LanguageException {
		
		IfStatement ifsb = (IfStatement) sb.getStatement(0);
		ArrayList<StatementBlock> ifBody = ifsb.getIfBody();
		ArrayList<StatementBlock> elseBody = ifsb.getElseBody();
	
		// construct hops for predicate in if statement
		constructHopsForConditionalPredicate(sb);
		
		// handle if statement body
		for( StatementBlock current : ifBody ) {
			constructHops(current);
		}
		
		// handle else stmt body
		for( StatementBlock current : elseBody ) {
			constructHops(current);
		}
	}
	
	/**
	 * Constructs Hops for a given ForStatementBlock or ParForStatementBlock, respectively.
	 * 
	 * @param sb
	 * @throws ParseException
	 * @throws LanguageException
	 */
	public void constructHopsForForControlBlock(ForStatementBlock sb) 
		throws ParseException, LanguageException 
	{
		
		ForStatement fs = (ForStatement) sb.getStatement(0);
		ArrayList<StatementBlock> body = fs.getBody();
			
		// construct hops for iterable predicate
		constructHopsForIterablePredicate(sb);
			
		for( StatementBlock current : body ) {
			constructHops(current);
		}
	}
	
	public void constructHopsForFunctionControlBlock(FunctionStatementBlock fsb) throws ParseException, LanguageException {

		ArrayList<StatementBlock> body = ((FunctionStatement)fsb.getStatement(0)).getBody();

		for( StatementBlock current : body ) {
			constructHops(current);
		}
	}
	
	public void constructHopsForWhileControlBlock(WhileStatementBlock sb) 
			throws ParseException, LanguageException {
		
		ArrayList<StatementBlock> body = ((WhileStatement)sb.getStatement(0)).getBody();
		
		// construct hops for while predicate
		constructHopsForConditionalPredicate(sb);
			
		for( StatementBlock current : body ) {
			constructHops(current);
		}
	}
	
	
	public void constructHopsForConditionalPredicate(StatementBlock passedSB) throws ParseException {

		HashMap<String, Hop> _ids = new HashMap<String, Hop>();
		
		// set conditional predicate
		ConditionalPredicate cp = null;
		
		if (passedSB instanceof WhileStatementBlock){
			WhileStatement ws = (WhileStatement) ((WhileStatementBlock)passedSB).getStatement(0);
			cp = ws.getConditionalPredicate();
		} 
		else if (passedSB instanceof IfStatementBlock) {
			IfStatement ws = (IfStatement) ((IfStatementBlock)passedSB).getStatement(0);
			cp = ws.getConditionalPredicate();
		}
		else {
			throw new ParseException("ConditionalPredicate expected only for while or if statements.");
		}
		
		VariableSet varsRead = cp.variablesRead();

		for (String varName : varsRead.getVariables().keySet()) {
			
			// creating transient read for live in variables
			DataIdentifier var = passedSB.liveIn().getVariables().get(varName);
			
			DataOp read = null;
			
			if (var == null) {
				LOG.error("variable " + varName + " not live variable for conditional predicate");
				throw new ParseException("variable " + varName + " not live variable for conditional predicate");
			} else {
				long actualDim1 = (var instanceof IndexedIdentifier) ? ((IndexedIdentifier)var).getOrigDim1() : var.getDim1();
				long actualDim2 = (var instanceof IndexedIdentifier) ? ((IndexedIdentifier)var).getOrigDim2() : var.getDim2();
				
				read = new DataOp(var.getName(), var.getDataType(), var.getValueType(), DataOpTypes.TRANSIENTREAD,
						null, actualDim1, actualDim2, var.getNnz(), var.getRowsInBlock(), var.getColumnsInBlock());
				read.setAllPositions(var.getBeginLine(), var.getBeginColumn(), var.getEndLine(), var.getEndColumn());
			}
			_ids.put(varName, read);
		}
		
		DataIdentifier target = new DataIdentifier(Expression.getTempName());
		target.setDataType(DataType.SCALAR);
		target.setValueType(ValueType.BOOLEAN);
		target.setAllPositions(passedSB.getFilename(), passedSB.getBeginLine(), passedSB.getBeginColumn(), passedSB.getEndLine(), passedSB.getEndColumn());
		Hop predicateHops = null;
		Expression predicate = cp.getPredicate();
		
		if (predicate instanceof RelationalExpression) {
			predicateHops = processRelationalExpression((RelationalExpression) cp.getPredicate(), target, _ids);
		} else if (predicate instanceof BooleanExpression) {
			predicateHops = processBooleanExpression((BooleanExpression) cp.getPredicate(), target, _ids);
		} else if (predicate instanceof DataIdentifier) {
			// handle data identifier predicate
			predicateHops = processExpression(cp.getPredicate(), null, _ids);
		} else if (predicate instanceof ConstIdentifier) {
			// handle constant identifier
			//  a) translate 0 --> FALSE; translate 1 --> TRUE
			//	b) disallow string values
			if ( (predicate instanceof IntIdentifier && ((IntIdentifier)predicate).getValue() == 0) || (predicate instanceof DoubleIdentifier && ((DoubleIdentifier)predicate).getValue() == 0.0)) {
				cp.setPredicate(new BooleanIdentifier(false, 
						predicate.getFilename(),
						predicate.getBeginLine(), predicate.getBeginColumn(),
						predicate.getEndLine(), predicate.getEndColumn()));
				
			}
			else if ( (predicate instanceof IntIdentifier && ((IntIdentifier)predicate).getValue() == 1) || (predicate instanceof DoubleIdentifier && ((DoubleIdentifier)predicate).getValue() == 1.0)) {
				cp.setPredicate(new BooleanIdentifier(true,
						predicate.getFilename(),
						predicate.getBeginLine(), predicate.getBeginColumn(),
						predicate.getEndLine(), predicate.getEndColumn()));
			}
			else if (predicate instanceof IntIdentifier || predicate instanceof DoubleIdentifier){
				cp.setPredicate(new BooleanIdentifier(true,
						predicate.getFilename(),
						predicate.getBeginLine(), predicate.getBeginColumn(),
						predicate.getEndLine(), predicate.getEndColumn()));
				LOG.warn(predicate.printWarningLocation() + "Numerical value '" + predicate.toString() + "' (!= 0/1) is converted to boolean TRUE by DML");
			}
			else if (predicate instanceof StringIdentifier) {
				LOG.error(predicate.printErrorLocation() + "String value '" + predicate.toString() + "' is not allowed for iterable predicate");
				throw new ParseException(predicate.printErrorLocation() + "String value '" + predicate.toString() + "' is not allowed for iterable predicate");

			}
			predicateHops = processExpression(cp.getPredicate(), null, _ids);
		}
		if (passedSB instanceof WhileStatementBlock)
			((WhileStatementBlock)passedSB).setPredicateHops(predicateHops);
		else if (passedSB instanceof IfStatementBlock)
			((IfStatementBlock)passedSB).setPredicateHops(predicateHops);
	}

	
	/**
	 * Constructs all predicate Hops (for FROM, TO, INCREMENT) of an iterable predicate
	 * and assigns these Hops to the passed statement block.
	 * 
	 * Method used for both ForStatementBlock and ParForStatementBlock.
	 * 
	 * @param passedSB
	 * @throws ParseException
	 */
	public void constructHopsForIterablePredicate(ForStatementBlock fsb) 
		throws ParseException 
	{
		HashMap<String, Hop> _ids = new HashMap<String, Hop>();
			
		// set iterable predicate 
		ForStatement fs = (ForStatement) fsb.getStatement(0);
		IterablePredicate ip = fs.getIterablePredicate();
	
		for(int i=0; i < 3; i++) {
			VariableSet varsRead = null;
			if (i==0)
				varsRead = ip.getFromExpr().variablesRead();
			else if (i==1)
				varsRead = ip.getToExpr().variablesRead();
			else if( ip.getIncrementExpr() != null )
				varsRead = ip.getIncrementExpr().variablesRead();

			if(varsRead != null) {
				for (String varName : varsRead.getVariables().keySet()) {
					
					DataIdentifier var = fsb.liveIn().getVariable(varName);
					DataOp read = null;
					if (var == null) {
						LOG.error("variable '" + varName + "' is not available for iterable predicate");
						throw new ParseException("variable '" + varName + "' is not available for iterable predicate");
					}
					else {
						long actualDim1 = (var instanceof IndexedIdentifier) ? ((IndexedIdentifier)var).getOrigDim1() : var.getDim1();
						long actualDim2 = (var instanceof IndexedIdentifier) ? ((IndexedIdentifier)var).getOrigDim2() : var.getDim2();
						read = new DataOp(var.getName(), var.getDataType(), var.getValueType(), DataOpTypes.TRANSIENTREAD,
								null, actualDim1, actualDim2,  var.getNnz(), var.getRowsInBlock(),  var.getColumnsInBlock());
						read.setAllPositions(var.getBeginLine(), var.getBeginColumn(), var.getEndLine(), var.getEndColumn());
					}
					_ids.put(varName, read);
				}
			}
			
			//construct hops for from, to, and increment expressions		
			if(i==0)
				fsb.setFromHops(      processTempIntExpression( ip.getFromExpr(),      _ids ));
			else if(i==1)
				fsb.setToHops(        processTempIntExpression( ip.getToExpr(),        _ids ));
			else if( ip.getIncrementExpr() != null )
				fsb.setIncrementHops( processTempIntExpression( ip.getIncrementExpr(), _ids ));					
				
		}
		
		/*VariableSet varsRead = ip.variablesRead();
		
		for (String varName : varsRead.getVariables().keySet()) {
			
			DataIdentifier var = passedSB.liveIn().getVariable(varName);
			DataOp read = null;
			if (var == null) {
				LOG.error(var.printErrorLocation() + "variable '" + varName + "' is not available for iterable predicate");
				throw new ParseException(var.printErrorLocation() + "variable '" + varName + "' is not available for iterable predicate");
			}
			else {
				long actualDim1 = (var instanceof IndexedIdentifier) ? ((IndexedIdentifier)var).getOrigDim1() : var.getDim1();
				long actualDim2 = (var instanceof IndexedIdentifier) ? ((IndexedIdentifier)var).getOrigDim2() : var.getDim2();
				read = new DataOp(var.getName(), var.getDataType(), var.getValueType(), DataOpTypes.TRANSIENTREAD,
						null, actualDim1, actualDim2,  var.getNnz(), var.getRowsInBlock(),  var.getColumnsInBlock());
				read.setAllPositions(var.getBeginLine(), var.getBeginColumn(), var.getEndLine(), var.getEndColumn());
			}
			_ids.put(varName, read);
		}

		//construct hops for from, to, and increment expressions		
		fsb.setFromHops(      processTempIntExpression( ip.getFromExpr(),      _ids ));
		fsb.setToHops(        processTempIntExpression( ip.getToExpr(),        _ids ));
		fsb.setIncrementHops( processTempIntExpression( ip.getIncrementExpr(), _ids ));*/					
	}
	 
	
	/**
	 * Construct Hops from parse tree : Process Expression in an assignment
	 * statement
	 * 
	 * @throws ParseException
	 */
	private Hop processExpression(Expression source, DataIdentifier target, HashMap<String, Hop> hops) throws ParseException {
		if (source.getKind() == Expression.Kind.BinaryOp) {
			return processBinaryExpression((BinaryExpression) source, target, hops);
		} else if (source.getKind() == Expression.Kind.RelationalOp) {
			return processRelationalExpression((RelationalExpression) source, target, hops);
		} else if (source.getKind() == Expression.Kind.BooleanOp) {
			return processBooleanExpression((BooleanExpression) source, target, hops);
		} else if (source.getKind() == Expression.Kind.Data) {
			if (source instanceof IndexedIdentifier){
				IndexedIdentifier sourceIndexed = (IndexedIdentifier) source;
				return processIndexingExpression(sourceIndexed,target,hops);
			} else if (source instanceof IntIdentifier) {
				IntIdentifier sourceInt = (IntIdentifier) source;
				LiteralOp litop = new LiteralOp(sourceInt.getValue());
				litop.setAllPositions(sourceInt.getBeginLine(), sourceInt.getBeginColumn(), sourceInt.getEndLine(), sourceInt.getEndColumn());
				setIdentifierParams(litop, sourceInt);
				return litop;
			} else if (source instanceof DoubleIdentifier) {
				DoubleIdentifier sourceDouble = (DoubleIdentifier) source;
				LiteralOp litop = new LiteralOp(sourceDouble.getValue());
				litop.setAllPositions(sourceDouble.getBeginLine(), sourceDouble.getBeginColumn(), sourceDouble.getEndLine(), sourceDouble.getEndColumn());
				setIdentifierParams(litop, sourceDouble);
				return litop;
			} else if (source instanceof DataIdentifier) {
				DataIdentifier sourceId = (DataIdentifier) source;
				return hops.get(sourceId.getName());
			} else if (source instanceof BooleanIdentifier) {
				BooleanIdentifier sourceBoolean = (BooleanIdentifier) source;
				LiteralOp litop = new LiteralOp(sourceBoolean.getValue());
				litop.setAllPositions(sourceBoolean.getBeginLine(), sourceBoolean.getBeginColumn(), sourceBoolean.getEndLine(), sourceBoolean.getEndColumn());
				setIdentifierParams(litop, sourceBoolean);
				return litop;
			} else if (source instanceof StringIdentifier) {
				StringIdentifier sourceString = (StringIdentifier) source;
				LiteralOp litop = new LiteralOp(sourceString.getValue());
				litop.setAllPositions(sourceString.getBeginLine(), sourceString.getBeginColumn(), sourceString.getEndLine(), sourceString.getEndColumn());
				setIdentifierParams(litop, sourceString);
				return litop;
			}
		} else if (source.getKind() == Expression.Kind.BuiltinFunctionOp) {
			try {
				return processBuiltinFunctionExpression((BuiltinFunctionExpression) source, target, hops);
			} catch (HopsException e) {
				throw new ParseException(e.getMessage());
			}
		} else if (source.getKind() == Expression.Kind.ParameterizedBuiltinFunctionOp ) {
			try {
				return processParameterizedBuiltinFunctionExpression((ParameterizedBuiltinFunctionExpression)source, target, hops);
			} catch ( HopsException e ) {
				throw new ParseException(e.getMessage());
			}
		} else if (source.getKind() == Expression.Kind.DataOp ) {
			try {	
				Hop ae = (Hop)processDataExpression((DataExpression)source, target, hops);
				
				if (ae instanceof DataOp){
					String formatName = ((DataExpression)source).getVarParam(DataExpression.FORMAT_TYPE).toString();
					((DataOp)ae).setInputFormatType(Expression.convertFormatType(formatName));
				}
				//hops.put(target.getName(), ae);
				return ae;
			} catch ( Exception e ) {
				throw new ParseException(e.getMessage());
			}
		}
		return null;
	} // end method processExpression

	private DataIdentifier createTarget(Expression source) {
		Identifier id = source.getOutput();
		if (id instanceof DataIdentifier && !(id instanceof DataExpression))
			return (DataIdentifier) id;
		DataIdentifier target = new DataIdentifier(Expression.getTempName());
		target.setProperties(id);
		return target;
	}

	private DataIdentifier createTarget() {
		DataIdentifier target = new DataIdentifier(Expression.getTempName());
		return target;
	}
	
	 
	/**
	 * Constructs the Hops for arbitrary expressions that eventually evaluate to an INT scalar. 
	 * 
	 * @param source 
	 * @param hops
	 * @return
	 * @throws ParseException
	 */
	private Hop processTempIntExpression( Expression source,  HashMap<String, Hop> hops ) 
		throws ParseException
	{
		DataIdentifier tmpOut = createTarget();		
		tmpOut.setDataType(DataType.SCALAR);
		tmpOut.setValueType(ValueType.INT);		
		source.setOutput(tmpOut);
		
		return processExpression(source, tmpOut, hops );	
	}
	
	private Hop processLeftIndexedExpression(Expression source, IndexedIdentifier target, HashMap<String, Hop> hops)  
			throws ParseException {

		// process target indexed expressions
		Hop rowLowerHops = null, rowUpperHops = null, colLowerHops = null, colUpperHops = null;
		
		if (target.getRowLowerBound() != null)
			rowLowerHops = processExpression(target.getRowLowerBound(),null,hops);
		else
			rowLowerHops = new LiteralOp(1);
		
		if (target.getRowUpperBound() != null)
			rowUpperHops = processExpression(target.getRowUpperBound(),null,hops);
		else
		{
			if ( target.getDim1() != -1 ) 
				rowUpperHops = new LiteralOp(target.getOrigDim1());
			else
			{
				try {
					//currBuiltinOp = new UnaryOp(target.getName(), target.getDataType(), target.getValueType(), Hops.OpOp1.NROW, expr);
					rowUpperHops = new UnaryOp(target.getName(), DataType.SCALAR, ValueType.INT, Hop.OpOp1.NROW, hops.get(target.getName()));
					rowUpperHops.setAllPositions(target.getBeginLine(), target.getBeginColumn(), target.getEndLine(), target.getEndColumn());
				} catch (HopsException e) {
					LOG.error(target.printErrorLocation() + "error processing row upper index for indexed expression " + target.toString());
					throw new RuntimeException(target.printErrorLocation() + "error processing row upper index for indexed expression " + target.toString());
				}
			}
		}
		if (target.getColLowerBound() != null)
			colLowerHops = processExpression(target.getColLowerBound(),null,hops);
		else
			colLowerHops = new LiteralOp(1);
		
		if (target.getColUpperBound() != null)
			colUpperHops = processExpression(target.getColUpperBound(),null,hops);
		else
		{
			if ( target.getDim2() != -1 ) 
				colUpperHops = new LiteralOp(target.getOrigDim2());
			else
			{
				try {
					colUpperHops = new UnaryOp(target.getName(), DataType.SCALAR, ValueType.INT, Hop.OpOp1.NCOL, hops.get(target.getName()));
				} catch (HopsException e) {
					LOG.error(target.printErrorLocation() + " error processing column upper index for indexed expression " + target.toString());
					throw new RuntimeException(target.printErrorLocation() + " error processing column upper index for indexed expression " + target.toString(), e);
				}
			}
		}
		
		//if (target == null) {
		//	target = createTarget(source);
		//}
		
		// process the source expression to get source Hops
		Hop sourceOp = processExpression(source, target, hops);
		
		// process the target to get targetHops
		Hop targetOp = hops.get(target.getName());
		if (targetOp == null){
			LOG.error(target.printErrorLocation() + " must define matrix " + target.getName() + " before indexing operations are allowed ");
			throw new ParseException(target.printErrorLocation() + " must define matrix " + target.getName() + " before indexing operations are allowed ");
		}
		
		//TODO Doug, please verify this (we need probably a cleaner way than this postprocessing)
		if( sourceOp.getDataType() == DataType.MATRIX && source.getOutput().getDataType() == DataType.SCALAR )
			sourceOp.setDataType(DataType.SCALAR);
		
		Hop leftIndexOp = new LeftIndexingOp(target.getName(), target.getDataType(), target.getValueType(), 
				targetOp, sourceOp, rowLowerHops, rowUpperHops, colLowerHops, colUpperHops, 
				target.getRowLowerEqualsUpper(), target.getColLowerEqualsUpper());
		
		setIdentifierParams(leftIndexOp, target);
	
		leftIndexOp.setAllPositions(target.getBeginLine(), target.getBeginColumn(), target.getEndLine(), target.getEndColumn());
		leftIndexOp.setDim1(target.getOrigDim1());
		leftIndexOp.setDim2(target.getOrigDim2());
	
		return leftIndexOp;
	}
	
	
	private Hop processIndexingExpression(IndexedIdentifier source, DataIdentifier target, HashMap<String, Hop> hops) 
		throws ParseException {
	
		// process Hops for indexes (for source)
		Hop rowLowerHops = null, rowUpperHops = null, colLowerHops = null, colUpperHops = null;
		
		if (source.getRowLowerBound() != null)
			rowLowerHops = processExpression(source.getRowLowerBound(),null,hops);
		else
			rowLowerHops = new LiteralOp(1);
		
		if (source.getRowUpperBound() != null)
			rowUpperHops = processExpression(source.getRowUpperBound(),null,hops);
		else
		{
			if ( source.getOrigDim1() != -1 ) 
				rowUpperHops = new LiteralOp(source.getOrigDim1());
			else
			{
				try {
					//currBuiltinOp = new UnaryOp(target.getName(), target.getDataType(), target.getValueType(), Hops.OpOp1.NROW, expr);
					rowUpperHops = new UnaryOp(source.getName(), DataType.SCALAR, ValueType.INT, Hop.OpOp1.NROW, hops.get(source.getName()));
					rowUpperHops.setAllPositions(source.getBeginLine(),source.getBeginColumn(), source.getEndLine(), source.getEndColumn());
				} catch (HopsException e) {
					LOG.error(source.printErrorLocation() + "error processing row upper index for indexed identifier " + source.toString());
					throw new RuntimeException(source.printErrorLocation() + "error processing row upper index for indexed identifier " + source.toString() + e);
				}
			}
		}
		if (source.getColLowerBound() != null)
			colLowerHops = processExpression(source.getColLowerBound(),null,hops);
		else
			colLowerHops = new LiteralOp(1);
		
		if (source.getColUpperBound() != null)
			colUpperHops = processExpression(source.getColUpperBound(),null,hops);
		else
		{
			if ( source.getOrigDim2() != -1 ) 
				colUpperHops = new LiteralOp(source.getOrigDim2());
			else
			{
				try {
					colUpperHops = new UnaryOp(source.getName(), DataType.SCALAR, ValueType.INT, Hop.OpOp1.NCOL, hops.get(source.getName()));
				} catch (HopsException e) {
					LOG.error(source.printErrorLocation() + "error processing column upper index for indexed indentifier " + source.toString(), e);
					throw new RuntimeException(source.printErrorLocation() + "error processing column upper index for indexed indentifier " + source.toString(), e);
				}
			}
		}
		
		if (target == null) {
			target = createTarget(source);
		}
		//unknown nnz after range indexing (applies to indexing op but also
		//data dependent operations)
		target.setNnz(-1); 
		
		Hop indexOp = new IndexingOp(target.getName(), target.getDataType(), target.getValueType(),
				hops.get(source.getName()), rowLowerHops, rowUpperHops, colLowerHops, colUpperHops,
				source.getRowLowerEqualsUpper(), source.getColLowerEqualsUpper());
	
		indexOp.setAllPositions(indexOp.getBeginLine(), indexOp.getBeginColumn(), indexOp.getEndLine(), indexOp.getEndColumn());
		setIdentifierParams(indexOp, target);
		
		return indexOp;
	}
	
	
	/**
	 * Construct Hops from parse tree : Process Binary Expression in an
	 * assignment statement
	 * 
	 * @throws ParseException
	 */
	private Hop processBinaryExpression(BinaryExpression source, DataIdentifier target, HashMap<String, Hop> hops)
		throws ParseException 
	{
		Hop left  = processExpression(source.getLeft(),  null, hops);
		Hop right = processExpression(source.getRight(), null, hops);

		if (left == null || right == null){
			left  = processExpression(source.getLeft(),  null, hops);
			right = processExpression(source.getRight(), null, hops);
		}
	
		Hop currBop = null;

		//prepare target identifier and ensure that output type is of inferred type 
        //(type should not be determined by target (e.g., string for print)
		if (target == null) {
		    target = createTarget(source);
		}
		target.setValueType(source.getOutput().getValueType());
		
		if (source.getOpCode() == Expression.BinaryOp.PLUS) {
			currBop = new BinaryOp(target.getName(), target.getDataType(), target.getValueType(), OpOp2.PLUS, left, right);
		} else if (source.getOpCode() == Expression.BinaryOp.MINUS) {
			currBop = new BinaryOp(target.getName(), target.getDataType(), target.getValueType(), OpOp2.MINUS, left, right);
		} else if (source.getOpCode() == Expression.BinaryOp.MULT) {
			currBop = new BinaryOp(target.getName(), target.getDataType(), target.getValueType(), OpOp2.MULT, left, right);
		} else if (source.getOpCode() == Expression.BinaryOp.DIV) {
			currBop = new BinaryOp(target.getName(), target.getDataType(), target.getValueType(), OpOp2.DIV, left, right);
		} else if (source.getOpCode() == Expression.BinaryOp.MODULUS) {
			currBop = new BinaryOp(target.getName(), target.getDataType(), target.getValueType(), OpOp2.MODULUS, left, right);
		} else if (source.getOpCode() == Expression.BinaryOp.INTDIV) {
			currBop = new BinaryOp(target.getName(), target.getDataType(), target.getValueType(), OpOp2.INTDIV, left, right);
		} else if (source.getOpCode() == Expression.BinaryOp.MATMULT) {
			currBop = new AggBinaryOp(target.getName(), target.getDataType(), target.getValueType(), OpOp2.MULT, AggOp.SUM, left, right);
		} else if (source.getOpCode() == Expression.BinaryOp.POW) {
			currBop = new BinaryOp(target.getName(), target.getDataType(), target.getValueType(), OpOp2.POW, left, right);
		}
		else {
			throw new ParseException("Unsupported parsing of binary expression: "+source.getOpCode());
		}
		setIdentifierParams(currBop, source.getOutput());
		currBop.setAllPositions(source.getBeginLine(), source.getBeginColumn(), source.getEndLine(), source.getEndColumn());
		return currBop;
		
	}

	private Hop processRelationalExpression(RelationalExpression source, DataIdentifier target,
			HashMap<String, Hop> hops) throws ParseException {

		Hop left = processExpression(source.getLeft(), null, hops);
		Hop right = processExpression(source.getRight(), null, hops);

		Hop currBop = null;

		if (target == null) {
			target = createTarget(source);
			if(left.getDataType() == DataType.MATRIX || right.getDataType() == DataType.MATRIX) {
				// Added to support matrix relational comparison
				target.setDataType(DataType.MATRIX);
				target.setValueType(ValueType.BOOLEAN);
			}
			else {
				// Added to support scalar relational comparison
				target.setDataType(DataType.SCALAR);
				target.setValueType(ValueType.BOOLEAN);
			}
		}
		
		OpOp2 op = null;

		if (source.getOpCode() == Expression.RelationalOp.LESS) {
			op = OpOp2.LESS;
		} else if (source.getOpCode() == Expression.RelationalOp.LESSEQUAL) {
			op = OpOp2.LESSEQUAL;
		} else if (source.getOpCode() == Expression.RelationalOp.GREATER) {
			op = OpOp2.GREATER;
		} else if (source.getOpCode() == Expression.RelationalOp.GREATEREQUAL) {
			op = OpOp2.GREATEREQUAL;
		} else if (source.getOpCode() == Expression.RelationalOp.EQUAL) {
			op = OpOp2.EQUAL;
		} else if (source.getOpCode() == Expression.RelationalOp.NOTEQUAL) {
			op = OpOp2.NOTEQUAL;
		}
		currBop = new BinaryOp(target.getName(), target.getDataType(), target.getValueType(), op, left, right);
		currBop.setAllPositions(source.getBeginLine(), source.getBeginColumn(), source.getEndLine(), source.getEndColumn());
		return currBop;
	}

	/**
	 * 
	 * @param source
	 * @param target
	 * @param hops
	 * @return
	 * @throws ParseException
	 */
	private Hop processBooleanExpression(BooleanExpression source, DataIdentifier target, HashMap<String, Hop> hops)
			throws ParseException 
	{
		// Boolean Not has a single parameter
		boolean constLeft = (source.getLeft().getOutput() instanceof ConstIdentifier);
		boolean constRight = false;
		if (source.getRight() != null) {
			constRight = (source.getRight().getOutput() instanceof ConstIdentifier);
		}

		if (constLeft || constRight) {
			LOG.error(source.printErrorLocation() + "Boolean expression with constant unsupported");
			throw new RuntimeException(source.printErrorLocation() + "Boolean expression with constant unsupported");
		}

		Hop left = processExpression(source.getLeft(), null, hops);
		Hop right = null;
		if (source.getRight() != null) {
			right = processExpression(source.getRight(), null, hops);
		}

	    //prepare target identifier and ensure that output type is boolean 
	    //(type should not be determined by target (e.g., string for print)
	    if (target == null) {
	        target = createTarget(source);
	    }
	    target.setValueType(ValueType.BOOLEAN);
		
		if (source.getRight() == null) {
			Hop currUop = null;
			try {
				currUop = new UnaryOp(target.getName(), target.getDataType(), target.getValueType(), Hop.OpOp1.NOT, left);
				currUop.setAllPositions(source.getBeginLine(), source.getBeginColumn(), source.getEndLine(), source.getEndColumn());
			} catch (HopsException e) {
				throw new ParseException(e.getMessage());
			}
			return currUop;
		} else {
			Hop currBop = null;
			OpOp2 op = null;

			if (source.getOpCode() == Expression.BooleanOp.LOGICALAND) {
				op = OpOp2.AND;
			} else if (source.getOpCode() == Expression.BooleanOp.LOGICALOR) {
				op = OpOp2.OR;
			} else {
				LOG.error(source.printErrorLocation() + "Unknown boolean operation " + source.getOpCode());
				throw new RuntimeException(source.printErrorLocation() + "Unknown boolean operation " + source.getOpCode());
			}
			currBop = new BinaryOp(target.getName(), target.getDataType(), target.getValueType(), op, left, right);
			currBop.setAllPositions(source.getBeginLine(), source.getBeginColumn(), source.getEndLine(), source.getEndColumn());
			// setIdentifierParams(currBop,source.getOutput());
			return currBop;
		}
	}

	private Hop constructDfHop(String name, DataType dt, ValueType vt, ParameterizedBuiltinFunctionOp op, HashMap<String,Hop> paramHops) throws HopsException {
		
		// Add a hop to paramHops to store distribution information. 
		// Distribution parameter hops would have been already present in paramHops.
		Hop distLop = null;
		switch(op) {
		case QNORM:
		case PNORM:
			distLop = new LiteralOp("normal");
			break;
		case QT:
		case PT:
			distLop = new LiteralOp("t");
			break;
		case QF:
		case PF:
			distLop = new LiteralOp("f");
			break;
		case QCHISQ:
		case PCHISQ:
			distLop = new LiteralOp("chisq");
			break;
		case QEXP:
		case PEXP:
			distLop = new LiteralOp("exp");
			break;
			
		case CDF:
		case INVCDF:
			break;
			
		default:
			throw new HopsException("Invalid operation: " + op);
		}
		if (distLop != null)
			paramHops.put("dist", distLop);
		
		return new ParameterizedBuiltinOp(name, dt, vt, ParameterizedBuiltinFunctionExpression.pbHopMap.get(op), paramHops);
	}
	
	/**
	 * 
	 * @param source
	 * @param targetList
	 * @param hops
	 * @return
	 * @throws ParseException
	 */
	private Hop processMultipleReturnParameterizedBuiltinFunctionExpression(ParameterizedBuiltinFunctionExpression source, ArrayList<DataIdentifier> targetList,
			HashMap<String, Hop> hops) throws ParseException 
	{
		FunctionType ftype = FunctionType.MULTIRETURN_BUILTIN;
		String nameSpace = DMLProgram.INTERNAL_NAMESPACE;
		
		// Create an array list to hold the outputs of this lop.
		// Exact list of outputs are added based on opcode.
		ArrayList<Hop> outputs = new ArrayList<Hop>();
		
		// Construct Hop for current builtin function expression based on its type
		Hop currBuiltinOp = null;
		switch (source.getOpCode()) {
			case TRANSFORMENCODE:
				ArrayList<Hop> inputs = new ArrayList<Hop>();
				inputs.add( processExpression(source.getVarParam("target"), null, hops) );
				inputs.add( processExpression(source.getVarParam("spec"), null, hops) );
				String[] outputNames = new String[targetList.size()]; 
				outputNames[0] = ((DataIdentifier)targetList.get(0)).getName();
				outputNames[1] = ((DataIdentifier)targetList.get(1)).getName();
				outputs.add(new DataOp(outputNames[0], DataType.MATRIX, ValueType.DOUBLE, inputs.get(0), DataOpTypes.FUNCTIONOUTPUT, outputNames[0]));
				outputs.add(new DataOp(outputNames[1], DataType.FRAME, ValueType.STRING, inputs.get(0), DataOpTypes.FUNCTIONOUTPUT, outputNames[1]));
				
				currBuiltinOp = new FunctionOp(ftype, nameSpace, source.getOpCode().toString(), inputs, outputNames, outputs);
				break;
				
			default:
				throw new ParseException("Invaid Opcode in DMLTranslator:processMultipleReturnParameterizedBuiltinFunctionExpression(): " + source.getOpCode());
		}

		// set properties for created hops based on outputs of source expression
		for ( int i=0; i < source.getOutputs().length; i++ ) {
			setIdentifierParams( outputs.get(i), source.getOutputs()[i]);
			outputs.get(i).setAllPositions(source.getBeginLine(), source.getBeginColumn(), source.getEndLine(), source.getEndColumn());
		}
		currBuiltinOp.setAllPositions(source.getBeginLine(), source.getBeginColumn(), source.getEndLine(), source.getEndColumn());

		return currBuiltinOp;
	}
	
	/**
	 * Construct Hops from parse tree : Process ParameterizedBuiltinFunction Expression in an
	 * assignment statement
	 * 
	 * @throws ParseException
	 * @throws HopsException 
	 */
	private Hop processParameterizedBuiltinFunctionExpression(ParameterizedBuiltinFunctionExpression source, DataIdentifier target,
			HashMap<String, Hop> hops) throws ParseException, HopsException {
		
		// this expression has multiple "named" parameters
		HashMap<String, Hop> paramHops = new HashMap<String,Hop>();
		
		// -- construct hops for all input parameters
		// -- store them in hashmap so that their "name"s are maintained
		Hop pHop = null;
		for ( String paramName : source.getVarParams().keySet() ) {
			pHop = processExpression(source.getVarParam(paramName), null, hops);
			paramHops.put(paramName, pHop);
		}
		
		Hop currBuiltinOp = null;

		if (target == null) {
			target = createTarget(source);
		}
		
		// construct hop based on opcode
		switch(source.getOpCode()) {
		case CDF:
		case INVCDF:
		case QNORM:
		case QT:
		case QF:
		case QCHISQ:
		case QEXP:
		case PNORM:
		case PT:
		case PF:
		case PCHISQ:
		case PEXP:
			currBuiltinOp = constructDfHop(target.getName(), target.getDataType(), target.getValueType(), source.getOpCode(), paramHops);
			break;
			
		case GROUPEDAGG:
			currBuiltinOp = new ParameterizedBuiltinOp(
					target.getName(), target.getDataType(), target.getValueType(), ParamBuiltinOp.GROUPEDAGG, paramHops);
			break;
		
		case RMEMPTY:
			currBuiltinOp = new ParameterizedBuiltinOp(
					target.getName(), target.getDataType(), target.getValueType(), ParamBuiltinOp.RMEMPTY, paramHops);
			break;
			
		case REPLACE:
			currBuiltinOp = new ParameterizedBuiltinOp(
					target.getName(), target.getDataType(), target.getValueType(), ParamBuiltinOp.REPLACE, paramHops);
			break;	
			
		case ORDER:
			ArrayList<Hop> inputs = new ArrayList<Hop>();
			inputs.add(paramHops.get("target"));
			inputs.add(paramHops.get("by"));
			inputs.add(paramHops.get("decreasing"));
			inputs.add(paramHops.get("index.return"));
			
			currBuiltinOp = new ReorgOp(target.getName(), target.getDataType(), target.getValueType(), ReOrgOp.SORT, inputs);
			
			break;
			
		case TRANSFORM:
			currBuiltinOp = new ParameterizedBuiltinOp(
									target.getName(), target.getDataType(), 
									target.getValueType(), ParamBuiltinOp.TRANSFORM, 
									paramHops);
			break;	
		
		case TRANSFORMAPPLY:
			currBuiltinOp = new ParameterizedBuiltinOp(
									target.getName(), target.getDataType(), 
									target.getValueType(), ParamBuiltinOp.TRANSFORMAPPLY, 
									paramHops);
			break;	
		
		case TRANSFORMDECODE:
			currBuiltinOp = new ParameterizedBuiltinOp(
									target.getName(), target.getDataType(), 
									target.getValueType(), ParamBuiltinOp.TRANSFORMDECODE, 
									paramHops);
			break;			

		case TRANSFORMMETA:
			currBuiltinOp = new ParameterizedBuiltinOp(
									target.getName(), target.getDataType(), 
									target.getValueType(), ParamBuiltinOp.TRANSFORMMETA, 
									paramHops);
			break;		
		
		case CAST_AS_STRING:
			currBuiltinOp = new ParameterizedBuiltinOp(
									target.getName(), target.getDataType(), 
									target.getValueType(), ParamBuiltinOp.CAST_AS_STRING, 
									paramHops);
			break;
			
		default:
			
			LOG.error(source.printErrorLocation() + 
					"processParameterizedBuiltinFunctionExpression() -- Unknown operation:  "
							+ source.getOpCode());
			
			throw new ParseException(source.printErrorLocation() + 
					"processParameterizedBuiltinFunctionExpression() -- Unknown operation:  "
							+ source.getOpCode());
		}
		
		setIdentifierParams(currBuiltinOp, source.getOutput());
		
		currBuiltinOp.setAllPositions(source.getBeginLine(), source.getBeginColumn(), source.getEndLine(), source.getEndColumn());
		
		return currBuiltinOp;
	}
	
	/**
	 * Construct Hops from parse tree : Process ParameterizedExpression in a
	 * read/write/rand statement
	 * 
	 * @throws ParseException
	 * @throws HopsException 
	 */
	private Hop processDataExpression(DataExpression source, DataIdentifier target,
			HashMap<String, Hop> hops) throws ParseException, HopsException {
		
		// this expression has multiple "named" parameters
		HashMap<String, Hop> paramHops = new HashMap<String,Hop>();
		
		// -- construct hops for all input parameters
		// -- store them in hashmap so that their "name"s are maintained
		Hop pHop = null; 
		for ( String paramName : source.getVarParams().keySet() ) {
			pHop = processExpression(source.getVarParam(paramName), null, hops);
			paramHops.put(paramName, pHop);
		}
		
		Hop currBuiltinOp = null;

		if (target == null) {
			target = createTarget(source);
		}
		
		// construct hop based on opcode
		switch(source.getOpCode()) {
		case READ:
			currBuiltinOp = new DataOp(target.getName(), target.getDataType(), target.getValueType(), DataOpTypes.PERSISTENTREAD, paramHops);
			((DataOp)currBuiltinOp).setFileName(((StringIdentifier)source.getVarParam(DataExpression.IO_FILENAME)).getValue());
			break;
			
		case WRITE:
			String name = target.getName();
			currBuiltinOp = new DataOp(
					target.getName(), target.getDataType(), target.getValueType(), DataOpTypes.PERSISTENTWRITE, hops.get(name), paramHops);
            //MB: commented for dynamic write
			/*Identifier ioFilename = (Identifier)source.getVarParam(DataExpression.IO_FILENAME);
			if (!(ioFilename instanceof StringIdentifier)) {
				LOG.error(source.printErrorLocation() + "processDataExpression():: Filename must be a constant string value");
				throw new ParseException(source.printErrorLocation() + "processDataExpression():: Filename must be a constant string value");
			} else {
				((DataOp)currBuiltinOp).setFileName(((StringIdentifier)ioFilename).getValue());
			}*/
			break;
			
		case RAND:
			// We limit RAND_MIN, RAND_MAX, RAND_SPARSITY, RAND_SEED, and RAND_PDF to be constants
			DataGenMethod method = (paramHops.get(DataExpression.RAND_MIN).getValueType()==ValueType.STRING) ?
					               DataGenMethod.SINIT : DataGenMethod.RAND;
			currBuiltinOp = new DataGenOp(method, target, paramHops);
			break;
		
		case MATRIX:
			ArrayList<Hop> tmp = new ArrayList<Hop>();
			tmp.add( 0, paramHops.get(DataExpression.RAND_DATA) );
			tmp.add( 1, paramHops.get(DataExpression.RAND_ROWS) );
			tmp.add( 2, paramHops.get(DataExpression.RAND_COLS) );
			tmp.add( 3, paramHops.get(DataExpression.RAND_BY_ROW) );
			currBuiltinOp = new ReorgOp(target.getName(), target.getDataType(), target.getValueType(), ReOrgOp.RESHAPE, tmp);
			break;
			
			
		default:
			LOG.error(source.printErrorLocation() + 
					"processDataExpression():: Unknown operation:  "
							+ source.getOpCode());
			
			throw new ParseException(source.printErrorLocation() + 
					"processDataExpression():: Unknown operation:  "
							+ source.getOpCode());
		}
		
		//set identifier meta data (incl dimensions and blocksizes)
		setIdentifierParams(currBuiltinOp, source.getOutput());
		if( source.getOpCode()==DataExpression.DataOp.READ )
			((DataOp)currBuiltinOp).setInputBlockSizes(target.getRowsInBlock(), target.getColumnsInBlock());
		currBuiltinOp.setAllPositions(source.getBeginLine(), source.getBeginColumn(), source.getEndLine(), source.getEndColumn());
		
		return currBuiltinOp;
	}

	/**
	 * Construct HOps from parse tree: process BuiltinFunction Expressions in 
	 * MultiAssignment Statements. For all other builtin function expressions,
	 * <code>processBuiltinFunctionExpression()</code> is used.
	 */
	private Hop processMultipleReturnBuiltinFunctionExpression(BuiltinFunctionExpression source, ArrayList<DataIdentifier> targetList,
			HashMap<String, Hop> hops) throws ParseException {
		
		// Construct Hops for all inputs
		ArrayList<Hop> inputs = new ArrayList<Hop>();
		inputs.add( processExpression(source.getFirstExpr(), null, hops) );
		if ( source.getSecondExpr() != null )
			inputs.add( processExpression(source.getSecondExpr(), null, hops) );
		if ( source.getThirdExpr() != null )
			inputs.add( processExpression(source.getThirdExpr(), null, hops) );
		
		FunctionType ftype = FunctionType.MULTIRETURN_BUILTIN;
		String nameSpace = DMLProgram.INTERNAL_NAMESPACE;
		
		// Create an array list to hold the outputs of this lop.
		// Exact list of outputs are added based on opcode.
		ArrayList<Hop> outputs = new ArrayList<Hop>();
		
		// Construct Hop for current builtin function expression based on its type
		Hop currBuiltinOp = null;
		switch (source.getOpCode()) {
		case QR:
		case LU:
		case EIGEN:
			
			// Number of outputs = size of targetList = #of identifiers in source.getOutputs
			String[] outputNames = new String[targetList.size()]; 
			for ( int i=0; i < targetList.size(); i++ ) {
				outputNames[i] = ((DataIdentifier)targetList.get(i)).getName();
				Hop output = new DataOp(outputNames[i], DataType.MATRIX, ValueType.DOUBLE, inputs.get(0), DataOpTypes.FUNCTIONOUTPUT, outputNames[i]);
				outputs.add(output);
			}
			
			// Create the hop for current function call
			FunctionOp fcall = new FunctionOp(ftype, nameSpace, source.getOpCode().toString(), inputs, outputNames, outputs);
			currBuiltinOp = fcall;
						
			break;
			
		default:
			throw new ParseException("Invaid Opcode in DMLTranslator:processMultipleReturnBuiltinFunctionExpression(): " + source.getOpCode());
		}

		// set properties for created hops based on outputs of source expression
		for ( int i=0; i < source.getOutputs().length; i++ ) {
			setIdentifierParams( outputs.get(i), source.getOutputs()[i]);
			outputs.get(i).setAllPositions(source.getBeginLine(), source.getBeginColumn(), source.getEndLine(), source.getEndColumn());
		}
		currBuiltinOp.setAllPositions(source.getBeginLine(), source.getBeginColumn(), source.getEndLine(), source.getEndColumn());

		return currBuiltinOp;
	}
	
	/**
	 * Construct Hops from parse tree : Process BuiltinFunction Expression in an
	 * assignment statement
	 * 
	 * @throws ParseException
	 * @throws HopsException 
	 */
	private Hop processBuiltinFunctionExpression(BuiltinFunctionExpression source, DataIdentifier target,
			HashMap<String, Hop> hops) throws ParseException, HopsException {
		Hop expr = processExpression(source.getFirstExpr(), null, hops);
		Hop expr2 = null;
		if (source.getSecondExpr() != null) {
			expr2 = processExpression(source.getSecondExpr(), null, hops);
		}
		Hop expr3 = null;
		if (source.getThirdExpr() != null) {
			expr3 = processExpression(source.getThirdExpr(), null, hops);
		}
		
		Hop currBuiltinOp = null;

		if (target == null) {
			target = createTarget(source);
		}

		// Construct the hop based on the type of Builtin function
		switch (source.getOpCode()) {

		case COLSUM:
			currBuiltinOp = new AggUnaryOp(target.getName(), target.getDataType(), target.getValueType(), AggOp.SUM,
					Direction.Col, expr);
			break;

		case COLMAX:
			currBuiltinOp = new AggUnaryOp(target.getName(), target.getDataType(), target.getValueType(), AggOp.MAX,
					Direction.Col, expr);
			break;

		case COLMIN:
			currBuiltinOp = new AggUnaryOp(target.getName(), target.getDataType(), target.getValueType(), AggOp.MIN,
					Direction.Col, expr);
			break;

		case COLMEAN:
			currBuiltinOp = new AggUnaryOp(target.getName(), target.getDataType(), target.getValueType(), AggOp.MEAN,
					Direction.Col, expr);
			break;

		case COLSD:
			// colStdDevs = sqrt(colVariances)
			currBuiltinOp = new AggUnaryOp(target.getName(), target.getDataType(),
					target.getValueType(), AggOp.VAR, Direction.Col, expr);
			currBuiltinOp = new UnaryOp(target.getName(), target.getDataType(),
					target.getValueType(), Hop.OpOp1.SQRT, currBuiltinOp);
			break;

		case COLVAR:
			currBuiltinOp = new AggUnaryOp(target.getName(), target.getDataType(),
					target.getValueType(), AggOp.VAR, Direction.Col, expr);
			break;

		case ROWSUM:
			currBuiltinOp = new AggUnaryOp(target.getName(), target.getDataType(), target.getValueType(), AggOp.SUM,
					Direction.Row, expr);
			break;

		case ROWMAX:
			currBuiltinOp = new AggUnaryOp(target.getName(), target.getDataType(), target.getValueType(), AggOp.MAX,
					Direction.Row, expr);
			break;

		case ROWINDEXMAX:
			currBuiltinOp = new AggUnaryOp(target.getName(), target.getDataType(), target.getValueType(), AggOp.MAXINDEX,
					Direction.Row, expr);
			break;
		
		case ROWINDEXMIN:
			currBuiltinOp = new AggUnaryOp(target.getName(), target.getDataType(), target.getValueType(), AggOp.MININDEX,
					Direction.Row, expr);
			break;
		
		case ROWMIN:
			currBuiltinOp = new AggUnaryOp(target.getName(), target.getDataType(), target.getValueType(), AggOp.MIN,
					Direction.Row, expr);
			break;

		case ROWMEAN:
			currBuiltinOp = new AggUnaryOp(target.getName(), target.getDataType(), target.getValueType(), AggOp.MEAN,
					Direction.Row, expr);
			break;

		case ROWSD:
			// rowStdDevs = sqrt(rowVariances)
			currBuiltinOp = new AggUnaryOp(target.getName(), target.getDataType(),
					target.getValueType(), AggOp.VAR, Direction.Row, expr);
			currBuiltinOp = new UnaryOp(target.getName(), target.getDataType(),
					target.getValueType(), Hop.OpOp1.SQRT, currBuiltinOp);
			break;

		case ROWVAR:
			currBuiltinOp = new AggUnaryOp(target.getName(), target.getDataType(),
					target.getValueType(), AggOp.VAR, Direction.Row, expr);
			break;

		case NROW:
			// If the dimensions are available at compile time, then create a LiteralOp (constant propagation)
			// Else create a UnaryOp so that a control program instruction is generated
			
			long nRows = expr.getDim1();
			if (nRows == -1) {
				currBuiltinOp = new UnaryOp(target.getName(), target.getDataType(), target.getValueType(), Hop.OpOp1.NROW, expr);
			}
			else {
				currBuiltinOp = new LiteralOp(nRows);
			}
			break;

		case NCOL:
			// If the dimensions are available at compile time, then create a LiteralOp (constant propagation)
			// Else create a UnaryOp so that a control program instruction is generated
			
			long nCols = expr.getDim2();
			if (nCols == -1) {
				currBuiltinOp = new UnaryOp(target.getName(), target.getDataType(), target.getValueType(), Hop.OpOp1.NCOL, expr);
			}
			else {
				currBuiltinOp = new LiteralOp(nCols);
			}
			break;
		case LENGTH:
			long nRows2 = expr.getDim1();
			long nCols2 = expr.getDim2();
			/* 
			 * If the dimensions are available at compile time, then create a LiteralOp (constant propagation)
			 * Else create a UnaryOp so that a control program instruction is generated
			 */
			if ((nCols2 == -1) || (nRows2 == -1)) {
				currBuiltinOp = new UnaryOp(target.getName(), target.getDataType(), target.getValueType(), Hop.OpOp1.LENGTH, expr);
			}
			else {
				long lval = (nCols2 * nRows2);
				currBuiltinOp = new LiteralOp(lval);
			}
			break;

		case SUM:
			currBuiltinOp = new AggUnaryOp(target.getName(), target.getDataType(), target.getValueType(), AggOp.SUM,
					Direction.RowCol, expr);
			break;
			
		case MEAN:
			if ( expr2 == null ) {
				// example: x = mean(Y);
				currBuiltinOp = new AggUnaryOp(target.getName(), target.getDataType(), target.getValueType(), AggOp.MEAN,
					Direction.RowCol, expr);
			}
			else {
				// example: x = mean(Y,W);
				// stable weighted mean is implemented by using centralMoment with order = 0
				Hop orderHop = new LiteralOp(0);
				currBuiltinOp=new TernaryOp(target.getName(), target.getDataType(), target.getValueType(), 
						Hop.OpOp3.CENTRALMOMENT, expr, expr2, orderHop);
			}
			break;

		case SD:
			// stdDev = sqrt(variance)
			currBuiltinOp = new AggUnaryOp(target.getName(), target.getDataType(),
					target.getValueType(), AggOp.VAR, Direction.RowCol, expr);
			currBuiltinOp = new UnaryOp(target.getName(), target.getDataType(),
					target.getValueType(), Hop.OpOp1.SQRT, currBuiltinOp);
			break;

		case VAR:
			currBuiltinOp = new AggUnaryOp(target.getName(), target.getDataType(),
					target.getValueType(), AggOp.VAR, Direction.RowCol, expr);
			break;

		case MIN:
			//construct AggUnary for min(X) but BinaryOp for min(X,Y)
			if( expr2 == null ) {
				currBuiltinOp = new AggUnaryOp(target.getName(), target.getDataType(), target.getValueType(),
						AggOp.MIN, Direction.RowCol, expr);
			} 
			else {
				currBuiltinOp = new BinaryOp(target.getName(), target.getDataType(), target.getValueType(), OpOp2.MIN,
						expr, expr2);
			}
			break;

		case MAX:
			//construct AggUnary for max(X) but BinaryOp for max(X,Y)
			if( expr2 == null ) {
				currBuiltinOp = new AggUnaryOp(target.getName(), target.getDataType(), target.getValueType(),
						AggOp.MAX, Direction.RowCol, expr);
			} else {
				currBuiltinOp = new BinaryOp(target.getName(), target.getDataType(), target.getValueType(), OpOp2.MAX,
						expr, expr2);
			}
			break;
			
		case PPRED:
			String sop = ((StringIdentifier)source.getThirdExpr()).getValue();
			sop = sop.replace("\"", "");
			OpOp2 operation;
			if ( sop.equalsIgnoreCase(">=") ) 
				operation = OpOp2.GREATEREQUAL;
			else if ( sop.equalsIgnoreCase(">") )
				operation = OpOp2.GREATER;
			else if ( sop.equalsIgnoreCase("<=") )
				operation = OpOp2.LESSEQUAL;
			else if ( sop.equalsIgnoreCase("<") )
				operation = OpOp2.LESS;
			else if ( sop.equalsIgnoreCase("==") )
				operation = OpOp2.EQUAL;
			else if ( sop.equalsIgnoreCase("!=") )
				operation = OpOp2.NOTEQUAL;
			else {
				LOG.error(source.printErrorLocation() + "Unknown argument (" + sop + ") for PPRED.");
				throw new ParseException(source.printErrorLocation() + "Unknown argument (" + sop + ") for PPRED.");
			}
			currBuiltinOp = new BinaryOp(target.getName(), target.getDataType(), target.getValueType(), operation, expr, expr2);
			break;
			
		case PROD:
			currBuiltinOp = new AggUnaryOp(target.getName(), target.getDataType(), target.getValueType(), AggOp.PROD,
					Direction.RowCol, expr);
			break;
		case TRACE:
			currBuiltinOp = new AggUnaryOp(target.getName(), target.getDataType(), target.getValueType(), AggOp.TRACE,
					Direction.RowCol, expr);
			break;

		case TRANS:
			currBuiltinOp = new ReorgOp(target.getName(), target.getDataType(), target.getValueType(),
					                    Hop.ReOrgOp.TRANSPOSE, expr);
			break;
		
		case REV:
			currBuiltinOp = new ReorgOp(target.getName(), target.getDataType(), target.getValueType(),
					                    Hop.ReOrgOp.REV, expr);
			break;
			
		case CBIND:
			currBuiltinOp = new BinaryOp(target.getName(), target.getDataType(), target.getValueType(), 
										Hop.OpOp2.CBIND, expr, expr2);
			break;
		
		case RBIND:
			currBuiltinOp = new BinaryOp(target.getName(), target.getDataType(), target.getValueType(), 
										Hop.OpOp2.RBIND, expr, expr2);
			break;
		
		case DIAG:
			currBuiltinOp = new ReorgOp(target.getName(), target.getDataType(), target.getValueType(),
						                Hop.ReOrgOp.DIAG, expr);
			break;
			
		case TABLE:
			
			// Always a TertiaryOp is created for table().
			// - create a hop for weights, if not provided in the function call.
			int numTableArgs = source._args.length;
			
			switch(numTableArgs) {
			case 2:
			case 4:
				// example DML statement: F = ctable(A,B) or F = ctable(A,B,10,15)
				// here, weight is interpreted as 1.0
				Hop weightHop = new LiteralOp(1.0);
				// set dimensions
				weightHop.setDim1(0);
				weightHop.setDim2(0);
				weightHop.setNnz(-1);
				weightHop.setRowsInBlock(0);
				weightHop.setColsInBlock(0);
				
				if ( numTableArgs == 2 )
					currBuiltinOp = new TernaryOp(target.getName(), target.getDataType(), target.getValueType(), OpOp3.CTABLE, expr, expr2, weightHop);
				else {
					Hop outDim1 = processExpression(source._args[2], null, hops);
					Hop outDim2 = processExpression(source._args[3], null, hops);
					currBuiltinOp = new TernaryOp(target.getName(), target.getDataType(), target.getValueType(), OpOp3.CTABLE, expr, expr2, weightHop, outDim1, outDim2);
				}
				break;
				
			case 3:
			case 5:
				// example DML statement: F = ctable(A,B,W) or F = ctable(A,B,W,10,15) 
				if (numTableArgs == 3) 
					currBuiltinOp = new TernaryOp(target.getName(), target.getDataType(), target.getValueType(), OpOp3.CTABLE, expr, expr2, expr3);
				else {
					Hop outDim1 = processExpression(source._args[3], null, hops);
					Hop outDim2 = processExpression(source._args[4], null, hops);
					currBuiltinOp = new TernaryOp(target.getName(), target.getDataType(), target.getValueType(), OpOp3.CTABLE, expr, expr2, expr3, outDim1, outDim2);
				}
				break;
				
			default: 
				throw new ParseException("Invalid number of arguments "+ numTableArgs + " to table() function.");
			}
			break;

		//data type casts	
		case CAST_AS_SCALAR:
			currBuiltinOp = new UnaryOp(target.getName(), DataType.SCALAR, target.getValueType(), Hop.OpOp1.CAST_AS_SCALAR, expr);
			break;
		case CAST_AS_MATRIX:
			currBuiltinOp = new UnaryOp(target.getName(), DataType.MATRIX, target.getValueType(), Hop.OpOp1.CAST_AS_MATRIX, expr);
			break;	
		case CAST_AS_FRAME:
			currBuiltinOp = new UnaryOp(target.getName(), DataType.FRAME, target.getValueType(), Hop.OpOp1.CAST_AS_FRAME, expr);
			break;	

		//value type casts
		case CAST_AS_DOUBLE:
			currBuiltinOp = new UnaryOp(target.getName(), target.getDataType(), ValueType.DOUBLE, Hop.OpOp1.CAST_AS_DOUBLE, expr);
			break;
		case CAST_AS_INT:
			currBuiltinOp = new UnaryOp(target.getName(), target.getDataType(), ValueType.INT, Hop.OpOp1.CAST_AS_INT, expr);
			break;
		case CAST_AS_BOOLEAN:
			currBuiltinOp = new UnaryOp(target.getName(), target.getDataType(), ValueType.BOOLEAN, Hop.OpOp1.CAST_AS_BOOLEAN, expr);
			break;
		case ABS:
		case SIN:
		case COS:
		case TAN:
		case ASIN:
		case ACOS:
		case ATAN:
		case SIGN:	
		case SQRT:
		case EXP:
		case ROUND:
		case CEIL:
		case FLOOR:
		case CUMSUM:
		case CUMPROD:
		case CUMMIN:
		case CUMMAX:
			Hop.OpOp1 mathOp1;
			switch (source.getOpCode()) {
			case ABS:
				mathOp1 = Hop.OpOp1.ABS;
				break;
			case SIN:
				mathOp1 = Hop.OpOp1.SIN;
				break;
			case COS:
				mathOp1 = Hop.OpOp1.COS;
				break;
			case TAN:
				mathOp1 = Hop.OpOp1.TAN;
				break;
			case ASIN:
				mathOp1 = Hop.OpOp1.ASIN;
				break;
			case ACOS:
				mathOp1 = Hop.OpOp1.ACOS;
				break;
			case ATAN:
				mathOp1 = Hop.OpOp1.ATAN;
				break;
			case SIGN:
				mathOp1 = Hop.OpOp1.SIGN;
				break;
			case SQRT:
				mathOp1 = Hop.OpOp1.SQRT;
				break;
			case EXP:
				mathOp1 = Hop.OpOp1.EXP;
				break;
			case ROUND:
				mathOp1 = Hop.OpOp1.ROUND;
				break;
			case CEIL:
				mathOp1 = Hop.OpOp1.CEIL;
				break;
			case FLOOR:
				mathOp1 = Hop.OpOp1.FLOOR;
				break;
			case CUMSUM:
				mathOp1 = Hop.OpOp1.CUMSUM;
				break;
			case CUMPROD:
				mathOp1 = Hop.OpOp1.CUMPROD;
				break;
			case CUMMIN:
				mathOp1 = Hop.OpOp1.CUMMIN;
				break;
			case CUMMAX:
				mathOp1 = Hop.OpOp1.CUMMAX;
				break;
			default:
				
				LOG.error(source.printErrorLocation() +
						"processBuiltinFunctionExpression():: Could not find Operation type for builtin function: "
								+ source.getOpCode());
				
				throw new ParseException(source.printErrorLocation() +
						"processBuiltinFunctionExpression():: Could not find Operation type for builtin function: "
								+ source.getOpCode());
			}
			currBuiltinOp = new UnaryOp(target.getName(), target.getDataType(), target.getValueType(), mathOp1, expr);
			break;
		case LOG:
				if (expr2 == null) {
					Hop.OpOp1 mathOp2;
					switch (source.getOpCode()) {
					case LOG:
						mathOp2 = Hop.OpOp1.LOG;
						break;
					default:
						
						LOG.error(source.printErrorLocation() +
								"processBuiltinFunctionExpression():: Could not find Operation type for builtin function: "
										+ source.getOpCode());
						
						throw new ParseException(source.printErrorLocation() +
								"processBuiltinFunctionExpression():: Could not find Operation type for builtin function: "
										+ source.getOpCode());
					}
					currBuiltinOp = new UnaryOp(target.getName(), target.getDataType(), target.getValueType(), mathOp2,
							expr);
				} else {
					Hop.OpOp2 mathOp3;
					switch (source.getOpCode()) {
					case LOG:
						mathOp3 = Hop.OpOp2.LOG;
						break;
					default:
						
						LOG.error(source.printErrorLocation() +
								"processBuiltinFunctionExpression():: Could not find Operation type for builtin function: "
										+ source.getOpCode());
						
						throw new ParseException(source.printErrorLocation() +
								"processBuiltinFunctionExpression():: Could not find Operation type for builtin function: "
										+ source.getOpCode());
					}
					currBuiltinOp = new BinaryOp(target.getName(), target.getDataType(), target.getValueType(), mathOp3,
							expr, expr2);
				}
			break;
		case MOMENT:
			if (expr3 == null){
				currBuiltinOp=new BinaryOp(target.getName(), target.getDataType(), target.getValueType(), 
						Hop.OpOp2.CENTRALMOMENT, expr, expr2);
			}
			else {
				currBuiltinOp=new TernaryOp(target.getName(), target.getDataType(), target.getValueType(), 
						Hop.OpOp3.CENTRALMOMENT, expr, expr2,expr3);
			}
			break;
			
		case COV:
			if (expr3 == null){
				currBuiltinOp=new BinaryOp(target.getName(), target.getDataType(), target.getValueType(), 
						Hop.OpOp2.COVARIANCE, expr, expr2);
			}
			else {
				currBuiltinOp=new TernaryOp(target.getName(), target.getDataType(), target.getValueType(), 
						Hop.OpOp3.COVARIANCE, expr, expr2,expr3);
			}
			break;
			
		case QUANTILE:
			if (expr3 == null){
				currBuiltinOp=new BinaryOp(target.getName(), target.getDataType(), target.getValueType(), 
						Hop.OpOp2.QUANTILE, expr, expr2);
			}
			else {
				currBuiltinOp=new TernaryOp(target.getName(), target.getDataType(), target.getValueType(), 
						Hop.OpOp3.QUANTILE, expr, expr2,expr3);
			}
			break;
			
		case INTERQUANTILE:
			if ( expr3 == null ) {
				currBuiltinOp=new BinaryOp(target.getName(), target.getDataType(), target.getValueType(), 
						Hop.OpOp2.INTERQUANTILE, expr, expr2);
			}
			else {
				currBuiltinOp=new TernaryOp(target.getName(), target.getDataType(), target.getValueType(), 
					Hop.OpOp3.INTERQUANTILE, expr, expr2,expr3);
			}
			break;	
			
		case IQM:
			if ( expr2 == null ) {
				currBuiltinOp=new UnaryOp(target.getName(), target.getDataType(), target.getValueType(), 
						Hop.OpOp1.IQM, expr);
			}
			else {
				currBuiltinOp=new BinaryOp(target.getName(), target.getDataType(), target.getValueType(), 
					Hop.OpOp2.IQM, expr, expr2);
			}
			break;	
		
		case MEDIAN:
			if ( expr2 == null ) {
				currBuiltinOp=new UnaryOp(target.getName(), target.getDataType(), target.getValueType(), 
						Hop.OpOp1.MEDIAN, expr);
			}
			else {
				currBuiltinOp=new BinaryOp(target.getName(), target.getDataType(), target.getValueType(), 
					Hop.OpOp2.MEDIAN, expr, expr2);
			}
			break;	
		
		case SEQ:
			HashMap<String,Hop> randParams = new HashMap<String,Hop>();
			randParams.put(Statement.SEQ_FROM, expr);
			randParams.put(Statement.SEQ_TO, expr2);
			randParams.put(Statement.SEQ_INCR, (expr3!=null)?expr3 : new LiteralOp(1)); 
			//note incr: default -1 (for from>to) handled during runtime
			currBuiltinOp = new DataGenOp(DataGenMethod.SEQ, target, randParams);
			break;
			
		case SAMPLE: 
		{
			Expression[] in = source.getAllExpr();
			
			// arguments: range/size/replace/seed; defaults: replace=FALSE
				
			HashMap<String,Hop> tmpparams = new HashMap<String, Hop>();
			tmpparams.put(DataExpression.RAND_MAX, expr); //range
			tmpparams.put(DataExpression.RAND_ROWS, expr2);
			tmpparams.put(DataExpression.RAND_COLS, new LiteralOp(1));
			
			if ( in.length == 4 ) 
			{
				tmpparams.put(DataExpression.RAND_PDF, expr3);
				Hop seed = processExpression(in[3], null, hops);
				tmpparams.put(DataExpression.RAND_SEED, seed);
			}
			else if ( in.length == 3 )
			{
				// check if the third argument is "replace" or "seed"
				if ( expr3.getValueType() == ValueType.BOOLEAN ) 
				{
					tmpparams.put(DataExpression.RAND_PDF, expr3);
					tmpparams.put(DataExpression.RAND_SEED, new LiteralOp(DataGenOp.UNSPECIFIED_SEED) );
				}
				else if ( expr3.getValueType() == ValueType.INT ) 
				{
					tmpparams.put(DataExpression.RAND_PDF, new LiteralOp(false));
					tmpparams.put(DataExpression.RAND_SEED, expr3 );
				}
				else 
					throw new HopsException("Invalid input type " + expr3.getValueType() + " in sample().");
				
			}
			else if ( in.length == 2 )
			{
				tmpparams.put(DataExpression.RAND_PDF, new LiteralOp(false));
				tmpparams.put(DataExpression.RAND_SEED, new LiteralOp(DataGenOp.UNSPECIFIED_SEED) );
			}
			
			currBuiltinOp = new DataGenOp(DataGenMethod.SAMPLE, target, tmpparams);
			break;
		}
		
		case SOLVE:
			currBuiltinOp = new BinaryOp(target.getName(), target.getDataType(), target.getValueType(), Hop.OpOp2.SOLVE, expr, expr2);
			break;
			
		case INVERSE:
			currBuiltinOp = new UnaryOp(target.getName(), target.getDataType(), target.getValueType(), 
					Hop.OpOp1.INVERSE, expr);
			break;
		
		case CHOLESKY:
			currBuiltinOp = new UnaryOp(target.getName(), target.getDataType(), target.getValueType(), 
					Hop.OpOp1.CHOLESKY, expr);
			break;	
			
		case OUTER:
			if( !(expr3 instanceof LiteralOp) )
				throw new HopsException("Operator for outer builtin function must be a constant: "+expr3);			
			OpOp2 op = Hop.getOpOp2ForOuterVectorOperation(((LiteralOp)expr3).getStringValue());
			if( op == null )
				throw new HopsException("Unsupported outer vector binary operation: "+((LiteralOp)expr3).getStringValue());
			
			currBuiltinOp = new BinaryOp(target.getName(), target.getDataType(), target.getValueType(), op, expr, expr2);
			((BinaryOp)currBuiltinOp).setOuterVectorOperation(true); //flag op as specific outer vector operation
			currBuiltinOp.refreshSizeInformation(); //force size reevaluation according to 'outer' flag otherwise danger of incorrect dims
			break;
			
		default:
			throw new ParseException("Unsupported builtin function type: "+source.getOpCode());
		}
		
		setIdentifierParams(currBuiltinOp, source.getOutput());
		currBuiltinOp.setAllPositions(source.getBeginLine(), source.getBeginColumn(), source.getEndLine(), source.getEndColumn());
		return currBuiltinOp;
	}
		
	public void setIdentifierParams(Hop h, Identifier id) {
		if( id.getDim1()>= 0 )
			h.setDim1(id.getDim1());
		if( id.getDim2()>= 0 )
			h.setDim2(id.getDim2());
		if( id.getNnz()>= 0 )
			h.setNnz(id.getNnz());
		h.setRowsInBlock(id.getRowsInBlock());
		h.setColsInBlock(id.getColumnsInBlock());
	}

	public void setIdentifierParams(Hop h, Hop source) {

		h.setDim1(source.getDim1());
		h.setDim2(source.getDim2());
		h.setNnz(source.getNnz());
		h.setRowsInBlock(source.getRowsInBlock());
		h.setColsInBlock(source.getColsInBlock());
	}

	/**
	 * 
	 * @param prog
	 * @param pWrites
	 * @throws LanguageException 
	 */
	private boolean prepareReadAfterWrite( DMLProgram prog, HashMap<String, DataIdentifier> pWrites ) 
		throws LanguageException
	{
		boolean ret = false;
		
		//process functions 
		/*MB: for the moment we only support read-after-write in the main program 
		for( FunctionStatementBlock fsb : prog.getFunctionStatementBlocks() )
			ret |= prepareReadAfterWrite(fsb, pWrites);
		*/
		
		//process main program
		for( StatementBlock sb : prog.getStatementBlocks() )
			ret |= prepareReadAfterWrite(sb, pWrites);
	
		return ret;
	}
	
	/**
	 * 
	 * @param sb
	 * @param pWrites
	 */
	private boolean prepareReadAfterWrite( StatementBlock sb, HashMap<String, DataIdentifier> pWrites )
	{
		boolean ret = false;
		
		if(sb instanceof FunctionStatementBlock)
		{
			FunctionStatementBlock fsb = (FunctionStatementBlock) sb;
			FunctionStatement fstmt = (FunctionStatement)fsb.getStatement(0);
			for (StatementBlock csb : fstmt.getBody())
				ret |= prepareReadAfterWrite(csb, pWrites);
		}
		else if(sb instanceof WhileStatementBlock)
		{
			WhileStatementBlock wsb = (WhileStatementBlock) sb;
			WhileStatement wstmt = (WhileStatement)wsb.getStatement(0);
			for (StatementBlock csb : wstmt.getBody())
				ret |= prepareReadAfterWrite(csb, pWrites);
		}	
		else if(sb instanceof IfStatementBlock)
		{
			IfStatementBlock isb = (IfStatementBlock) sb;
			IfStatement istmt = (IfStatement)isb.getStatement(0);
			for (StatementBlock csb : istmt.getIfBody())
				ret |= prepareReadAfterWrite(csb, pWrites);
			for (StatementBlock csb : istmt.getElseBody())
				ret |= prepareReadAfterWrite(csb, pWrites);
		}
		else if(sb instanceof ForStatementBlock) //incl parfor
		{
			ForStatementBlock fsb = (ForStatementBlock) sb;
			ForStatement fstmt = (ForStatement)fsb.getStatement(0);
			for (StatementBlock csb : fstmt.getBody())
				ret |= prepareReadAfterWrite(csb, pWrites);
		}
		else //generic (last-level)
		{
			for( Statement s : sb.getStatements() )
			{
				//collect persistent write information
				if( s instanceof OutputStatement )
				{
					OutputStatement os = (OutputStatement) s;
					String pfname = os.getExprParam(DataExpression.IO_FILENAME).toString();
					DataIdentifier di = (DataIdentifier) os.getSource().getOutput();
					pWrites.put(pfname, di);
				}
				//propagate size information into reads-after-write
				else if( s instanceof AssignmentStatement 
						&& ((AssignmentStatement)s).getSource() instanceof DataExpression )
				{
					DataExpression dexpr = (DataExpression) ((AssignmentStatement)s).getSource();
					if( dexpr.isRead() ){
						String pfname = dexpr.getVarParam(DataExpression.IO_FILENAME).toString();
						if( pWrites.containsKey(pfname) && !pfname.trim().isEmpty() ) //found read-after-write
						{
							//update read with essential write meta data
							DataIdentifier di = pWrites.get(pfname);
							FormatType ft = (di.getFormatType()!=null) ? di.getFormatType() : FormatType.TEXT;
							dexpr.addVarParam(DataExpression.FORMAT_TYPE, new StringIdentifier(ft.toString(),di.getFilename(),di.getBeginLine(),di.getBeginColumn(),di.getEndLine(),di.getEndColumn()));							
							if( di.getDim1()>=0 )
								dexpr.addVarParam(DataExpression.READROWPARAM, new IntIdentifier(di.getDim1(),di.getFilename(),di.getBeginLine(),di.getBeginColumn(),di.getEndLine(),di.getEndColumn()));
							if( di.getDim2()>=0 )
								dexpr.addVarParam(DataExpression.READCOLPARAM, new IntIdentifier(di.getDim2(),di.getFilename(),di.getBeginLine(),di.getBeginColumn(),di.getEndLine(),di.getEndColumn()));
							if( di.getValueType()!=ValueType.UNKNOWN )
								dexpr.addVarParam(DataExpression.VALUETYPEPARAM, new StringIdentifier(di.getValueType().toString(),di.getFilename(),di.getBeginLine(),di.getBeginColumn(),di.getEndLine(),di.getEndColumn()));
							if( di.getDataType()!=DataType.UNKNOWN )
								dexpr.addVarParam(DataExpression.DATATYPEPARAM, new StringIdentifier(di.getDataType().toString(),di.getFilename(),di.getBeginLine(),di.getBeginColumn(),di.getEndLine(),di.getEndColumn()));
							ret = true;
						}
					}
				}
			}
		}
		
		return ret;
	}
}
