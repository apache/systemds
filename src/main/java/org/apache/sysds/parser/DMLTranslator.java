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
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Builtins;
import org.apache.sysds.common.Types.AggOp;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.Direction;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.OpOp1;
import org.apache.sysds.common.Types.OpOp2;
import org.apache.sysds.common.Types.OpOp3;
import org.apache.sysds.common.Types.OpOpDG;
import org.apache.sysds.common.Types.OpOpData;
import org.apache.sysds.common.Types.OpOpDnn;
import org.apache.sysds.common.Types.OpOpN;
import org.apache.sysds.common.Types.ParamBuiltinOp;
import org.apache.sysds.common.Types.ReOrgOp;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.hops.AggUnaryOp;
import org.apache.sysds.hops.BinaryOp;
import org.apache.sysds.hops.DataGenOp;
import org.apache.sysds.hops.DataOp;
import org.apache.sysds.hops.DnnOp;
import org.apache.sysds.hops.FunctionOp;
import org.apache.sysds.hops.FunctionOp.FunctionType;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.HopsException;
import org.apache.sysds.hops.IndexingOp;
import org.apache.sysds.hops.LeftIndexingOp;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.MemoTable;
import org.apache.sysds.hops.NaryOp;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.ParameterizedBuiltinOp;
import org.apache.sysds.hops.ReorgOp;
import org.apache.sysds.hops.TernaryOp;
import org.apache.sysds.hops.UnaryOp;
import org.apache.sysds.hops.codegen.SpoofCompiler;
import org.apache.sysds.hops.codegen.SpoofCompiler.IntegrationType;
import org.apache.sysds.hops.codegen.SpoofCompiler.PlanCachePolicy;
import org.apache.sysds.hops.ipa.InterProceduralAnalysis;
import org.apache.sysds.hops.recompile.Recompiler;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.hops.rewrite.ProgramRewriter;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.lops.LopsException;
import org.apache.sysds.lops.compile.Dag;
import org.apache.sysds.lops.rewrite.LopRewriter;
import org.apache.sysds.parser.PrintStatement.PRINTTYPE;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.BasicProgramBlock;
import org.apache.sysds.runtime.controlprogram.ForProgramBlock;
import org.apache.sysds.runtime.controlprogram.FunctionProgramBlock;
import org.apache.sysds.runtime.controlprogram.IfProgramBlock;
import org.apache.sysds.runtime.controlprogram.ParForProgramBlock;
import org.apache.sysds.runtime.controlprogram.Program;
import org.apache.sysds.runtime.controlprogram.ProgramBlock;
import org.apache.sysds.runtime.controlprogram.WhileProgramBlock;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.cp.VariableCPInstruction;

public class DMLTranslator 
{
	private static final Log LOG = LogFactory.getLog(DMLTranslator.class.getName());
	private DMLProgram _dmlProg;

	public DMLTranslator(DMLProgram dmlp) {
		_dmlProg = dmlp;
		//setup default size for unknown dimensions
		OptimizerUtils.resetDefaultSize();
		//reinit rewriter according to opt level flags
		Recompiler.reinitRecompiler(); 
	}

	public void validateParseTree(DMLProgram dmlp) {
		validateParseTree(dmlp, true);
	}

	public void validateParseTree(DMLProgram dmlp, boolean inclFuns) 
	{
		//STEP1: Pre-processing steps for validate - e.g., prepare read-after-write meta data
		boolean fWriteRead = prepareReadAfterWrite(dmlp, new HashMap<String, DataIdentifier>());

		//STEP2: Actual Validate
		if( inclFuns ) {
			// handle functions in namespaces (current program has default namespace)
			for (String namespaceKey : dmlp.getNamespaces().keySet()) {
				// for each function defined in the namespace
				for (String fname :  dmlp.getFunctionStatementBlocks(namespaceKey).keySet()) {
					FunctionStatementBlock fblock = dmlp.getFunctionStatementBlock(namespaceKey,fname);
					validateFunction(dmlp, fblock);
				}
			}
		}

		// handle regular blocks -- "main" program
		VariableSet vs = new VariableSet();
		HashMap<String, ConstIdentifier> constVars = new HashMap<>();
		for (int i = 0; i < dmlp.getNumStatementBlocks(); i++) {
			StatementBlock sb = dmlp.getStatementBlock(i);
			vs = sb.validate(dmlp, vs, constVars, fWriteRead);
			constVars = sb.getConstOut();
		}

		//STEP3: Post-processing steps after validate - e.g., prepare read-after-write meta data
		if( fWriteRead ) 
		{
			//propagate size and datatypes into read
			prepareReadAfterWrite(dmlp, new HashMap<>());

			//re-validate main program for datatype propagation
			vs = new VariableSet();
			constVars = new HashMap<>();
			for (int i = 0; i < dmlp.getNumStatementBlocks(); i++) {
				StatementBlock sb = dmlp.getStatementBlock(i);
				vs = sb.validate(dmlp, vs, constVars, fWriteRead);
				constVars = sb.getConstOut();
			}
		}
	}

	public void validateFunction(DMLProgram dmlp, FunctionStatementBlock fsb) {
		validateFunction(dmlp, fsb, false);
	}

	public void validateFunction(DMLProgram dmlp, FunctionStatementBlock fsb, boolean conditional) {
		HashMap<String, ConstIdentifier> constVars = new HashMap<>();
		VariableSet vs = new VariableSet();

		// add the input variables for the function to input variable list
		FunctionStatement fstmt = (FunctionStatement)fsb.getStatement(0);
		for (DataIdentifier currVar : fstmt.getInputParams()) {
			if (currVar.getDataType() == DataType.SCALAR)
				currVar.setDimensions(0, 0);
			vs.addVariable(currVar.getName(), currVar);
		}
		fsb.validate(dmlp, vs, constVars, conditional);
	}

	public void liveVariableAnalysis(DMLProgram dmlp) {
		liveVariableAnalysis(dmlp, true);
	}

	public void liveVariableAnalysis(DMLProgram dmlp, boolean inclFuns) {

		// for each namespace, handle function statement blocks
		if( inclFuns ) {
			for (String namespaceKey : dmlp.getNamespaces().keySet()) {
				for (String fname: dmlp.getFunctionStatementBlocks(namespaceKey).keySet()) {
					FunctionStatementBlock fsb = dmlp.getFunctionStatementBlock(namespaceKey, fname);
					liveVariableAnalysisFunction(dmlp, fsb);
				}
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

		cleanupLiveOutVariables(dmlp.getStatementBlocks(), new VariableSet());
	}

	public void liveVariableAnalysisFunction(DMLProgram dmlp, FunctionStatementBlock fsb) {
		//STEP 1: forward direction
		FunctionStatement fstmt = (FunctionStatement)fsb.getStatement(0);

		// perform function inlining
		fstmt.setBody(StatementBlock.mergeFunctionCalls(fstmt.getBody(), dmlp));

		VariableSet activeIn = new VariableSet();
		for (DataIdentifier id : fstmt.getInputParams()){
			activeIn.addVariable(id.getName(), id); 
		}
		fsb.initializeforwardLV(activeIn);

		//STEP 2: backward direction
		VariableSet currentLiveOut = new VariableSet();
		VariableSet currentLiveIn = new VariableSet();
		VariableSet unionLiveIn = new VariableSet();

		for (DataIdentifier id : fstmt.getInputParams())
			currentLiveIn.addVariable(id.getName(), id);

		for (DataIdentifier id : fstmt.getOutputParams()) {
			currentLiveOut.addVariable(id.getName(), id);
			unionLiveIn.addVariable(id.getName(), id);
		}

		fsb._liveOut = currentLiveOut;
		fsb.analyze(currentLiveIn, currentLiveOut);
		cleanupLiveOutVariables(fstmt.getBody(), unionLiveIn);
	}

	public void cleanupLiveOutVariables(List<StatementBlock> sbs, VariableSet unionLiveIn) {
		//backwards pass to collect union of livein variables of all successors
		//and cleanup unnecessary liveout variables
		for(int i=sbs.size()-1; i>=0; i--) {
			StatementBlock sb = sbs.get(i);
			//remove liveout variables that are not in unionLivein
			sb.liveOut().removeVariables(
				VariableSet.minus(sb.liveOut(), unionLiveIn));
			//collect all livein information
			unionLiveIn.addVariables(sb.liveIn());
		}
	}

	public void constructHops(DMLProgram dmlp) {
		constructHops(dmlp, true);
	}

	public void constructHops(DMLProgram dmlp, boolean inclFuns) {
		// Step 1: construct hops for all functions
		if( inclFuns ) {
			// for each namespace, handle function program blocks
			for( FunctionDictionary<FunctionStatementBlock> fdict : dmlp.getNamespaces().values() )
				for( FunctionStatementBlock fsb : fdict.getFunctions().values() )
					constructHops(fsb);
		}

		// Step 2: construct hops for main program
		// handle regular program blocks
		for (int i = 0; i < dmlp.getNumStatementBlocks(); i++) {
			StatementBlock current = dmlp.getStatementBlock(i);
			constructHops(current);
		}
	}

	public void rewriteHopsDAG(DMLProgram dmlp) 
	{
		//apply hop rewrites (static rewrites)
		ProgramRewriter rewriter = new ProgramRewriter(true, false);
		rewriter.rewriteProgramHopDAGs(dmlp, false); //rewrite and merge
		resetHopsDAGVisitStatus(dmlp);
		rewriter.rewriteProgramHopDAGs(dmlp, true); //rewrite and split
		resetHopsDAGVisitStatus(dmlp);

		//propagate size information from main into functions (but conservatively)
		if( OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS ) {
			InterProceduralAnalysis ipa = new InterProceduralAnalysis(dmlp);
			ipa.analyzeProgram(OptimizerUtils.IPA_NUM_REPETITIONS);
			resetHopsDAGVisitStatus(dmlp);
		}

		//apply hop rewrites (dynamic rewrites, after IPA)
		ProgramRewriter rewriter2 = new ProgramRewriter(false, true);
		rewriter2.rewriteProgramHopDAGs(dmlp);
		resetHopsDAGVisitStatus(dmlp);

		//compute memory estimates for all the hops. These estimates are used
		//subsequently in various optimizations, e.g. CP vs. MR scheduling and parfor.
		refreshMemEstimates(dmlp);
		resetHopsDAGVisitStatus(dmlp);

		//enhance HOP DAGs by automatic operator fusion
		DMLConfig dmlconf = ConfigurationManager.getDMLConfig();
		if( ConfigurationManager.isCodegenEnabled() ){
			SpoofCompiler.PLAN_CACHE_POLICY = PlanCachePolicy.get(
				dmlconf.getBooleanValue(DMLConfig.CODEGEN_PLANCACHE),
				dmlconf.getIntValue(DMLConfig.CODEGEN_LITERALS)==2);
			SpoofCompiler.setConfiguredPlanSelector();
			SpoofCompiler.setExecTypeSpecificJavaCompiler();
			if( SpoofCompiler.INTEGRATION==IntegrationType.HOPS )
				codgenHopsDAG(dmlp);
		}
	}

	public void rewriteLopDAG(DMLProgram dmlp) {
		LopRewriter rewriter = new LopRewriter();
		rewriter.rewriteProgramLopDAGs(dmlp);
	}

	public void codgenHopsDAG(DMLProgram dmlp) {
		SpoofCompiler.generateCode(dmlp);
	}

	public void codgenHopsDAG(Program rtprog) {
		SpoofCompiler.generateCode(rtprog);
	}

	public void codgenHopsDAG(ProgramBlock pb) {
		SpoofCompiler.generateCodeFromProgramBlock(pb);
	}

	public void constructLops(DMLProgram dmlp) {
		// for each namespace, handle function program blocks
		for( FunctionDictionary<FunctionStatementBlock> fdict : dmlp.getNamespaces().values() ) {
			//handle optimized functions
			for( FunctionStatementBlock fsb : fdict.getFunctions().values() )
				constructLops(fsb);
			//handle unoptimized functions
			if( fdict.getFunctions(false) != null )
				for( FunctionStatementBlock fsb : fdict.getFunctions(false).values() )
					constructLops(fsb);
		}

		// handle regular program blocks
		for( StatementBlock sb : dmlp.getStatementBlocks() )
			constructLops(sb);
	}

	public boolean constructLops(StatementBlock sb) 
	{
		boolean ret = false;

		if (sb instanceof WhileStatementBlock)
		{
			WhileStatementBlock wsb = (WhileStatementBlock)sb;
			WhileStatement whileStmt = (WhileStatement)wsb.getStatement(0);
			ArrayList<StatementBlock> body = whileStmt.getBody();

			// step through stmt blocks in while stmt body
			for (StatementBlock stmtBlock : body)
				ret |= constructLops(stmtBlock);

			// handle while stmt predicate
			Lop l = wsb.getPredicateHops().constructLops();
			wsb.setPredicateLops(l);
			ret |= wsb.updatePredicateRecompilationFlag();
		}

		else if (sb instanceof IfStatementBlock)
		{
			IfStatementBlock isb = (IfStatementBlock) sb;
			IfStatement ifStmt = (IfStatement)isb.getStatement(0);
			ArrayList<StatementBlock> ifBody = ifStmt.getIfBody();
			ArrayList<StatementBlock> elseBody = ifStmt.getElseBody();

			// step through stmt blocks in if stmt ifBody
			for (StatementBlock stmtBlock : ifBody)
				ret |= constructLops(stmtBlock);

			// step through stmt blocks in if stmt elseBody
			for (StatementBlock stmtBlock : elseBody)
				ret |= constructLops(stmtBlock);

			// handle if stmt predicate
			Lop l = isb.getPredicateHops().constructLops();
			isb.setPredicateLops(l);
			ret |= isb.updatePredicateRecompilationFlag();
		}

		else if (sb instanceof ForStatementBlock) //NOTE: applies to ForStatementBlock and ParForStatementBlock
		{
			ForStatementBlock fsb =  (ForStatementBlock) sb;
			ForStatement fs = (ForStatement)sb.getStatement(0);
			ArrayList<StatementBlock> body = fs.getBody();

			// step through stmt blocks in FOR stmt body
			for (StatementBlock stmtBlock : body)
				ret |= constructLops(stmtBlock);

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
			ret |= fsb.updatePredicateRecompilationFlags();
		}
		else if (sb instanceof FunctionStatementBlock) {
			FunctionStatementBlock fsb = (FunctionStatementBlock) sb;
			FunctionStatement functStmt = (FunctionStatement)sb.getStatement(0);
			ArrayList<StatementBlock> body = functStmt.getBody();

			// step through stmt blocks in while stmt body
			for( StatementBlock stmtBlock : body )
				ret |= constructLops(stmtBlock);
			if( fsb.isRecompileOnce() )
				fsb.setRecompileOnce(ret);
		}

		// handle default case for regular StatementBlock
		else {
			if (sb.getHops() == null)
				sb.setHops(new ArrayList<Hop>());
			ArrayList<Lop> lops = new ArrayList<>();
			for (Hop hop : sb.getHops())
				lops.add(hop.constructLops());
			sb.setLops(lops);
			ret |= sb.updateRecompilationFlag();
		}

		return ret;
	}

	public Program getRuntimeProgram(DMLProgram prog, DMLConfig config) 
		throws LanguageException, DMLRuntimeException, LopsException, HopsException 
	{
		// constructor resets the set of registered functions
		Program rtprog = new Program(prog);

		// for all namespaces, translate function statement blocks into function program blocks
		for (String namespace : prog.getNamespaces().keySet()){

			for (String fname : prog.getFunctionStatementBlocks(namespace).keySet()){
				// add program block to program
				FunctionStatementBlock fsb = prog.getFunctionStatementBlocks(namespace).get(fname);
				prepareAndAddFunctionProgramBlock(rtprog, config, namespace, fname, fsb, true);
				// add unoptimized block to program (for second-order calls)
				if( prog.getNamespaces().get(namespace).containsFunction(fname, false) ) {
					prepareAndAddFunctionProgramBlock(rtprog, config, namespace, fname,
						prog.getNamespaces().get(namespace).getFunction(fname, false), false);
				}
			}
		}

		// translate all top-level statement blocks to program blocks
		for (StatementBlock sb : prog.getStatementBlocks() ) {
			// add program block to program
			ProgramBlock rtpb = createRuntimeProgramBlock(rtprog, sb, config);
			rtprog.addProgramBlock(rtpb);
		}

		//enhance runtime program by automatic operator fusion
		if( ConfigurationManager.isCodegenEnabled() 
			&& SpoofCompiler.INTEGRATION==IntegrationType.RUNTIME ){
			codgenHopsDAG(rtprog);
		}

		return rtprog ;
	}

	private void prepareAndAddFunctionProgramBlock(Program rtprog, DMLConfig config,
		String fnamespace, String fname, FunctionStatementBlock fsb, boolean opt)
	{
		FunctionProgramBlock rtpb = (FunctionProgramBlock)createRuntimeProgramBlock(rtprog, fsb, config);
		rtprog.addFunctionProgramBlock(fnamespace, fname, rtpb, opt);
		rtpb.setRecompileOnce(fsb.isRecompileOnce());
		rtpb.setNondeterministic(fsb.isNondeterministic());
	}

	public ProgramBlock createRuntimeProgramBlock(Program prog, StatementBlock sb, DMLConfig config) {
		Dag<Lop> dag = null;
		Dag<Lop> pred_dag = null;

		ArrayList<Instruction> instruct;
		ArrayList<Instruction> pred_instruct = null;

		ProgramBlock retPB = null;

		// process While Statement - add runtime program blocks to program
		if (sb instanceof WhileStatementBlock){

			// create DAG for loop predicates
			pred_dag = new Dag<>();
			((WhileStatementBlock) sb).getPredicateLops().addToDag(pred_dag);

			// create instructions for loop predicates
			pred_instruct = new ArrayList<>();
			ArrayList<Instruction> pInst = pred_dag.getJobs(null, config);
			for (Instruction i : pInst ) {
				pred_instruct.add(i);
			}

			// create while program block
			WhileProgramBlock rtpb = new WhileProgramBlock(prog, pred_instruct);

			//// process the body of the while statement block ////

			WhileStatementBlock wsb = (WhileStatementBlock)sb;
			WhileStatement wstmt = (WhileStatement)wsb.getStatement(0);
			for (StatementBlock sblock : wstmt.getBody()){

				// process the body
				ProgramBlock childBlock = createRuntimeProgramBlock(prog, sblock, config);
				rtpb.addProgramBlock(childBlock);
			}

			retPB = rtpb;

			//post processing for generating missing instructions
			retPB.setExitInstruction(deriveExitInstruction(sb));

			// add statement block
			retPB.setStatementBlock(sb);

			// add location information
			retPB.setParseInfo(sb);
		}

		// process If Statement - add runtime program blocks to program
		else if (sb instanceof IfStatementBlock){

			// create DAG for loop predicates
			pred_dag = new Dag<>();
			((IfStatementBlock) sb).getPredicateLops().addToDag(pred_dag);

			// create instructions for loop predicates
			pred_instruct = new ArrayList<>();
			ArrayList<Instruction> pInst = pred_dag.getJobs(null, config);
			for (Instruction i : pInst ) {
				pred_instruct.add(i);
			}

			// create if program block
			IfProgramBlock rtpb = new IfProgramBlock(prog, pred_instruct);

			// process the body of the if statement block
			IfStatementBlock isb = (IfStatementBlock)sb;
			IfStatement istmt = (IfStatement)isb.getStatement(0);

			// process the if body
			for (StatementBlock sblock : istmt.getIfBody()){
				ProgramBlock childBlock = createRuntimeProgramBlock(prog, sblock, config);
				rtpb.addProgramBlockIfBody(childBlock);
			}

			// process the else body
			for (StatementBlock sblock : istmt.getElseBody()){
				ProgramBlock childBlock = createRuntimeProgramBlock(prog, sblock, config);
				rtpb.addProgramBlockElseBody(childBlock); 
			}

			retPB = rtpb;

			//post processing for generating missing instructions
			retPB.setExitInstruction(deriveExitInstruction(sb));

			// add statement block
			retPB.setStatementBlock(sb);

			// add location information
			retPB.setParseInfo(sb);
		}

		// process For Statement - add runtime program blocks to program
		// NOTE: applies to ForStatementBlock and ParForStatementBlock
		else if (sb instanceof ForStatementBlock) 
		{
			ForStatementBlock fsb = (ForStatementBlock) sb;

			// create DAGs for loop predicates 
			Dag<Lop> fromDag = new Dag<>();
			Dag<Lop> toDag = new Dag<>();
			Dag<Lop> incrementDag = new Dag<>();
			if( fsb.getFromHops()!=null )
				fsb.getFromLops().addToDag(fromDag);
			if( fsb.getToHops()!=null )
				fsb.getToLops().addToDag(toDag);
			if( fsb.getIncrementHops()!=null )
				fsb.getIncrementLops().addToDag(incrementDag);

			// create instructions for loop predicates
			ArrayList<Instruction> fromInstructions = fromDag.getJobs(null, config);
			ArrayList<Instruction> toInstructions = toDag.getJobs(null, config);
			ArrayList<Instruction> incrementInstructions = incrementDag.getJobs(null, config);

			// create for program block
			ForProgramBlock rtpb = null;
			IterablePredicate iterPred = fsb.getIterPredicate();

			if( sb instanceof ParForStatementBlock && ConfigurationManager.isParallelParFor() ) {
				rtpb = new ParForProgramBlock(prog, iterPred.getIterVar().getName(),
					iterPred.getParForParams(), ((ParForStatementBlock)sb).getResultVariables());
				ParForProgramBlock pfrtpb = (ParForProgramBlock)rtpb;
				pfrtpb.setStatementBlock(sb); //used for optimization and creating unscoped variables
			}
			else {//ForStatementBlock
				rtpb = new ForProgramBlock(prog, iterPred.getIterVar().getName());
			}

			rtpb.setFromInstructions(fromInstructions);
			rtpb.setToInstructions(toInstructions);
			rtpb.setIncrementInstructions(incrementInstructions);

			// process the body of the for statement block
			ForStatement fs = (ForStatement)fsb.getStatement(0);
			for (StatementBlock sblock : fs.getBody()){
				ProgramBlock childBlock = createRuntimeProgramBlock(prog, sblock, config);
				rtpb.addProgramBlock(childBlock); 
			}

			retPB = rtpb;

			//post processing for generating missing instructions
			retPB.setExitInstruction(deriveExitInstruction(sb));

			// add statement block
			retPB.setStatementBlock(sb);

			// add location information
			retPB.setParseInfo(sb);
		}

		// process function statement block - add runtime program blocks to program
		else if (sb instanceof FunctionStatementBlock){

			FunctionStatementBlock fsb = (FunctionStatementBlock)sb;
			FunctionStatement fstmt = (FunctionStatement)fsb.getStatement(0);
			FunctionProgramBlock rtpb = null;

			// create function program block
			rtpb = new FunctionProgramBlock(prog, fstmt.getInputParams(), fstmt.getOutputParams());

			// process the function statement body
			for (StatementBlock sblock : fstmt.getBody()){
				// process the body
				ProgramBlock childBlock = createRuntimeProgramBlock(prog, sblock, config);
				rtpb.addProgramBlock(childBlock);
			}

			// check there are actually Lops in to process (loop stmt body will not have any)
			if (fsb.getLops() != null && !fsb.getLops().isEmpty()){
				throw new LopsException(fsb.printBlockErrorLocation() + "FunctionStatementBlock should have no Lops");
			}

			retPB = rtpb;

			// add statement block
			retPB.setStatementBlock(sb);

			// add location information
			retPB.setParseInfo(sb);
		}
		else {

			// handle general case
			BasicProgramBlock rtpb = new BasicProgramBlock(prog);

			// DAGs for Lops
			dag = new Dag<>();

			// check there are actually Lops in to process (loop stmt body will not have any)
			if (sb.getLops() != null && !sb.getLops().isEmpty()){

				for (Lop l : sb.getLops()) {
					l.addToDag(dag);
				}

				// Instructions for Lops DAGs
				instruct = dag.getJobs(sb, config);
				rtpb.addInstructions(instruct);
			}

			retPB = rtpb;

			//post processing for generating missing instructions
			//retPB.setExitInstruction(deriveExitInstruction(sb));

			// add statement block
			retPB.setStatementBlock(sb);

			// add location information
			retPB.setParseInfo(sb);
		}

		return retPB;
	}

	public static void refreshMemEstimates(DMLProgram dmlp) {

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

	private static Instruction deriveExitInstruction(StatementBlock sb) {
		Set<String> rmVars = VariableSet.union(
			VariableSet.minus(sb.liveIn(), sb.liveOut()),
			VariableSet.minus(sb.getKill(), sb.liveOut())).getVariableNames();
		return rmVars.isEmpty() ? null :
			VariableCPInstruction.prepareRemoveInstruction(rmVars.toArray(new String[0]));
	}

	public static void refreshMemEstimates(StatementBlock current) {

		MemoTable memo = new MemoTable();

		if( HopRewriteUtils.isLastLevelStatementBlock(current) ) {
			ArrayList<Hop> hopsDAG = current.getHops();
			if (hopsDAG != null && !hopsDAG.isEmpty())
				for( Hop hop : hopsDAG )
					hop.refreshMemEstimates(memo);
		}

		if (current instanceof FunctionStatementBlock) {

			FunctionStatement fstmt = (FunctionStatement)current.getStatement(0);
			for (StatementBlock sb : fstmt.getBody()){
				refreshMemEstimates(sb);
			}
		}
		else if (current instanceof WhileStatementBlock) {
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
		else if (current instanceof IfStatementBlock) {
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
		else if (current instanceof ForStatementBlock) {
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

	public static void resetHopsDAGVisitStatus(DMLProgram dmlp) {

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

	public static void resetHopsDAGVisitStatus(StatementBlock current) {

		if( HopRewriteUtils.isLastLevelStatementBlock(current) ) {
			ArrayList<Hop> hopsDAG = current.getHops();
			if (hopsDAG != null && !hopsDAG.isEmpty() ) {
				Hop.resetVisitStatus(hopsDAG);
			}
		}

		if (current instanceof FunctionStatementBlock) {
			FunctionStatement fstmt = (FunctionStatement)current.getStatement(0);
			for (StatementBlock sb : fstmt.getBody()){
				resetHopsDAGVisitStatus(sb);
			}
		}
		else if (current instanceof WhileStatementBlock) {
			// handle predicate
			WhileStatementBlock wstb = (WhileStatementBlock) current;
			wstb.getPredicateHops().resetVisitStatus();

			WhileStatement ws = (WhileStatement)wstb.getStatement(0);
			for (StatementBlock sb : ws.getBody())
				resetHopsDAGVisitStatus(sb);
		}
		else if (current instanceof IfStatementBlock) {
			// handle predicate
			IfStatementBlock istb = (IfStatementBlock) current;
			istb.getPredicateHops().resetVisitStatus();

			IfStatement is = (IfStatement)istb.getStatement(0);
			for (StatementBlock sb : is.getIfBody())
				resetHopsDAGVisitStatus(sb);
			for (StatementBlock sb : is.getElseBody())
				resetHopsDAGVisitStatus(sb);
		}
		else if (current instanceof ForStatementBlock) {
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

	public void resetLopsDAGVisitStatus(DMLProgram dmlp) {

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

	public void resetLopsDAGVisitStatus(StatementBlock current) {

		ArrayList<Hop> hopsDAG = current.getHops();

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
			wstb.getPredicateLops().resetVisitStatus();
			if (wstb.getNumStatements() > 1)
				LOG.debug("While statement block has more than 1 stmt");
			WhileStatement ws = (WhileStatement)wstb.getStatement(0);

			for (StatementBlock sb : ws.getBody()){
				resetLopsDAGVisitStatus(sb);
			}
		}

		if (current instanceof IfStatementBlock) {
			IfStatementBlock istb = (IfStatementBlock) current;
			istb.getPredicateLops().resetVisitStatus();
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

	public void constructHops(StatementBlock sb) {
		if (sb instanceof WhileStatementBlock) {
			constructHopsForWhileControlBlock((WhileStatementBlock) sb);
			return;
		}

		if (sb instanceof IfStatementBlock) {
			constructHopsForIfControlBlock((IfStatementBlock) sb);
			return;
		}

		if (sb instanceof ForStatementBlock) { //incl ParForStatementBlock
			constructHopsForForControlBlock((ForStatementBlock) sb);
			return;
		}

		if (sb instanceof FunctionStatementBlock) {
			constructHopsForFunctionControlBlock((FunctionStatementBlock) sb);
			return;
		}

		HashMap<String, Hop> ids = new HashMap<>();
		ArrayList<Hop> output = new ArrayList<>();

		VariableSet liveIn  = sb.liveIn();
		VariableSet liveOut = sb.liveOut();
		VariableSet updated = sb._updated;
		VariableSet gen     = sb._gen;
		VariableSet updatedLiveOut = new VariableSet();

		// handle liveout variables that are updated --> target identifiers for Assignment
		HashMap<String, Integer> liveOutToTemp = new HashMap<>();
		for (int i = 0; i < sb.getNumStatements(); i++) {
			Statement current = sb.getStatement(i);

			if (current instanceof AssignmentStatement) {
				AssignmentStatement as = (AssignmentStatement) current;
				DataIdentifier target = as.getTarget();
				if (target != null) {
					if (liveOut.containsVariable(target.getName())) {
						liveOutToTemp.put(target.getName(), Integer.valueOf(i));
					}
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
					DataOp read = new DataOp(var.getName(), var.getDataType(), var.getValueType(), OpOpData.TRANSIENTREAD, null, actualDim1, actualDim2, var.getNnz(), var.getBlocksize());
					read.setParseInfo(var);
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
				Expression fmtExpr = os.getExprParam(DataExpression.FORMAT_TYPE);
				ae.setFileFormat((fmtExpr instanceof StringIdentifier) ?
					Expression.convertFormatType(fmtExpr.toString()) : FileFormat.UNKNOWN);

				if (ae.getDataType() == DataType.SCALAR ) {
					ae.setOutputParams(ae.getDim1(), ae.getDim2(), ae.getNnz(), ae.getUpdateType(), -1);
				}
				else {
					switch(ae.getFileFormat()) {
						case TEXT:
						case MM:
						case CSV:
						case LIBSVM:
						case HDF5:
							// write output in textcell format
							ae.setOutputParams(ae.getDim1(), ae.getDim2(), ae.getNnz(), ae.getUpdateType(), -1);
							break;
						case BINARY:
						case COMPRESSED:
						case UNKNOWN:
							// write output in binary block format
							ae.setOutputParams(ae.getDim1(), ae.getDim2(), ae.getNnz(), ae.getUpdateType(), ae.getBlocksize());
							break;
						case FEDERATED:
							ae.setOutputParams(ae.getDim1(), ae.getDim2(), -1, ae.getUpdateType(), -1);
							break;
						default:
							throw new LanguageException("Unrecognized file format: " + ae.getFileFormat());
					}
				}

				output.add(ae);
			}

			if (current instanceof PrintStatement) {
				DataIdentifier target = createTarget();
				target.setDataType(DataType.SCALAR);
				target.setValueType(ValueType.STRING);
				target.setParseInfo(current);

				PrintStatement ps = (PrintStatement) current;
				PRINTTYPE ptype = ps.getType();

				try {
					if (ptype == PRINTTYPE.PRINT) {
						OpOp1 op = OpOp1.PRINT;
						Expression source = ps.getExpressions().get(0);
						Hop ae = processExpression(source, target, ids);
						Hop printHop = new UnaryOp(target.getName(), target.getDataType(), target.getValueType(), op, ae);
						printHop.setParseInfo(current);
						output.add(printHop);
					}
					else if (ptype == PRINTTYPE.ASSERT) {
						OpOp1 op = OpOp1.ASSERT;
						Expression source = ps.getExpressions().get(0);
						Hop ae = processExpression(source, target, ids);
						Hop printHop = new UnaryOp(target.getName(), target.getDataType(), target.getValueType(), op, ae);
						printHop.setParseInfo(current);
						output.add(printHop);
					}
					else if (ptype == PRINTTYPE.STOP) {
						OpOp1 op = OpOp1.STOP;
						Expression source = ps.getExpressions().get(0);
						Hop ae = processExpression(source, target, ids);
						Hop stopHop = new UnaryOp(target.getName(), target.getDataType(), target.getValueType(), op, ae);
						stopHop.setParseInfo(current);
						output.add(stopHop);
						sb.setSplitDag(true); //avoid merge
					} else if (ptype == PRINTTYPE.PRINTF) {
						List<Expression> expressions = ps.getExpressions();
						Hop[] inHops = new Hop[expressions.size()];
						// process the expressions (function parameters) that
						// make up the printf-styled print statement
						// into Hops so that these can be passed to the printf
						// Hop (ie, MultipleOp) as input Hops
						for (int j = 0; j < expressions.size(); j++) {
							Hop inHop = processExpression(expressions.get(j), target, ids);
							inHops[j] = inHop;
						}
						target.setValueType(ValueType.STRING);
						Hop printfHop = new NaryOp(target.getName(), target.getDataType(),
							target.getValueType(), OpOpN.PRINTF, inHops);
						output.add(printfHop);
					}

				} catch (HopsException e) {
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
						//process right hand side and accumulation
						Hop ae = processExpression(source, target, ids);
						if( as.isAccumulator() ) {
							DataIdentifier accum = getAccumulatorData(liveIn, target.getName());
							ae = HopRewriteUtils.createBinary(ids.get(target.getName()), ae, OpOp2.PLUS);
							target.setProperties(accum.getOutput());
						}
						else
							target.setProperties(source.getOutput());

						if (source instanceof BuiltinFunctionExpression){
							BuiltinFunctionExpression BuiltinSource = (BuiltinFunctionExpression)source;
							if (BuiltinSource.getOpCode() == Builtins.TIME)
								sb.setSplitDag(true);
						}

						ids.put(target.getName(), ae);

						//add transient write if needed
						Integer statementId = liveOutToTemp.get(target.getName());
						if ((statementId != null) && (statementId.intValue() == i)) {
							DataOp transientwrite = new DataOp(target.getName(), target.getDataType(), target.getValueType(), ae, OpOpData.TRANSIENTWRITE, null);
							transientwrite.setOutputParams(ae.getDim1(), ae.getDim2(), ae.getNnz(), ae.getUpdateType(), ae.getBlocksize());
							transientwrite.setParseInfo(target);
							updatedLiveOut.addVariable(target.getName(), target);
							output.add(transientwrite);
						}
					} 
					// CASE: target is indexed identifier (left-hand side indexed expression)
					else {
						Hop ae = processLeftIndexedExpression(source, (IndexedIdentifier)target, ids);

						if( as.isAccumulator() ) {
							DataIdentifier accum = getAccumulatorData(liveIn, target.getName());
							Hop rix = processIndexingExpression((IndexedIdentifier)target, null, ids);
							Hop rhs = processExpression(source, null, ids);
							Hop binary = HopRewriteUtils.createBinary(rix, rhs, OpOp2.PLUS);
							HopRewriteUtils.replaceChildReference(ae, ae.getInput(1), binary);
							target.setProperties(accum.getOutput());
						}

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
							target.setValueType(ValueType.FP64);
							target.setBlocksize(ConfigurationManager.getBlocksize());
						}

						Integer statementId = liveOutToTemp.get(target.getName());
						if ((statementId != null) && (statementId.intValue() == i)) {
							DataOp transientwrite = new DataOp(target.getName(), target.getDataType(), target.getValueType(), ae, OpOpData.TRANSIENTWRITE, null);
							transientwrite.setOutputParams(origDim1, origDim2, ae.getNnz(), ae.getUpdateType(), ae.getBlocksize());
							transientwrite.setParseInfo(target);
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
					if (fsb == null) { 
						throw new LanguageException(source.printErrorLocation() + "function " 
							+ fci.getName() + " is undefined in namespace " + fci.getNamespace());
					}

					FunctionStatement fstmt = (FunctionStatement)fsb.getStatement(0);
					String fkey = DMLProgram.constructFunctionKey(fci.getNamespace(),fci.getName());

					//error handling unsupported function call in indexing expression
					if( target instanceof IndexedIdentifier ) {
						throw new LanguageException("Unsupported function call to '"+fkey+"' in left indexing "
							+ "expression. Please, assign the function output to a variable.");
					}

					//prepare function input names and inputs
					List<String> inputNames = new ArrayList<>(fci.getParamExprs().stream()
						.map(e -> e.getName()).collect(Collectors.toList()));
					List<Hop> finputs = new ArrayList<>(fci.getParamExprs().stream()
						.map(e -> processExpression(e.getExpr(), null, ids)).collect(Collectors.toList()));

					//append default expression for missing arguments
					appendDefaultArguments(fstmt, inputNames, finputs, ids);

					//use function signature to obtain names for unnamed args
					//(note: consistent parameters already checked for functions in general)
					if( inputNames.stream().allMatch(n -> n==null) )
						inputNames = fstmt._inputParams.stream().map(d -> d.getName()).collect(Collectors.toList());

					//create function op
					String[] inputNames2 = inputNames.toArray(new String[0]);
					FunctionType ftype = fsb.getFunctionOpType();
					FunctionOp fcall = (target == null) ?
						new FunctionOp(ftype, fci.getNamespace(), fci.getName(), inputNames2, finputs, new String[]{}, false) :
						new FunctionOp(ftype, fci.getNamespace(), fci.getName(), inputNames2, finputs, new String[]{target.getName()}, false);
					fcall.setParseInfo(fci);
					output.add(fcall);
				}
			}

			else if (current instanceof MultiAssignmentStatement) {
				//multi-assignment, by definition a function call
				MultiAssignmentStatement mas = (MultiAssignmentStatement) current;
				Expression source = mas.getSource();

				if ( source instanceof FunctionCallIdentifier ) {
					FunctionCallIdentifier fci = (FunctionCallIdentifier) source;
					FunctionStatementBlock fsb = this._dmlProg.getFunctionStatementBlock(fci.getNamespace(),fci.getName());
					if (fsb == null){
						throw new LanguageException(source.printErrorLocation() + "function " 
							+ fci.getName() + " is undefined in namespace " + fci.getNamespace());
					}

					FunctionStatement fstmt = (FunctionStatement)fsb.getStatement(0);

					//prepare function input names and inputs
					List<String> inputNames = new ArrayList<>(fci.getParamExprs().stream()
						.map(e -> e.getName()).collect(Collectors.toList()));
					List<Hop> finputs = new ArrayList<>(fci.getParamExprs().stream()
						.map(e -> processExpression(e.getExpr(), null, ids)).collect(Collectors.toList()));

					//use function signature to obtain names for unnamed args
					//(note: consistent parameters already checked for functions in general)
					if( inputNames.stream().allMatch(n -> n==null) )
						inputNames = fstmt._inputParams.stream().map(d -> d.getName()).collect(Collectors.toList());

					//append default expression for missing arguments
					appendDefaultArguments(fstmt, inputNames, finputs, ids);

					//create function op
					String[] foutputs = mas.getTargetList().stream()
						.map(d -> d.getName()).toArray(String[]::new);
					FunctionType ftype = fsb.getFunctionOpType();
					FunctionOp fcall = new FunctionOp(ftype, fci.getNamespace(), fci.getName(),
						inputNames.toArray(new String[0]), finputs, foutputs, false);
					fcall.setParseInfo(fci);
					output.add(fcall);
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
		sb.setHops(output);
	}

	private static DataIdentifier getAccumulatorData(VariableSet liveIn, String varname) {
		DataIdentifier accum = liveIn.getVariable(varname);
		if( accum == null )
			throw new LanguageException("Invalid accumulator assignment "
				+ "to non-existing variable "+varname+".");
		return accum;
	}

	private void appendDefaultArguments(FunctionStatement fstmt, List<String> inputNames, List<Hop> inputs, HashMap<String, Hop> ids) {
		//NOTE: For default expressions of unspecified function arguments, we have two choices:
		//either (a) compile ifelse(exist(argName),default, argName) into the function, or
		//simply (b) add the default to the argument list of function calls when needed.
		//We decided for (b) because it simplifies IPA and dynamic recompilation.

		if( fstmt.getInputParams().size() == inputs.size() )
			return;
		HashSet<String> probeNames = new HashSet<>(inputNames);
		for( DataIdentifier di : fstmt.getInputParams() ) {
			if( probeNames.contains(di.getName()) ) continue;
			Expression exp = fstmt.getInputDefault(di.getName());
			if( exp == null ) {
				throw new LanguageException("Missing default expression for unspecified "
					+ "function argument '"+di.getName()+"' in call to function '"+fstmt.getName()+"'.");
			}
			//compile and add default expression
			inputNames.add(di.getName());
			inputs.add(processExpression(exp, null, ids));
		}
	}

	public void constructHopsForIfControlBlock(IfStatementBlock sb) {
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
	 * @param sb for statement block
	 */
	public void constructHopsForForControlBlock(ForStatementBlock sb)  {
		ForStatement fs = (ForStatement) sb.getStatement(0);
		ArrayList<StatementBlock> body = fs.getBody();
		constructHopsForIterablePredicate(sb);
		for( StatementBlock current : body )
			constructHops(current);
	}

	public void constructHopsForFunctionControlBlock(FunctionStatementBlock fsb) {
		ArrayList<StatementBlock> body = ((FunctionStatement)fsb.getStatement(0)).getBody();
		for( StatementBlock current : body )
			constructHops(current);
	}

	public void constructHopsForWhileControlBlock(WhileStatementBlock sb) {
		ArrayList<StatementBlock> body = ((WhileStatement)sb.getStatement(0)).getBody();
		constructHopsForConditionalPredicate(sb);
		for( StatementBlock current : body )
			constructHops(current);
	}

	public void constructHopsForConditionalPredicate(StatementBlock passedSB) {

		HashMap<String, Hop> _ids = new HashMap<>();

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
				throw new ParseException("variable " + varName + " not live variable for conditional predicate");
			} else {
				long actualDim1 = (var instanceof IndexedIdentifier) ? ((IndexedIdentifier)var).getOrigDim1() : var.getDim1();
				long actualDim2 = (var instanceof IndexedIdentifier) ? ((IndexedIdentifier)var).getOrigDim2() : var.getDim2();

				read = new DataOp(var.getName(), var.getDataType(), var.getValueType(), OpOpData.TRANSIENTREAD,
						null, actualDim1, actualDim2, var.getNnz(), var.getBlocksize());
				read.setParseInfo(var);
			}
			_ids.put(varName, read);
		}

		DataIdentifier target = new DataIdentifier(Expression.getTempName());
		target.setDataType(DataType.SCALAR);
		target.setValueType(ValueType.BOOLEAN);
		target.setParseInfo(passedSB);
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
			if ((predicate instanceof IntIdentifier && ((IntIdentifier) predicate).getValue() == 0)
					|| (predicate instanceof DoubleIdentifier && ((DoubleIdentifier) predicate).getValue() == 0.0)) {
				cp.setPredicate(new BooleanIdentifier(false, predicate));
			} else if ((predicate instanceof IntIdentifier && ((IntIdentifier) predicate).getValue() == 1)
					|| (predicate instanceof DoubleIdentifier && ((DoubleIdentifier) predicate).getValue() == 1.0)) {
				cp.setPredicate(new BooleanIdentifier(true, predicate));
			} else if (predicate instanceof IntIdentifier || predicate instanceof DoubleIdentifier) {
				cp.setPredicate(new BooleanIdentifier(true, predicate));
				LOG.warn(predicate.printWarningLocation() + "Numerical value '" + predicate.toString()
						+ "' (!= 0/1) is converted to boolean TRUE by DML");
			} else if (predicate instanceof StringIdentifier) {
				throw new ParseException(predicate.printErrorLocation() + "String value '" + predicate.toString()
						+ "' is not allowed for iterable predicate");
			}
			predicateHops = processExpression(cp.getPredicate(), null, _ids);
		}

		//create transient write to internal variable name on top of expression
		//in order to ensure proper instruction generation
		predicateHops = HopRewriteUtils.createDataOp(
			ProgramBlock.PRED_VAR, predicateHops, OpOpData.TRANSIENTWRITE);

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
	 * @param fsb for statement block
	 */
	public void constructHopsForIterablePredicate(ForStatementBlock fsb) 
	{
		HashMap<String, Hop> _ids = new HashMap<>();

		// set iterable predicate 
		ForStatement fs = (ForStatement) fsb.getStatement(0);
		IterablePredicate ip = fs.getIterablePredicate();

		for(int i=0; i < 3; i++) {
			Expression expr = (i == 0) ? ip.getFromExpr() : (i == 1) ? ip.getToExpr() :
				( ip.getIncrementExpr() != null ) ? ip.getIncrementExpr() : null;
			VariableSet varsRead = (expr != null) ? expr.variablesRead() : null;

			if(varsRead != null) {
				for (String varName : varsRead.getVariables().keySet()) {

					DataIdentifier var = fsb.liveIn().getVariable(varName);
					DataOp read = null;
					if (var == null) {
						throw new ParseException("variable '" + varName + "' is not available for iterable predicate");
					}
					else {
						long actualDim1 = (var instanceof IndexedIdentifier) ? ((IndexedIdentifier)var).getOrigDim1() : var.getDim1();
						long actualDim2 = (var instanceof IndexedIdentifier) ? ((IndexedIdentifier)var).getOrigDim2() : var.getDim2();
						read = new DataOp(var.getName(), var.getDataType(), var.getValueType(), OpOpData.TRANSIENTREAD,
								null, actualDim1, actualDim2,  var.getNnz(), var.getBlocksize());
						read.setParseInfo(var);
					}
					_ids.put(varName, read);
				}
			}

			//create transient write to internal variable name on top of expression
			//in order to ensure proper instruction generation
			Hop predicateHops = processTempIntExpression(expr, _ids);
			if( predicateHops != null )
				predicateHops = HopRewriteUtils.createDataOp(
					ProgramBlock.PRED_VAR, predicateHops, OpOpData.TRANSIENTWRITE);

			//construct hops for from, to, and increment expressions
			if( i == 0 )
				fsb.setFromHops( predicateHops );
			else if( i == 1 )
				fsb.setToHops( predicateHops );
			else if( ip.getIncrementExpr() != null )
				fsb.setIncrementHops( predicateHops );
		}
	}

	/**
	 * Construct Hops from parse tree : Process Expression in an assignment
	 * statement
	 * 
	 * @param source source expression
	 * @param target data identifier
	 * @param hops map of high-level operators
	 * @return high-level operator
	 */
	private Hop processExpression(Expression source, DataIdentifier target, HashMap<String, Hop> hops) {
		try {
			if( source instanceof BinaryExpression )
				return processBinaryExpression((BinaryExpression) source, target, hops);
			else if( source instanceof RelationalExpression )
				return processRelationalExpression((RelationalExpression) source, target, hops);
			else if( source instanceof BooleanExpression )
				return processBooleanExpression((BooleanExpression) source, target, hops);
			else if( source instanceof BuiltinFunctionExpression )
				return processBuiltinFunctionExpression((BuiltinFunctionExpression) source, target, hops);
			else if( source instanceof ParameterizedBuiltinFunctionExpression )
				return processParameterizedBuiltinFunctionExpression((ParameterizedBuiltinFunctionExpression)source, target, hops);
			else if( source instanceof DataExpression ) {
				Hop ae = processDataExpression((DataExpression)source, target, hops);
				if (ae instanceof DataOp && ((DataOp) ae).getOp() != OpOpData.SQLREAD &&
						((DataOp) ae).getOp() != OpOpData.FEDERATED) {
					Expression expr = ((DataExpression)source).getVarParam(DataExpression.FORMAT_TYPE);
					if( expr instanceof StringIdentifier )
						((DataOp)ae).setFileFormat(Expression.convertFormatType(expr.toString()));
					else
						((DataOp)ae).setFileFormat(FileFormat.UNKNOWN);
				}
				return ae;
			}
			else if (source instanceof IndexedIdentifier)
				return processIndexingExpression((IndexedIdentifier) source,target,hops);
			else if (source instanceof IntIdentifier) {
				IntIdentifier sourceInt = (IntIdentifier) source;
				LiteralOp litop = new LiteralOp(sourceInt.getValue());
				litop.setParseInfo(sourceInt);
				setIdentifierParams(litop, sourceInt);
				return litop;
			} 
			else if (source instanceof DoubleIdentifier) {
				DoubleIdentifier sourceDouble = (DoubleIdentifier) source;
				LiteralOp litop = new LiteralOp(sourceDouble.getValue());
				litop.setParseInfo(sourceDouble);
				setIdentifierParams(litop, sourceDouble);
				return litop;
			}
			else if (source instanceof BooleanIdentifier) {
				BooleanIdentifier sourceBoolean = (BooleanIdentifier) source;
				LiteralOp litop = new LiteralOp(sourceBoolean.getValue());
				litop.setParseInfo(sourceBoolean);
				setIdentifierParams(litop, sourceBoolean);
				return litop;
			} 
			else if (source instanceof StringIdentifier) {
				StringIdentifier sourceString = (StringIdentifier) source;
				LiteralOp litop = new LiteralOp(sourceString.getValue());
				litop.setParseInfo(sourceString);
				setIdentifierParams(litop, sourceString);
				return litop;
			} 
			else if (source instanceof DataIdentifier)
				return hops.get(((DataIdentifier) source).getName());
			else if (source instanceof ExpressionList){
				ExpressionList sourceList = (ExpressionList) source;
				List<Expression> expressions = sourceList.getValue();
				Hop[] listHops = new Hop[expressions.size()];
				int idx = 0;
				for( Expression ex : expressions){
					listHops[idx++] = processExpression(ex, null, hops);
				}
				Hop currBuiltinOp = HopRewriteUtils.createNary(OpOpN.LIST,listHops );
				return currBuiltinOp;
			}
			else{
				throw new ParseException("Unhandled instance of source type: " + source);
			}
		} 
		catch(ParseException e ){
			throw e;
		}
		catch ( Exception e ) {
			throw new ParseException("A Parsing exception occurred", e);
		}
	}

	private static DataIdentifier createTarget(Expression source) {
		Identifier id = source.getOutput();
		if (id instanceof DataIdentifier && !(id instanceof DataExpression))
			return (DataIdentifier) id;
		DataIdentifier target = new DataIdentifier(Expression.getTempName());
		target.setProperties(id);
		return target;
	}

	private static DataIdentifier createTarget() {
		return new DataIdentifier(Expression.getTempName());
	}

	/**
	 * Constructs the Hops for arbitrary expressions that eventually evaluate to an INT scalar. 
	 * 
	 * @param source source expression
	 * @param hops map of high-level operators
	 * @return high-level operatos
	 */
	private Hop processTempIntExpression( Expression source,  HashMap<String, Hop> hops ) {
		if( source == null )
			return null;
		DataIdentifier tmpOut = createTarget();
		tmpOut.setDataType(DataType.SCALAR);
		tmpOut.setValueType(ValueType.INT64);
		source.setOutput(tmpOut);
		return processExpression(source, tmpOut, hops );
	}

	private Hop processLeftIndexedExpression(Expression source, IndexedIdentifier target, HashMap<String, Hop> hops) {
		// process target indexed expressions
		Hop[] ixRange = getIndexingBounds(target, hops, true);

		// process the source expression to get source Hops
		Hop sourceOp = processExpression(source, target, hops);

		// process the target to get targetHops
		Hop targetOp = hops.get(target.getName());
		if (targetOp == null){
			throw new ParseException(target.printErrorLocation() + " must define matrix " + target.getName() + " before indexing operations are allowed ");
		}

		if( sourceOp.getDataType().isMatrix() && source.getOutput().getDataType().isScalar() )
			sourceOp.setDataType(DataType.SCALAR);

		Hop leftIndexOp = new LeftIndexingOp(target.getName(), target.getDataType(),
			ValueType.FP64, targetOp, sourceOp, ixRange[0], ixRange[1], ixRange[2], ixRange[3],
			target.getRowLowerEqualsUpper(), target.getColLowerEqualsUpper());

		setIdentifierParams(leftIndexOp, target);
		leftIndexOp.setParseInfo(target);
		leftIndexOp.setDim1(target.getOrigDim1());
		leftIndexOp.setDim2(target.getOrigDim2());

		return leftIndexOp;
	}

	private Hop processIndexingExpression(IndexedIdentifier source, DataIdentifier target, HashMap<String, Hop> hops) {
		// process Hops for indexes (for source)
		Hop[] ixRange = getIndexingBounds(source, hops, false);

		if (target == null) {
			target = createTarget(source);
		}
		
		//unknown nnz after range indexing (applies to indexing op but also
		//data dependent operations)
		target.setNnz(-1); 

		DataType dt = target.getDataType().isScalar() ? DataType.MATRIX : target.getDataType();
		Hop indexOp = new IndexingOp(target.getName(), dt, target.getValueType(),
			hops.get(source.getName()), ixRange[0], ixRange[1], ixRange[2], ixRange[3],
			source.getRowLowerEqualsUpper(), source.getColLowerEqualsUpper());

		indexOp.setParseInfo(target);
		setIdentifierParams(indexOp, target);

		return indexOp;
	}

	private Hop[] getIndexingBounds(IndexedIdentifier ix, HashMap<String, Hop> hops, boolean lix) {
		Hop rowLowerHops = (ix.getRowLowerBound() != null) ?
			processExpression(ix.getRowLowerBound(),null, hops) : new LiteralOp(1);
		Hop colLowerHops = (ix.getColLowerBound() != null) ?
			processExpression(ix.getColLowerBound(),null, hops) : new LiteralOp(1);

		Hop rowUpperHops = null, colUpperHops = null;
		if (ix.getRowUpperBound() != null)
			rowUpperHops = processExpression(ix.getRowUpperBound(),null,hops);
		else {
			rowUpperHops = ((lix ? ix.getDim1() : ix.getOrigDim1()) != -1) ?
				new LiteralOp(ix.getOrigDim1()) :
				new UnaryOp(ix.getName(), DataType.SCALAR, ValueType.INT64, OpOp1.NROW, hops.get(ix.getName()));
			rowUpperHops.setParseInfo(ix);
		}

		if (ix.getColUpperBound() != null)
			colUpperHops = processExpression(ix.getColUpperBound(),null,hops);
		else {
			colUpperHops = ((lix ? ix.getDim2() : ix.getOrigDim2()) != -1) ?
				new LiteralOp(ix.getOrigDim2()) :
				new UnaryOp(ix.getName(), DataType.SCALAR, ValueType.INT64, OpOp1.NCOL, hops.get(ix.getName()));
			colUpperHops.setParseInfo(ix);
		}

		return new Hop[] {rowLowerHops, rowUpperHops, colLowerHops, colUpperHops};
	}

	/**
	 * Construct Hops from parse tree : Process Binary Expression in an
	 * assignment statement
	 * 
	 * @param source binary expression
	 * @param target data identifier
	 * @param hops map of high-level operators
	 * @return high-level operator
	 */
	private Hop processBinaryExpression(BinaryExpression source, DataIdentifier target, HashMap<String, Hop> hops)
	{
		Hop left  = processExpression(source.getLeft(),  null, hops);
		Hop right = processExpression(source.getRight(), null, hops);

		if (left == null || right == null) {
			throw new ParseException("Missing input in binary expressions (" + source.toString()+"): "
				+ ((left==null)?source.getLeft():source.getRight())+", line="+source.getBeginLine());
		}

		//prepare target identifier and ensure that output type is of inferred type 
		//(type should not be determined by target (e.g., string for print)
		if (target == null) {
			target = createTarget(source);
		}
		target.setValueType(source.getOutput().getValueType());

		Hop currBop = null;
		switch( source.getOpCode() ) {
			case PLUS:
			case MINUS:
			case MULT:
			case DIV:
			case MODULUS:
			case POW:
			case INTDIV:
				currBop = new BinaryOp(target.getName(), target.getDataType(),
					target.getValueType(), OpOp2.valueOf(source.getOpCode().name()), left, right);
				break;
			case MATMULT:
				currBop = new AggBinaryOp(target.getName(), target.getDataType(), target.getValueType(), OpOp2.MULT, org.apache.sysds.common.Types.AggOp.SUM, left, right);
				break;
			default:
				throw new ParseException("Unsupported parsing of binary expression: "+source.getOpCode());
		}

		setIdentifierParams(currBop, source.getOutput());
		currBop.setParseInfo(source);
		return currBop;
	}

	private Hop processRelationalExpression(RelationalExpression source, DataIdentifier target, HashMap<String, Hop> hops) {

		Hop left = processExpression(source.getLeft(), null, hops);
		Hop right = processExpression(source.getRight(), null, hops);

		Hop currBop = null;

		if (target == null) {
			target = createTarget(source);
			if(left.getDataType() == DataType.MATRIX || right.getDataType() == DataType.MATRIX) {
				// Added to support matrix relational comparison
				// (we support only matrices of value type double)
				target.setDataType(DataType.MATRIX);
				target.setValueType(ValueType.FP64);
			}
			else if(left.getDataType() == DataType.FRAME || right.getDataType() == DataType.FRAME) {
				target.setDataType(DataType.FRAME);
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
		currBop.setParseInfo(source);
		return currBop;
	}

	private Hop processBooleanExpression(BooleanExpression source, DataIdentifier target, HashMap<String, Hop> hops)
	{
		// Boolean Not has a single parameter
		boolean constLeft = (source.getLeft().getOutput() instanceof ConstIdentifier);
		boolean constRight = false;
		if (source.getRight() != null) {
			constRight = (source.getRight().getOutput() instanceof ConstIdentifier);
		}

		if (constLeft || constRight) {
			throw new RuntimeException(source.printErrorLocation() + "Boolean expression with constant unsupported");
		}

		Hop left = processExpression(source.getLeft(), null, hops);
		Hop right = null;
		if (source.getRight() != null) {
			right = processExpression(source.getRight(), null, hops);
		}

		//prepare target identifier and ensure that output type is boolean 
		//(type should not be determined by target (e.g., string for print)
		if (target == null)
			target = createTarget(source);
		if( target.getDataType().isScalar() )
			target.setValueType(ValueType.BOOLEAN);

		if (source.getRight() == null) {
			Hop currUop = new UnaryOp(target.getName(), target.getDataType(), target.getValueType(), OpOp1.NOT, left);
			currUop.setParseInfo(source);
			return currUop;
		} 
		else {
			Hop currBop = null;
			OpOp2 op = null;

			if (source.getOpCode() == Expression.BooleanOp.LOGICALAND) {
				op = OpOp2.AND;
			} else if (source.getOpCode() == Expression.BooleanOp.LOGICALOR) {
				op = OpOp2.OR;
			} else {
				throw new RuntimeException(source.printErrorLocation() + "Unknown boolean operation " + source.getOpCode());
			}
			currBop = new BinaryOp(target.getName(), target.getDataType(), target.getValueType(), op, left, right);
			currBop.setParseInfo(source);
			// setIdentifierParams(currBop,source.getOutput());
			return currBop;
		}
	}

	private static Hop constructDfHop(String name, DataType dt, ValueType vt, Builtins op, LinkedHashMap<String,Hop> paramHops) {

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

	private Hop processMultipleReturnParameterizedBuiltinFunctionExpression(ParameterizedBuiltinFunctionExpression source, ArrayList<DataIdentifier> targetList,
			HashMap<String, Hop> hops)
	{
		FunctionType ftype = FunctionType.MULTIRETURN_BUILTIN;
		String nameSpace = DMLProgram.INTERNAL_NAMESPACE;

		// Create an array list to hold the outputs of this lop.
		// Exact list of outputs are added based on opcode.
		ArrayList<Hop> outputs = new ArrayList<>();

		// Construct Hop for current builtin function expression based on its type
		Hop currBuiltinOp = null;
		switch (source.getOpCode()) {
			case TRANSFORMENCODE:
				ArrayList<Hop> inputs = new ArrayList<>();
				inputs.add( processExpression(source.getVarParam("target"), null, hops) );
				inputs.add( processExpression(source.getVarParam("spec"), null, hops) );
				String[] outputNames = new String[targetList.size()]; 
				outputNames[0] = targetList.get(0).getName();
				outputNames[1] = targetList.get(1).getName();
				outputs.add(new DataOp(outputNames[0], DataType.MATRIX, ValueType.FP64, inputs.get(0), OpOpData.FUNCTIONOUTPUT, inputs.get(0).getFilename()));
				outputs.add(new DataOp(outputNames[1], DataType.FRAME, ValueType.STRING, inputs.get(0), OpOpData.FUNCTIONOUTPUT, inputs.get(0).getFilename()));

				currBuiltinOp = new FunctionOp(ftype, nameSpace, source.getOpCode().toString(), null, inputs, outputNames, outputs);
				break;

			default:
				throw new ParseException("Invaid Opcode in DMLTranslator:processMultipleReturnParameterizedBuiltinFunctionExpression(): " + source.getOpCode());
		}

		// set properties for created hops based on outputs of source expression
		for ( int i=0; i < source.getOutputs().length; i++ ) {
			setIdentifierParams( outputs.get(i), source.getOutputs()[i]);
			outputs.get(i).setParseInfo(source);
		}
		currBuiltinOp.setParseInfo(source);

		return currBuiltinOp;
	}

	/**
	 * Construct Hops from parse tree : Process ParameterizedBuiltinFunction Expression in an
	 * assignment statement
	 * 
	 * @param source parameterized built-in function
	 * @param target data identifier
	 * @param hops map of high-level operators
	 * @return high-level operator
	 */
	private Hop processParameterizedBuiltinFunctionExpression(ParameterizedBuiltinFunctionExpression source, DataIdentifier target,
			HashMap<String, Hop> hops) {

		// this expression has multiple "named" parameters
		LinkedHashMap<String, Hop> paramHops = new LinkedHashMap<>();

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
				currBuiltinOp = constructDfHop(target.getName(), target.getDataType(),
					target.getValueType(), source.getOpCode(), paramHops);
				break;
			case CONTAINS:
			case GROUPEDAGG:
			case RMEMPTY:
			case REPLACE:
			case LOWER_TRI:
			case UPPER_TRI:
			case TOKENIZE:
			case TRANSFORMAPPLY:
			case TRANSFORMDECODE:
			case TRANSFORMCOLMAP:
			case TRANSFORMMETA:
			case PARAMSERV:
			case AUTODIFF:
				currBuiltinOp = new ParameterizedBuiltinOp(target.getName(), target.getDataType(),
					target.getValueType(), ParamBuiltinOp.valueOf(source.getOpCode().name()), paramHops);
				break;

			case ORDER:
				ArrayList<Hop> inputs = new ArrayList<>();
				inputs.add(paramHops.get("target"));
				inputs.add(paramHops.get("by"));
				inputs.add(paramHops.get("decreasing"));
				inputs.add(paramHops.get("index.return"));
				currBuiltinOp = new ReorgOp(target.getName(), target.getDataType(), target.getValueType(), ReOrgOp.SORT, inputs);
				break;

			case TOSTRING:
				//check for input data type and only compile toString Hop for matrices/frames,
				//for scalars, we compile (s + "") to ensure consistent string output value types
				currBuiltinOp = !paramHops.get("target").getDataType().isScalar() ?
					new ParameterizedBuiltinOp(target.getName(), target.getDataType(), 
						target.getValueType(), ParamBuiltinOp.TOSTRING, paramHops) :
					HopRewriteUtils.createBinary(paramHops.get("target"), new LiteralOp(""), OpOp2.PLUS);
				break;

			case LISTNV:
				currBuiltinOp = new ParameterizedBuiltinOp(target.getName(), target.getDataType(),
					target.getValueType(), ParamBuiltinOp.LIST, paramHops);
				break;

			case COUNT_DISTINCT:
			case COUNT_DISTINCT_APPROX: {
				Direction dir = Direction.RowCol;  // Default direction
				DataType dataType = DataType.SCALAR;  // Default output data type

				LiteralOp dirOp = (LiteralOp) paramHops.get("dir");
				if (dirOp != null) {
					String dirString = dirOp.getStringValue().toUpperCase();
					if (dirString.equals(Direction.RowCol.toString())) {
						dir = Direction.RowCol;
						dataType = DataType.SCALAR;
					} else if (dirString.equals(Direction.Row.toString())) {
						dir = Direction.Row;
						dataType = DataType.MATRIX;
					} else if (dirString.equals(Direction.Col.toString())) {
						dir = Direction.Col;
						dataType = DataType.MATRIX;
					}
				}

				currBuiltinOp = new AggUnaryOp(target.getName(), dataType, target.getValueType(),
						AggOp.valueOf(source.getOpCode().name()), dir, paramHops.get("data"));
				break;
			}

			case COUNT_DISTINCT_APPROX_ROW:
				currBuiltinOp = new AggUnaryOp(target.getName(), DataType.MATRIX, target.getValueType(),
						AggOp.COUNT_DISTINCT_APPROX, Direction.Row, paramHops.get("data"));
				break;

			case COUNT_DISTINCT_APPROX_COL:
				currBuiltinOp = new AggUnaryOp(target.getName(), DataType.MATRIX, target.getValueType(),
						AggOp.COUNT_DISTINCT_APPROX, Direction.Col, paramHops.get("data"));
				break;

			case UNIQUE:
				Direction dir = Direction.RowCol;
				DataType dataType = DataType.MATRIX;

				LiteralOp dirOp = (LiteralOp) paramHops.get("dir");
				if (dirOp != null) {
					String dirString = dirOp.getStringValue().toUpperCase();
					if (dirString.equals(Direction.RowCol.toString())) {
						dir = Direction.RowCol;
					} else if (dirString.equals(Direction.Row.toString())) {
						dir = Direction.Row;
					} else if (dirString.equals(Direction.Col.toString())) {
						dir = Direction.Col;
					}
				}

				currBuiltinOp = new AggUnaryOp(target.getName(), dataType, target.getValueType(),
						AggOp.valueOf(source.getOpCode().name()), dir, paramHops.get("data"));
				break;

			default:
				throw new ParseException(source.printErrorLocation() + 
					"processParameterizedBuiltinFunctionExpression() -- Unknown operation: " + source.getOpCode());
		}

		setIdentifierParams(currBuiltinOp, source.getOutput());
		currBuiltinOp.setParseInfo(source);
		return currBuiltinOp;
	}

	/**
	 * Construct Hops from parse tree : Process ParameterizedExpression in a
	 * read/write/rand statement
	 * 
	 * @param source data expression
	 * @param target data identifier
	 * @param hops map of high-level operators
	 * @return high-level operator
	 */
	private Hop processDataExpression(DataExpression source, DataIdentifier target,
			HashMap<String, Hop> hops) {

		// this expression has multiple "named" parameters
		HashMap<String, Hop> paramHops = new HashMap<>();

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
			currBuiltinOp = new DataOp(target.getName(), target.getDataType(), target.getValueType(), OpOpData.PERSISTENTREAD, paramHops);
			((DataOp)currBuiltinOp).setFileName(((StringIdentifier)source.getVarParam(DataExpression.IO_FILENAME)).getValue());
			break;

		case WRITE:
			currBuiltinOp = new DataOp(target.getName(), target.getDataType(), target.getValueType(), 
				OpOpData.PERSISTENTWRITE, hops.get(target.getName()), paramHops);
			break;

		case RAND:
			// We limit RAND_MIN, RAND_MAX, RAND_SPARSITY, RAND_SEED, and RAND_PDF to be constants
			OpOpDG method = (paramHops.get(DataExpression.RAND_MIN).getValueType()==ValueType.STRING &&
					target.getDataType() == DataType.MATRIX) ? OpOpDG.SINIT : OpOpDG.RAND;
			currBuiltinOp = new DataGenOp(method, target, paramHops);
			break;

		case FRAME:
			// We limit RAND_MIN, RAND_MAX, RAND_SPARSITY, RAND_SEED, and RAND_PDF to be constants
			method = OpOpDG.FRAMEINIT;
			currBuiltinOp = new DataGenOp(method, target, paramHops);
			break;

		case TENSOR:
		case MATRIX:
			ArrayList<Hop> tmpMatrix = new ArrayList<>();
			tmpMatrix.add( 0, paramHops.get(DataExpression.RAND_DATA) );
			tmpMatrix.add( 1, paramHops.get(DataExpression.RAND_ROWS) );
			tmpMatrix.add( 2, paramHops.get(DataExpression.RAND_COLS) );
			tmpMatrix.add( 3, !paramHops.containsKey(DataExpression.RAND_DIMS) ?
				new LiteralOp("-1") : paramHops.get(DataExpression.RAND_DIMS));
			tmpMatrix.add( 4, paramHops.get(DataExpression.RAND_BY_ROW) );
			currBuiltinOp = new ReorgOp(target.getName(), target.getDataType(),
				target.getValueType(), ReOrgOp.RESHAPE, tmpMatrix);
			break;

		case SQL:
			currBuiltinOp = new DataOp(target.getName(), target.getDataType(),
				target.getValueType(), OpOpData.SQLREAD, paramHops);
			break;

		case FEDERATED:
			currBuiltinOp = new DataOp(target.getName(), target.getDataType(),
					target.getValueType(), OpOpData.FEDERATED, paramHops);
			break;

		default:
			throw new ParseException(source.printErrorLocation() + 
				"processDataExpression():: Unknown operation:  " + source.getOpCode());
		}

		//set identifier meta data (incl dimensions and blocksizes)
		setIdentifierParams(currBuiltinOp, source.getOutput());
		if( source.getOpCode()==DataExpression.DataOp.READ )
			((DataOp)currBuiltinOp).setInputBlocksize(target.getBlocksize());
		else if ( source.getOpCode() == DataExpression.DataOp.WRITE ) {
			if( source.getVarParam(DataExpression.ROWBLOCKCOUNTPARAM) != null )
				currBuiltinOp.setBlocksize(Integer.parseInt(
					source.getVarParam(DataExpression.ROWBLOCKCOUNTPARAM).toString()));
		}
		currBuiltinOp.setParseInfo(source);

		return currBuiltinOp;
	}

	/**
	 * Construct HOps from parse tree: process BuiltinFunction Expressions in 
	 * MultiAssignment Statements. For all other builtin function expressions,
	 * <code>processBuiltinFunctionExpression()</code> is used.
	 * 
	 * @param source built-in function expression
	 * @param targetList list of data identifiers
	 * @param hops map of high-level operators
	 * @return high-level operator
	 */
	private Hop processMultipleReturnBuiltinFunctionExpression(BuiltinFunctionExpression source, ArrayList<DataIdentifier> targetList,
			HashMap<String, Hop> hops) {

		// Construct Hops for all inputs
		ArrayList<Hop> inputs = new ArrayList<>();
		inputs.add( processExpression(source.getFirstExpr(), null, hops) );
		Expression[] expr = source.getAllExpr();
		if(expr != null && expr.length > 1) {
			for(int i = 1; i < expr.length; i++) {
				inputs.add( processExpression(expr[i], null, hops) );
			}
		}

		FunctionType ftype = FunctionType.MULTIRETURN_BUILTIN;
		String nameSpace = DMLProgram.INTERNAL_NAMESPACE;

		// Create an array list to hold the outputs of this lop.
		// Exact list of outputs are added based on opcode.
		ArrayList<Hop> outputs = new ArrayList<>();

		// Construct Hop for current builtin function expression based on its type
		Hop currBuiltinOp = null;
		switch (source.getOpCode()) {
			case QR:
			case LU:
			case EIGEN:
			case FFT:
			case IFFT:
			case FFT_LINEARIZED:
			case IFFT_LINEARIZED:
			case IMG_TRANSFORM_MATRIX:
			case STFT:
			case LSTM:
			case LSTM_BACKWARD:
			case BATCH_NORM2D:
			case BATCH_NORM2D_BACKWARD:
			case REMOVE:
			case SVD:
			case RCM:

				// Number of outputs = size of targetList = #of identifiers in source.getOutputs
				String[] outputNames = new String[targetList.size()]; 
				for ( int i=0; i < targetList.size(); i++ ) {
					outputNames[i] = targetList.get(i).getName();
					Hop output = new DataOp(outputNames[i], DataType.MATRIX, ValueType.FP64, inputs.get(0), OpOpData.FUNCTIONOUTPUT, inputs.get(0).getFilename());
					outputs.add(output);
				}

				// Create the hop for current function call
				currBuiltinOp = new FunctionOp(ftype, nameSpace, source.getOpCode().toString(), null, inputs, outputNames, outputs);
				break;
			case COMPRESS:
				// Number of outputs = size of targetList = #of identifiers in source.getOutputs
				String[] outputNamesCompress = new String[targetList.size()];
				outputNamesCompress[0] = targetList.get(0).getName();
				outputNamesCompress[1] = targetList.get(1).getName();
				outputs.add(new DataOp(outputNamesCompress[0], DataType.MATRIX, ValueType.FP64, inputs.get(0), OpOpData.FUNCTIONOUTPUT, inputs.get(0).getFilename()));
				outputs.add(new DataOp(outputNamesCompress[1], DataType.FRAME, ValueType.STRING, inputs.get(0), OpOpData.FUNCTIONOUTPUT, inputs.get(0).getFilename()));

				// Create the hop for current function call
				currBuiltinOp = new FunctionOp(ftype, nameSpace, source.getOpCode().toString(), null, inputs, outputNamesCompress, outputs);
				break;
			default:
				throw new ParseException("Invaid Opcode in DMLTranslator:processMultipleReturnBuiltinFunctionExpression(): " + source.getOpCode());
		}

		// set properties for created hops based on outputs of source expression
		for ( int i=0; i < source.getOutputs().length; i++ ) {
			setIdentifierParams( outputs.get(i), source.getOutputs()[i]);
			outputs.get(i).setParseInfo(source);
		}
		currBuiltinOp.setParseInfo(source);

		return currBuiltinOp;
	}

	/**
	 * Construct Hops from parse tree : Process BuiltinFunction Expression in an
	 * assignment statement
	 * 
	 * @param source built-in function expression
	 * @param target data identifier
	 * @param hops map of high-level operators
	 * @return high-level operator
	 */
	private Hop processBuiltinFunctionExpression(BuiltinFunctionExpression source, DataIdentifier target,
			HashMap<String, Hop> hops) {
		Hop expr = null;
		if(source.getFirstExpr() != null){
			expr = processExpression(source.getFirstExpr(), null, hops);
		}
		Hop expr2 = null;
		if (source.getSecondExpr() != null) {
			expr2 = processExpression(source.getSecondExpr(), null, hops);
		}
		Hop expr3 = null;
		if (source.getThirdExpr() != null) {
			expr3 = processExpression(source.getThirdExpr(), null, hops);
		}

		Hop currBuiltinOp = null;
		target = (target == null) ? createTarget(source) : target;

		// Construct the hop based on the type of Builtin function
		switch (source.getOpCode()) {

		case EVAL:
		case EVALLIST:
			currBuiltinOp = new NaryOp(target.getName(), target.getDataType(), target.getValueType(),
				OpOpN.EVAL, processAllExpressions(source.getAllExpr(), hops));
			break;

		case COLSUM:
		case COLMAX:
		case COLMIN:
		case COLMEAN:
		case COLPROD:
		case COLVAR:
			currBuiltinOp = new AggUnaryOp(target.getName(), DataType.MATRIX, target.getValueType(),
				AggOp.valueOf(source.getOpCode().name().substring(3)), Direction.Col, expr);
			break;

		case COLSD:
			// colStdDevs = sqrt(colVariances)
			currBuiltinOp = new AggUnaryOp(target.getName(), DataType.MATRIX,
					target.getValueType(), AggOp.VAR, Direction.Col, expr);
			currBuiltinOp = new UnaryOp(target.getName(), DataType.MATRIX,
					target.getValueType(), OpOp1.SQRT, currBuiltinOp);
			break;

		case ROWSUM:
		case ROWMIN:
		case ROWMAX:
		case ROWMEAN:
		case ROWPROD:
		case ROWVAR:
			currBuiltinOp = new AggUnaryOp(target.getName(), DataType.MATRIX, target.getValueType(),
				AggOp.valueOf(source.getOpCode().name().substring(3)), Direction.Row, expr);
			break;

		case ROWINDEXMAX:
			currBuiltinOp = new AggUnaryOp(target.getName(), DataType.MATRIX, target.getValueType(), AggOp.MAXINDEX,
					Direction.Row, expr);
			break;

		case ROWINDEXMIN:
			currBuiltinOp = new AggUnaryOp(target.getName(), DataType.MATRIX, target.getValueType(), AggOp.MININDEX,
					Direction.Row, expr);
			break;

		case ROWSD:
			// rowStdDevs = sqrt(rowVariances)
			currBuiltinOp = new AggUnaryOp(target.getName(), DataType.MATRIX,
					target.getValueType(), AggOp.VAR, Direction.Row, expr);
			currBuiltinOp = new UnaryOp(target.getName(), DataType.MATRIX,
					target.getValueType(), OpOp1.SQRT, currBuiltinOp);
			break;

		case NROW:
			// If the dimensions are available at compile time, then create a LiteralOp (constant propagation)
			// Else create a UnaryOp so that a control program instruction is generated
			currBuiltinOp = (expr.getDim1()==-1) ? new UnaryOp(target.getName(), target.getDataType(),
				target.getValueType(), OpOp1.NROW, expr) : new LiteralOp(expr.getDim1());
			break;
		case NCOL:
			// If the dimensions are available at compile time, then create a LiteralOp (constant propagation)
			// Else create a UnaryOp so that a control program instruction is generated
			currBuiltinOp = (expr.getDim2()==-1) ? new UnaryOp(target.getName(), target.getDataType(),
				target.getValueType(), OpOp1.NCOL, expr) : new LiteralOp(expr.getDim2());
			break;
		case LENGTH:
			// If the dimensions are available at compile time, then create a LiteralOp (constant propagation)
			// Else create a UnaryOp so that a control program instruction is generated
			currBuiltinOp = (expr.getDim1()==-1 || expr.getDim2()==-1) ? new UnaryOp(target.getName(), target.getDataType(),
				target.getValueType(), OpOp1.LENGTH, expr) : new LiteralOp(expr.getDim1()*expr.getDim2());
			break;

		case LINEAGE:
			//construct hop and enable lineage tracing if necessary
			currBuiltinOp = new UnaryOp(target.getName(), target.getDataType(),
				target.getValueType(), OpOp1.LINEAGE, expr);
			DMLScript.LINEAGE = true;
			break;

		case LIST:
			currBuiltinOp = new NaryOp(target.getName(), DataType.LIST, ValueType.UNKNOWN,
				OpOpN.LIST, processAllExpressions(source.getAllExpr(), hops));
			break;

		case EXISTS:
			currBuiltinOp = new UnaryOp(target.getName(), DataType.SCALAR,
				target.getValueType(), OpOp1.EXISTS, expr);
			break;

		case SUM:
		case PROD:
		case VAR:
			currBuiltinOp = new AggUnaryOp(target.getName(), DataType.SCALAR, target.getValueType(),
					AggOp.valueOf(source.getOpCode().name()), Direction.RowCol, expr);
			break;

		case MEAN:
			if ( expr2 == null ) {
				// example: x = mean(Y);
				currBuiltinOp = new AggUnaryOp(target.getName(), DataType.SCALAR, target.getValueType(), AggOp.MEAN,
					Direction.RowCol, expr);
			}
			else {
				// example: x = mean(Y,W);
				// stable weighted mean is implemented by using centralMoment with order = 0
				Hop orderHop = new LiteralOp(0);
				currBuiltinOp=new TernaryOp(target.getName(), DataType.SCALAR,
					target.getValueType(), OpOp3.MOMENT, expr, expr2, orderHop);
			}
			break;

		case SD:
			// stdDev = sqrt(variance)
			currBuiltinOp = new AggUnaryOp(target.getName(), DataType.SCALAR,
				target.getValueType(), AggOp.VAR, Direction.RowCol, expr);
			HopRewriteUtils.setOutputParametersForScalar(currBuiltinOp);
			currBuiltinOp = new UnaryOp(target.getName(), DataType.SCALAR,
				target.getValueType(), OpOp1.SQRT, currBuiltinOp);
			break;

		case MIN:
		case MAX:
			//construct AggUnary for min(X) but BinaryOp for min(X,Y) and NaryOp for min(X,Y,Z)
			currBuiltinOp = (expr2 == null) ? 
				new AggUnaryOp(target.getName(), DataType.SCALAR, target.getValueType(),
					AggOp.valueOf(source.getOpCode().name()), Direction.RowCol, expr) : 
				(source.getAllExpr().length == 2) ?
				new BinaryOp(target.getName(), target.getDataType(), target.getValueType(),
					OpOp2.valueOf(source.getOpCode().name()), expr, expr2) : 
				new NaryOp(target.getName(), target.getDataType(), target.getValueType(),
					OpOpN.valueOf(source.getOpCode().name()), processAllExpressions(source.getAllExpr(), hops));
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
				throw new ParseException(source.printErrorLocation() + "Unknown argument (" + sop + ") for PPRED.");
			}
			currBuiltinOp = new BinaryOp(target.getName(), target.getDataType(), target.getValueType(), operation, expr, expr2);
			break;

		case TRACE:
			currBuiltinOp = new AggUnaryOp(target.getName(), DataType.SCALAR, target.getValueType(), AggOp.TRACE,
					Direction.RowCol, expr);
			break;

		case TRANS:
		case DIAG:
		case REV:
			currBuiltinOp = new ReorgOp(target.getName(), DataType.MATRIX,
				target.getValueType(), ReOrgOp.valueOf(source.getOpCode().name()), expr);
			break;

		case ROLL:
			ArrayList<Hop> inputs = new ArrayList<>();
			inputs.add(expr);
			inputs.add(expr2);
			currBuiltinOp = new ReorgOp(target.getName(), DataType.MATRIX,
					target.getValueType(), ReOrgOp.valueOf(source.getOpCode().name()), inputs);
			break;

		case CBIND:
		case RBIND:
			OpOp2 appendOp2 = (source.getOpCode()==Builtins.CBIND) ? OpOp2.CBIND : OpOp2.RBIND;
			OpOpN appendOpN = (source.getOpCode()==Builtins.CBIND) ? OpOpN.CBIND : OpOpN.RBIND;
			currBuiltinOp = (source.getAllExpr().length == 2) ?
				new BinaryOp(target.getName(), target.getDataType(), target.getValueType(), appendOp2, expr, expr2) :
				new NaryOp(target.getName(), target.getDataType(), target.getValueType(), appendOpN,
					processAllExpressions(source.getAllExpr(), hops));
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
				weightHop.setBlocksize(0);

				if ( numTableArgs == 2 )
					currBuiltinOp = new TernaryOp(target.getName(), target.getDataType(), target.getValueType(), OpOp3.CTABLE, expr, expr2, weightHop);
				else {
					Hop outDim1 = processExpression(source._args[2], null, hops);
					Hop outDim2 = processExpression(source._args[3], null, hops);
					currBuiltinOp = new TernaryOp(target.getName(), target.getDataType(), target.getValueType(),
						OpOp3.CTABLE, expr, expr2, weightHop, outDim1, outDim2, new LiteralOp(true));
				}
				break;

			case 3:
			case 5:
			case 6:
				// example DML statement: F = ctable(A,B,W) or F = ctable(A,B,W,10,15) 
				if (numTableArgs == 3) 
					currBuiltinOp = new TernaryOp(target.getName(), target.getDataType(), target.getValueType(), OpOp3.CTABLE, expr, expr2, expr3);
				else {
					Hop outDim1 = processExpression(source._args[3], null, hops);
					Hop outDim2 = processExpression(source._args[4], null, hops);
					Hop outputEmptyBlocks = numTableArgs == 6 ?
						processExpression(source._args[5], null, hops) : new LiteralOp(true);
					currBuiltinOp = new TernaryOp(target.getName(), target.getDataType(), target.getValueType(),
						OpOp3.CTABLE, expr, expr2, expr3, outDim1, outDim2, outputEmptyBlocks);
				}
				break;

			default: 
				throw new ParseException("Invalid number of arguments "+ numTableArgs + " to table() function.");
			}
			break;

		//data type casts
		case CAST_AS_SCALAR:
			currBuiltinOp = new UnaryOp(target.getName(), DataType.SCALAR, target.getValueType(), OpOp1.CAST_AS_SCALAR, expr);
			break;
		case CAST_AS_MATRIX:
			currBuiltinOp = new UnaryOp(target.getName(), DataType.MATRIX, target.getValueType(), OpOp1.CAST_AS_MATRIX, expr);
			break;
		case CAST_AS_FRAME:
			if(expr2 != null)
				currBuiltinOp = new BinaryOp(target.getName(), DataType.FRAME, target.getValueType(), OpOp2.CAST_AS_FRAME, expr, expr2);
			else
				currBuiltinOp = new UnaryOp(target.getName(), DataType.FRAME, target.getValueType(), OpOp1.CAST_AS_FRAME, expr);
			break;
		case CAST_AS_LIST:
			currBuiltinOp = new UnaryOp(target.getName(), DataType.LIST, target.getValueType(), OpOp1.CAST_AS_LIST, expr);
			break;

		//value type casts
		case CAST_AS_DOUBLE:
			currBuiltinOp = new UnaryOp(target.getName(), target.getDataType(), ValueType.FP64, OpOp1.CAST_AS_DOUBLE, expr);
			break;
		case CAST_AS_INT:
			currBuiltinOp = new UnaryOp(target.getName(), target.getDataType(), ValueType.INT64, OpOp1.CAST_AS_INT, expr);
			break;
		case CAST_AS_BOOLEAN:
			currBuiltinOp = new UnaryOp(target.getName(), target.getDataType(), ValueType.BOOLEAN, OpOp1.CAST_AS_BOOLEAN, expr);
			break;
		case LOCAL:
			currBuiltinOp = new UnaryOp(target.getName(), target.getDataType(), ValueType.FP64, OpOp1.LOCAL, expr);
			break;
		case COMPRESS:
			currBuiltinOp = new UnaryOp(target.getName(), target.getDataType(), ValueType.FP64, OpOp1.COMPRESS, expr);
			break;
		case DECOMPRESS:
			currBuiltinOp = new UnaryOp(target.getName(), target.getDataType(), ValueType.FP64, OpOp1.DECOMPRESS, expr);
			break;

		// Boolean binary
		case XOR:
		case BITWAND:
		case BITWOR:
		case BITWXOR:
		case BITWSHIFTL:
		case BITWSHIFTR:
			currBuiltinOp = new BinaryOp(target.getName(), target.getDataType(),
				target.getValueType(), OpOp2.valueOf(source.getOpCode().name()), expr, expr2);
			break;
		case ABS:
		case SIN:
		case COS:
		case TAN:
		case ASIN:
		case ACOS:
		case ATAN:
		case SINH:
		case COSH:
		case TANH:
		case SIGN:
		case SQRT:
		case EXP:
		case ROUND:
		case CEIL:
		case FLOOR:
		case CUMSUM:
		case CUMPROD:
		case CUMSUMPROD:
		case CUMMIN:
		case CUMMAX:
		case ISNA:
		case ISNAN:
		case ISINF:
			currBuiltinOp = new UnaryOp(target.getName(), target.getDataType(), target.getValueType(),
				OpOp1.valueOf(source.getOpCode().name()), expr);
			break;
		case DROP_INVALID_TYPE:
		case DROP_INVALID_LENGTH:
		case VALUE_SWAP:
		case FRAME_ROW_REPLICATE:
		case APPLY_SCHEMA:
			currBuiltinOp = new BinaryOp(target.getName(), target.getDataType(),
				target.getValueType(), OpOp2.valueOf(source.getOpCode().name()), expr, expr2);
			break;
		case MAP:
			currBuiltinOp = new TernaryOp(target.getName(), target.getDataType(),
				target.getValueType(), OpOp3.valueOf(source.getOpCode().name()),
				expr, expr2, (expr3==null) ? new LiteralOp(0L) : expr3);
			break;

		case LOG:
				if (expr2 == null) {
					OpOp1 mathOp2;
					switch (source.getOpCode()) {
					case LOG:
						mathOp2 = OpOp1.LOG;
						break;
					default:
						throw new ParseException(source.printErrorLocation() +
							"processBuiltinFunctionExpression():: Could not find Operation type for builtin function: "
							+ source.getOpCode());
					}
					currBuiltinOp = new UnaryOp(target.getName(),
						target.getDataType(), target.getValueType(), mathOp2, expr);
				} else {
					OpOp2 mathOp3;
					switch (source.getOpCode()) {
					case LOG:
						mathOp3 = OpOp2.LOG;
						break;
					default:
						throw new ParseException(source.printErrorLocation() +
							"processBuiltinFunctionExpression():: Could not find Operation type for builtin function: "
								+ source.getOpCode());
					}
					currBuiltinOp = new BinaryOp(target.getName(), target.getDataType(), target.getValueType(), mathOp3,
							expr, expr2);
				}
			break;

		case MOMENT:
		case COV:
		case QUANTILE:
		case INTERQUANTILE:
			currBuiltinOp = (expr3 == null) ? new BinaryOp(target.getName(), target.getDataType(), target.getValueType(),
				OpOp2.valueOf(source.getOpCode().name()), expr, expr2) :  new TernaryOp(target.getName(), target.getDataType(),
				target.getValueType(), OpOp3.valueOf(source.getOpCode().name()), expr, expr2,expr3);
			break;

		case IQM:
		case MEDIAN:
			currBuiltinOp = (expr2 == null) ? new UnaryOp(target.getName(), target.getDataType(), target.getValueType(), 
				OpOp1.valueOf(source.getOpCode().name()), expr) : new BinaryOp(target.getName(), target.getDataType(),
				target.getValueType(), OpOp2.valueOf(source.getOpCode().name()), expr, expr2);
			break;

		case IFELSE:
			currBuiltinOp=new TernaryOp(target.getName(), target.getDataType(),
				target.getValueType(), OpOp3.IFELSE, expr, expr2, expr3);
			break;

		case SEQ:
			HashMap<String,Hop> randParams = new HashMap<>();
			randParams.put(Statement.SEQ_FROM, expr);
			randParams.put(Statement.SEQ_TO, expr2);
			randParams.put(Statement.SEQ_INCR, (expr3!=null)?expr3 : new LiteralOp(1)); 
			//note incr: default -1 (for from>to) handled during runtime
			currBuiltinOp = new DataGenOp(OpOpDG.SEQ, target, randParams);
			break;

		case TIME:
			currBuiltinOp = new DataGenOp(OpOpDG.TIME, target);
			break;

		case SAMPLE:
		{
			Expression[] in = source.getAllExpr();

			// arguments: range/size/replace/seed; defaults: replace=FALSE

			HashMap<String,Hop> tmpparams = new HashMap<>();
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
				else if ( expr3.getValueType() == ValueType.INT64 ) 
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

			currBuiltinOp = new DataGenOp(OpOpDG.SAMPLE, target, tmpparams);
			break;
		}

		case SOLVE:
			currBuiltinOp = new BinaryOp(target.getName(), target.getDataType(), target.getValueType(), OpOp2.SOLVE, expr, expr2);
			break;

		case INVERSE:
		case SQRT_MATRIX_JAVA:
		case CHOLESKY:
		case TYPEOF:
		case DETECTSCHEMA:
		case COLNAMES:
			currBuiltinOp = new UnaryOp(target.getName(), target.getDataType(),
				target.getValueType(), OpOp1.valueOf(source.getOpCode().name()), expr);
			break;

		case OUTER:
			if( !(expr3 instanceof LiteralOp) )
				throw new HopsException("Operator for outer builtin function must be a constant: "+expr3);
			OpOp2 op = OpOp2.valueOfByOpcode(((LiteralOp)expr3).getStringValue());
			if( op == null )
				throw new HopsException("Unsupported outer vector binary operation: "+((LiteralOp)expr3).getStringValue());

			currBuiltinOp = new BinaryOp(target.getName(), DataType.MATRIX, target.getValueType(), op, expr, expr2);
			((BinaryOp)currBuiltinOp).setOuterVectorOperation(true); //flag op as specific outer vector operation
			currBuiltinOp.refreshSizeInformation(); //force size reevaluation according to 'outer' flag otherwise danger of incorrect dims
			break;

		case BIASADD:
		case BIASMULT: {
			ArrayList<Hop> inHops1 = new ArrayList<>();
			inHops1.add(expr);
			inHops1.add(expr2);
			currBuiltinOp = new DnnOp(target.getName(), DataType.MATRIX, target.getValueType(),
				OpOpDnn.valueOf(source.getOpCode().name()), inHops1);
			setBlockSizeAndRefreshSizeInfo(expr, currBuiltinOp);
			break;
		}
		case AVG_POOL:
		case MAX_POOL: {
			currBuiltinOp = new DnnOp(target.getName(), DataType.MATRIX, target.getValueType(),
				OpOpDnn.valueOf(source.getOpCode().name()), getALHopsForPoolingForwardIM2COL(expr, source, 1, hops));
			setBlockSizeAndRefreshSizeInfo(expr, currBuiltinOp);
			break;
		}
		case AVG_POOL_BACKWARD:
		case MAX_POOL_BACKWARD: {
			currBuiltinOp = new DnnOp(target.getName(), DataType.MATRIX, target.getValueType(),
				OpOpDnn.valueOf(source.getOpCode().name()), getALHopsForConvOpPoolingCOL2IM(expr, source, 1, hops));
			setBlockSizeAndRefreshSizeInfo(expr, currBuiltinOp);
			break;
		}
		case CONV2D:
		case CONV2D_BACKWARD_FILTER:
		case CONV2D_BACKWARD_DATA: {
			currBuiltinOp = new DnnOp(target.getName(), DataType.MATRIX, target.getValueType(),
				OpOpDnn.valueOf(source.getOpCode().name()), getALHopsForConvOp(expr, source, 1, hops));
			setBlockSizeAndRefreshSizeInfo(expr, currBuiltinOp);
			break;
		}

		case ROW_COUNT_DISTINCT:
			currBuiltinOp = new AggUnaryOp(target.getName(),
				DataType.MATRIX, target.getValueType(), AggOp.COUNT_DISTINCT, Direction.Row, expr);
			break;

		case COL_COUNT_DISTINCT:
			currBuiltinOp = new AggUnaryOp(target.getName(),
				DataType.MATRIX, target.getValueType(), AggOp.COUNT_DISTINCT, Direction.Col, expr);
			break;

		default:
			throw new ParseException("Unsupported builtin function type: "+source.getOpCode());
		}

		boolean isConvolution = source.getOpCode() == Builtins.CONV2D || source.getOpCode() == Builtins.CONV2D_BACKWARD_DATA ||
			source.getOpCode() == Builtins.CONV2D_BACKWARD_FILTER || 
			source.getOpCode() == Builtins.MAX_POOL || source.getOpCode() == Builtins.MAX_POOL_BACKWARD || 
			source.getOpCode() == Builtins.AVG_POOL || source.getOpCode() == Builtins.AVG_POOL_BACKWARD;
		if( !isConvolution) {
			// Since the dimension of output doesnot match that of input variable for these operations
			setIdentifierParams(currBuiltinOp, source.getOutput());
		}
		currBuiltinOp.setParseInfo(source);
		return currBuiltinOp;
	}

	private Hop[] processAllExpressions(Expression[] expr, HashMap<String, Hop> hops) {
		Hop[] ret = new Hop[expr.length];
		for(int i=0; i<expr.length; i++)
			ret[i] = processExpression(expr[i], null, hops);
		return ret;
	}

	private static void setBlockSizeAndRefreshSizeInfo(Hop in, Hop out) {
		out.setBlocksize(in.getBlocksize());
		out.refreshSizeInformation();
		HopRewriteUtils.copyLineNumbers(in, out);
	}

	private ArrayList<Hop> getALHopsForConvOpPoolingCOL2IM(Hop first, BuiltinFunctionExpression source, int skip, HashMap<String, Hop> hops) {
		ArrayList<Hop> ret = new ArrayList<>();
		ret.add(first);
		Expression[] allExpr = source.getAllExpr();

		for(int i = skip; i < allExpr.length; i++) {
			if(i == 11) {
				ret.add(processExpression(allExpr[7], null, hops)); // Make number of channels of images and filter the same
			}
			else
				ret.add(processExpression(allExpr[i], null, hops));
		}
		return ret;
	}

	private ArrayList<Hop> getALHopsForPoolingForwardIM2COL(Hop first, BuiltinFunctionExpression source, int skip, HashMap<String, Hop> hops) {
		ArrayList<Hop> ret = new ArrayList<>();
		ret.add(first);
		Expression[] allExpr = source.getAllExpr();
		if(skip != 1) {
			throw new ParseException("Unsupported skip");
		}

		Expression numChannels = allExpr[6];

		for(int i = skip; i < allExpr.length; i++) {
			if(i == 10) { 
				ret.add(processExpression(numChannels, null, hops));
			}
			else
				ret.add(processExpression(allExpr[i], null, hops));
		}
		return ret;
	}

	@SuppressWarnings("unused") //TODO remove if not used
	private ArrayList<Hop> getALHopsForConvOpPoolingIM2COL(Hop first, BuiltinFunctionExpression source, int skip, HashMap<String, Hop> hops) {
		ArrayList<Hop> ret = new ArrayList<Hop>();
		ret.add(first);
		Expression[] allExpr = source.getAllExpr();
		int numImgIndex = -1;
		if(skip == 1) {
			numImgIndex = 5;
		}
		else if(skip == 2) {
			numImgIndex = 6;
		}
		else {
			throw new ParseException("Unsupported skip");
		}

		for (int i = skip; i < allExpr.length; i++) {
			if (i == numImgIndex) { // skip=1 ==> i==5 and skip=2 => i==6
				Expression numImg = allExpr[numImgIndex];
				Expression numChannels = allExpr[numImgIndex + 1];
				BinaryExpression tmp = new BinaryExpression(org.apache.sysds.parser.Expression.BinaryOp.MULT, numImg);
				tmp.setLeft(numImg);
				tmp.setRight(numChannels);
				ret.add(processTempIntExpression(tmp, hops));
				ret.add(processExpression(new IntIdentifier(1, numImg), null, hops));
				i++;
			} else
				ret.add(processExpression(allExpr[i], null, hops));
		}
		return ret;
	}

	private ArrayList<Hop> getALHopsForConvOp(Hop first, BuiltinFunctionExpression source, int skip, HashMap<String, Hop> hops) {
		ArrayList<Hop> ret = new ArrayList<>();
		ret.add(first);
		Expression[] allExpr = source.getAllExpr();
		for(int i = skip; i < allExpr.length; i++) {
			ret.add(processExpression(allExpr[i], null, hops));
		}
		return ret;
	}

	public void setIdentifierParams(Hop h, Identifier id) {
		if( id.getDim1()>= 0 )
			h.setDim1(id.getDim1());
		if( id.getDim2()>= 0 )
			h.setDim2(id.getDim2());
		if( id.getNnz()>= 0 )
			h.setNnz(id.getNnz());
		h.setBlocksize(id.getBlocksize());
	}

	private boolean prepareReadAfterWrite( DMLProgram prog, HashMap<String, DataIdentifier> pWrites ) {
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
					if (dexpr.isRead()) {
						String pfname = dexpr.getVarParam(DataExpression.IO_FILENAME).toString();
						// found read-after-write
						if (pWrites.containsKey(pfname) && !pfname.trim().isEmpty()) {
							// update read with essential write meta data
							DataIdentifier di = pWrites.get(pfname);
							FileFormat ft = (di.getFileFormat() != null) ? di.getFileFormat() : FileFormat.TEXT;
							dexpr.addVarParam(DataExpression.FORMAT_TYPE, new StringIdentifier(ft.toString(), di));
							if (di.getDim1() >= 0)
								dexpr.addVarParam(DataExpression.READROWPARAM, new IntIdentifier(di.getDim1(), di));
							if (di.getDim2() >= 0)
								dexpr.addVarParam(DataExpression.READCOLPARAM, new IntIdentifier(di.getDim2(), di));
							if (di.getValueType() != ValueType.UNKNOWN)
								dexpr.addVarParam(DataExpression.VALUETYPEPARAM,
										new StringIdentifier(di.getValueType().toExternalString(), di));
							if (di.getDataType() != DataType.UNKNOWN)
								dexpr.addVarParam(DataExpression.DATATYPEPARAM,
										new StringIdentifier(di.getDataType().toString(), di));
							ret = true;
						}
					}
				}
			}
		}

		return ret;
	}
}
