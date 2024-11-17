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

package org.apache.sysds.hops.recompile;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.api.jmlc.JMLCUtils;
import org.apache.sysds.common.Types.DataType;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.common.Types.OpOp1;
import org.apache.sysds.common.Types.OpOpDG;
import org.apache.sysds.common.Types.OpOpData;
import org.apache.sysds.common.Types.ReOrgOp;
import org.apache.sysds.common.Types.ValueType;
import org.apache.sysds.conf.CompilerConfig.ConfigType;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.hops.DataGenOp;
import org.apache.sysds.hops.DataOp;
import org.apache.sysds.hops.FunctionOp;
import org.apache.sysds.hops.FunctionOp.FunctionType;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.HopsException;
import org.apache.sysds.hops.IndexingOp;
import org.apache.sysds.hops.LiteralOp;
import org.apache.sysds.hops.MemoTable;
import org.apache.sysds.hops.MultiThreadedHop;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.hops.UnaryOp;
import org.apache.sysds.hops.codegen.SpoofCompiler;
import org.apache.sysds.hops.rewrite.HopRewriteUtils;
import org.apache.sysds.hops.rewrite.ProgramRewriter;
import org.apache.sysds.lops.Lop;
import org.apache.sysds.lops.compile.Dag;
import org.apache.sysds.lops.rewrite.LopRewriter;
import org.apache.sysds.parser.DMLProgram;
import org.apache.sysds.parser.DataExpression;
import org.apache.sysds.parser.DataIdentifier;
import org.apache.sysds.parser.ForStatementBlock;
import org.apache.sysds.parser.IfStatementBlock;
import org.apache.sysds.parser.ParseInfo;
import org.apache.sysds.parser.Statement;
import org.apache.sysds.parser.StatementBlock;
import org.apache.sysds.parser.WhileStatementBlock;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.BasicProgramBlock;
import org.apache.sysds.runtime.controlprogram.ForProgramBlock;
import org.apache.sysds.runtime.controlprogram.FunctionProgramBlock;
import org.apache.sysds.runtime.controlprogram.IfProgramBlock;
import org.apache.sysds.runtime.controlprogram.LocalVariableMap;
import org.apache.sysds.runtime.controlprogram.ParForProgramBlock;
import org.apache.sysds.runtime.controlprogram.Program;
import org.apache.sysds.runtime.controlprogram.ProgramBlock;
import org.apache.sysds.runtime.controlprogram.WhileProgramBlock;
import org.apache.sysds.runtime.controlprogram.caching.CacheBlock;
import org.apache.sysds.runtime.controlprogram.caching.CacheableData;
import org.apache.sysds.runtime.controlprogram.caching.FrameObject;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.caching.TensorObject;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysds.runtime.controlprogram.parfor.opt.OptTreeConverter;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.instructions.cp.CPOperand;
import org.apache.sysds.runtime.instructions.cp.Data;
import org.apache.sysds.runtime.instructions.cp.EvalNaryCPInstruction;
import org.apache.sysds.runtime.instructions.cp.FunctionCallCPInstruction;
import org.apache.sysds.runtime.instructions.cp.IntObject;
import org.apache.sysds.runtime.instructions.cp.ListObject;
import org.apache.sysds.runtime.instructions.cp.ScalarObject;
import org.apache.sysds.runtime.io.IOUtilFunctions;
import org.apache.sysds.runtime.lineage.LineageItem;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.DataCharacteristics;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.meta.MetaDataAll;
import org.apache.sysds.runtime.meta.MetaDataFormat;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.runtime.util.ProgramConverter;
import org.apache.sysds.runtime.util.UtilFunctions;
import org.apache.sysds.utils.Explain;
import org.apache.sysds.utils.Statistics;
import org.apache.sysds.utils.Explain.ExplainType;

/**
 * Dynamic recompilation of hop dags to runtime instructions, which includes the 
 * following substeps:
 * 
 * (1) deep copy hop dag, (2) refresh matrix characteristics, (3) apply
 * dynamic rewrites, (4) refresh memory estimates, (5) construct lops (incl
 * operator selection), and (6) generate runtime program (incl piggybacking).  
 * 
 * 
 */
public class Recompiler {
	protected static final Log LOG =  LogFactory.getLog(Recompiler.class.getName());

	//Max threshold for in-memory reblock of text input [in bytes]
	//reason: single-threaded text read at 20MB/s, 1GB input -> 50s (should exploit parallelism)
	//note that we scale this threshold up by the degree of available parallelism
	private static final long CP_REBLOCK_THRESHOLD_SIZE = 1L*1024*1024*1024; 
	private static final long CP_CSV_REBLOCK_UNKNOWN_THRESHOLD_SIZE = CP_REBLOCK_THRESHOLD_SIZE;
	
	/** Local reused rewriter for dynamic rewrites during recompile */

	/** Local DML configuration for thread-local config updates */
	private static ThreadLocal<ProgramRewriter> _rewriter = new ThreadLocal<>() {
		@Override protected ProgramRewriter initialValue() { return new ProgramRewriter(false, true); }
	};

	private static ThreadLocal<LopRewriter> _lopRewriter = new ThreadLocal<>() {
		@Override protected LopRewriter initialValue() {return new LopRewriter();}
	};
	
	// additional reused objects to avoid repeated, incremental reallocation on deepCopyDags
	private static ThreadLocal<Map<Long,Hop>> _memoHop = new ThreadLocal<>() {
		@Override protected Map<Long,Hop> initialValue() { return new HashMap<>(); }
		@Override public Map<Long,Hop> get() { var tmp = super.get(); tmp.clear(); return tmp; }
	};
	
	public enum ResetType {
		RESET,
		RESET_KNOWN_DIMS,
		NO_RESET;
		public boolean isReset() {
			return this != NO_RESET;
		}
	}
	
	/**
	 * Re-initializes the recompiler according to the current optimizer flags.
	 */
	public static void reinitRecompiler() {
		_rewriter.set(new ProgramRewriter(false, true));
		_lopRewriter.set(new LopRewriter());
	}
	
	public static ArrayList<Instruction> recompileHopsDag( StatementBlock sb, ArrayList<Hop> hops, 
			ExecutionContext ec, RecompileStatus status, boolean inplace, boolean replaceLit, long tid ) 
	{
		ArrayList<Instruction> newInst = null;

		//need for synchronization as we do temp changes in shared hops/lops
		//however, we create deep copies for most dags to allow for concurrent recompile
		synchronized( hops ) {
			newInst = recompile(sb, hops, ec, status, inplace, replaceLit, true, false, false, null, tid);
		}
		
		// replace thread ids in new instructions
		if( ProgramBlock.isThreadID(tid) ) //only in parfor context
			newInst = ProgramConverter.createShallowCopyInstructionSet(newInst, tid);
		
		// remove writes if called through mlcontext or jmlc 
		if( ec.getVariables().getRegisteredOutputs() != null )
			newInst = JMLCUtils.cleanupRuntimeInstructions(newInst, ec.getVariables().getRegisteredOutputs());
		
		// explain recompiled hops / instructions
		if( DMLScript.EXPLAIN == ExplainType.RECOMPILE_RUNTIME )
			logExplainDAG(sb, hops, newInst);
	
		return newInst;
	}

	public static ArrayList<Instruction> recompileHopsDag( StatementBlock sb, ArrayList<Hop> hops, 
			LocalVariableMap vars, RecompileStatus status, boolean inplace, boolean replaceLit, long tid ) 
	{
		return recompileHopsDag(sb, hops, new ExecutionContext(vars), status, inplace, replaceLit, tid);
	}

	public static ArrayList<Instruction> recompileHopsDag( Hop hop, LocalVariableMap vars, 
			RecompileStatus status, boolean inplace, boolean replaceLit, long tid ) 
	{
		ArrayList<Instruction> newInst = null;

		//need for synchronization as we do temp changes in shared hops/lops
		synchronized( hop ) {
			newInst = recompile(null, new ArrayList<>(Arrays.asList(hop)),
				vars, status, inplace, replaceLit, true, false, true, null, tid);
		}
		
		// replace thread ids in new instructions
		if( ProgramBlock.isThreadID(tid) ) //only in parfor context
			newInst = ProgramConverter.createShallowCopyInstructionSet(newInst, tid);
		
		// explain recompiled instructions
		if( DMLScript.EXPLAIN == ExplainType.RECOMPILE_RUNTIME )
			logExplainPred(hop, newInst);
		
		return newInst;
	}
	
	public static ArrayList<Instruction> recompileHopsDag2Forced( StatementBlock sb, ArrayList<Hop> hops, long tid, ExecType et )
	{
		ArrayList<Instruction> newInst = null;
		
		//need for synchronization as we do temp changes in shared hops/lops
		//however, we create deep copies for most dags to allow for concurrent recompile
		synchronized( hops ) {
			//always in place, no stats update/rewrites, but forced exec type
			newInst = recompile(sb, hops, (LocalVariableMap)null, null, true, false, false, true, false, et, tid);
		}
		
		// replace thread ids in new instructions
		if( ProgramBlock.isThreadID(tid) ) //only in parfor context
			newInst = ProgramConverter.createShallowCopyInstructionSet(newInst, tid);
		
		// explain recompiled hops / instructions
		if( DMLScript.EXPLAIN == ExplainType.RECOMPILE_RUNTIME )
			logExplainDAG(sb, hops, newInst);
		
		return newInst;
	}
	
	public static ArrayList<Instruction> recompileHopsDag2Forced( Hop hop, long tid, ExecType et ) 
	{
		ArrayList<Instruction> newInst = null;

		//need for synchronization as we do temp changes in shared hops/lops
		synchronized( hop ) {
			//always in place, no stats update/rewrites, but forced exec type
			newInst = recompile(null, new ArrayList<>(Arrays.asList(hop)),
				(LocalVariableMap)null, null, true, false, false, true, true, et, tid);
		}

		// replace thread ids in new instructions
		if( ProgramBlock.isThreadID(tid) ) //only in parfor context
			newInst = ProgramConverter.createShallowCopyInstructionSet(newInst, tid);
		
		// explain recompiled hops / instructions
		if( DMLScript.EXPLAIN == ExplainType.RECOMPILE_RUNTIME )
			logExplainPred(hop, newInst);
		
		return newInst;
	}

	public static ArrayList<Instruction> recompileHopsDagInstructions( StatementBlock sb, ArrayList<Hop> hops ) 
	{
		ArrayList<Instruction> newInst = null;

		//need for synchronization as we do temp changes in shared hops/lops
		//however, we create deep copies for most dags to allow for concurrent recompile
		synchronized( hops ) {
			//always in place, no stats update/rewrites
			newInst = recompile(sb, hops, (LocalVariableMap)null, null, true, false, false, false, false, null, 0);
		}
		
		// explain recompiled hops / instructions
		if( DMLScript.EXPLAIN == ExplainType.RECOMPILE_RUNTIME )
			logExplainDAG(sb, hops, newInst);
		
		return newInst;
	}

	public static ArrayList<Instruction> recompileHopsDagInstructions( Hop hop )
	{
		ArrayList<Instruction> newInst = null;

		//need for synchronization as we do temp changes in shared hops/lops
		synchronized( hop ) {
			//always in place, no stats update/rewrites
			newInst = recompile(null, new ArrayList<>(Arrays.asList(hop)),
				(LocalVariableMap)null, null, true, false, false, false, true, null, 0);
		}
		
		// explain recompiled instructions
		if( DMLScript.EXPLAIN == ExplainType.RECOMPILE_RUNTIME )
			logExplainPred(hop, newInst);
		

		return newInst;
	}
	
	/**
	 * Core internal primitive for the dynamic recompilation of any DAGs/predicate,
	 * including all variants with slightly different configurations.
	 * 
	 * @param sb statement block of DAG, null for predicates
	 * @param hops list of DAG root nodes
	 * @param ec Execution context
	 * @param status recompilation status
	 * @param inplace modify DAG in place, otherwise deep copy
	 * @param replaceLit replace literals (only applicable on deep copy)
	 * @param updateStats update statistics, rewrites, and memory estimates
	 * @param forceEt force a given execution type, null for reset
	 * @param pred recompile for predicate DAG
	 * @param et given execution type
	 * @param tid thread id, 0 for main or before worker creation
	 * @return modified list of instructions
	 */
	public static ArrayList<Instruction> recompile(StatementBlock sb, ArrayList<Hop> hops, ExecutionContext ec, RecompileStatus status,
		boolean inplace, boolean replaceLit, boolean updateStats, boolean forceEt, boolean pred, ExecType et, long tid ) 
	{
		boolean codegen = ConfigurationManager.isCodegenEnabled()
			&& !(forceEt && et == null ) //not on reset
			&& SpoofCompiler.RECOMPILE_CODEGEN;
		boolean rewrittenHops = false;
		
		// prepare hops dag for recompile
		if( !inplace ){ 
			// deep copy hop dag (for non-reversable rewrites)
			hops = deepCopyHopsDag(hops);
		}
		else if( !codegen ) {
			// clear existing lops
			Hop.resetVisitStatus(hops);
			for( Hop hopRoot : hops )
				rClearLops( hopRoot );
		}
		
		// get max parallelism constraint, see below
		Hop.resetVisitStatus(hops);
		int maxK = rGetMaxParallelism(hops);
		
		// replace scalar reads with literals 
		if( !inplace && replaceLit ) {
			Hop.resetVisitStatus(hops);
			for( Hop hopRoot : hops )
				rReplaceLiterals( hopRoot, ec, false );
		}
		
		// force exec type (et=null for reset)
		if( forceEt ) {
			Hop.resetVisitStatus(hops);
			for( Hop hopRoot : hops )
				rSetExecType( hopRoot, et );
			Hop.resetVisitStatus(hops);
		}
		
		// update statistics, rewrites, and mem estimates
		if( updateStats ) {
			// refresh matrix characteristics (update stats)
			Hop.resetVisitStatus(hops);
			for( Hop hopRoot : hops )
				rUpdateStatistics( hopRoot, ec.getVariables() );
			
			// dynamic hop rewrites
			if( !inplace ) {
				_rewriter.get().rewriteHopDAG( hops, null );
				
				//update stats after rewrites
				Hop.resetVisitStatus(hops);
				for( Hop hopRoot : hops )
					rUpdateStatistics( hopRoot, ec.getVariables() );
				rewrittenHops = true;
			}
			
			// refresh memory estimates (based on updated stats,
			// before: init memo table with propagated worst-case estimates,
			// after: extract worst-case estimates from memo table 
			Hop.resetVisitStatus(hops);
			MemoTable memo = new MemoTable();
			memo.init(hops, status);
			Hop.resetVisitStatus(hops);
			for( Hop hopRoot : hops )
				hopRoot.refreshMemEstimates(memo);
			memo.extract(hops, status);
		}
		
		// codegen if enabled
		if( codegen ) {
			//create deep copy for in-place
			if( inplace )
				hops = deepCopyHopsDag(hops);
			Hop.resetVisitStatus(hops);
			hops = SpoofCompiler.optimize(hops,
				(status==null || !status.isInitialCodegen()));
		}
		
		// set max parallelism constraint to ensure compilation 
		// incl rewrites does not lose these hop-lop constraints
		Hop.resetVisitStatus(hops);
		rSetMaxParallelism(hops, maxK);
		
		// construct lops
		ArrayList<Lop> lops = new ArrayList<>(hops.size());
		for( Hop hopRoot : hops ){
			lops.add(hopRoot.constructLops());
		}

		// dynamic lop rewrites for the updated hop DAGs
		if (rewrittenHops && sb != null)
			_lopRewriter.get().rewriteLopDAG(sb, lops);

		Dag<Lop> dag = new Dag<>();
		for (Lop l : lops)
			l.addToDag(dag);
		
		// generate runtime instructions (incl piggybacking)
		ArrayList<Instruction> newInst = dag
			.getJobs(sb, ConfigurationManager.getDMLConfig());
		
		// explain recompiled (and potentially deep copied) DAG, but
		// defer the explain of instructions after additional modifications
		if( DMLScript.EXPLAIN == ExplainType.RECOMPILE_HOPS ) {
			if( pred )
				logExplainPred(hops.get(0), newInst);
			else
				logExplainDAG(sb, hops, newInst);
		}
		
		return newInst;
	}

	private static ArrayList<Instruction> recompile(StatementBlock sb, ArrayList<Hop> hops, LocalVariableMap vars, RecompileStatus status,
		boolean inplace, boolean replaceLit, boolean updateStats, boolean forceEt, boolean pred, ExecType et, long tid ) 
	{
		return recompile(sb, hops, new ExecutionContext(vars), status, inplace, replaceLit,
				updateStats, forceEt, pred, et, tid);
	}
	
	private static void logExplainDAG(StatementBlock sb, ArrayList<Hop> hops, ArrayList<Instruction> inst) {
		ParseInfo pi = (sb != null) ? sb : hops.get(0);
		if( DMLScript.EXPLAIN == ExplainType.RECOMPILE_HOPS ) {
			System.out.println("EXPLAIN RECOMPILE \nGENERIC (lines "+pi.getBeginLine()+"-"+pi.getEndLine()+"):\n" +
					Explain.explainHops(hops, 1));
		}
		if( DMLScript.EXPLAIN == ExplainType.RECOMPILE_RUNTIME ) {
			System.out.println("EXPLAIN RECOMPILE \nGENERIC (lines "+pi.getBeginLine()+"-"+pi.getEndLine()+"):\n" +
				Explain.explain(inst, 1));
		}
	}
	
	private static void logExplainPred(Hop hops, ArrayList<Instruction> inst) {
		if( DMLScript.EXPLAIN == ExplainType.RECOMPILE_HOPS )
			System.out.println("EXPLAIN RECOMPILE \nPRED (line "+hops.getBeginLine()+"):\n" + Explain.explain(hops,1));
		if( DMLScript.EXPLAIN == ExplainType.RECOMPILE_RUNTIME )
			System.out.println("EXPLAIN RECOMPILE \nPRED (line "+hops.getBeginLine()+"):\n" + Explain.explain(inst,1));
	}

	public static void recompileProgramBlockHierarchy( List<ProgramBlock> pbs, LocalVariableMap vars, long tid, boolean inplace, ResetType resetRecompile ) {
		//function recompilation via two-phase approach due to challenges 
		//of unclear reconciliation of arbitrary complex control flow
		
		// phase 1: normal inplace=true w/o rewrite as usual, but track requiresRecompile
		// (preserve variables for potential second pass, otherwise corrupted stats)
		RecompileStatus status1 = new RecompileStatus(tid, true, resetRecompile, false);
		synchronized( pbs ) {
			for( ProgramBlock pb : pbs )
				rRecompileProgramBlock(pb, vars, status1);
		
			// phase 2: if called with inplace-false, run a second in-place=false pass in
			// order to apply rewrites (at this point sizes are already propagated, but for
			// correctness we call it with an empty symbol table to avoid invalid size updates)
			if( !status1.requiresRecompile() && !inplace ) {
				RecompileStatus status2 = new RecompileStatus(tid, false, resetRecompile, false);
				for( ProgramBlock pb : pbs )
					rRecompileProgramBlock(pb, new LocalVariableMap(), status2);
			}
		}
	}
	
	/**
	 * Method to recompile program block hierarchy to forced execution time. This affects also
	 * referenced functions and chains of functions. Use et==null in order to release the forced 
	 * exec type.
	 * 
	 * @param pbs list of program blocks
	 * @param tid thread id
	 * @param fnStack function stack
	 * @param et execution type
	 */
	public static void recompileProgramBlockHierarchy2Forced( ArrayList<ProgramBlock> pbs, long tid, Set<String> fnStack, ExecType et ) {
		synchronized( pbs ) {
			for( ProgramBlock pb : pbs )
				rRecompileProgramBlock2Forced(pb, tid, fnStack, et);
		}
	}
	
	/**
	 * This method does NO full program block recompile (no stats update, no rewrites, no recursion) but
	 * only regenerates lops and instructions. The primary use case is recompilation after are hop configuration 
	 * changes which allows to preserve statistics (e.g., propagated worst case stats from other program blocks)
	 * and better performance for recompiling individual program blocks.  
	 * 
	 * @param pb program block
	 * @throws IOException if IOException occurs
	 */
	public static void recompileProgramBlockInstructions(ProgramBlock pb) 
		throws IOException
	{
		if( pb instanceof WhileProgramBlock ) {
			//recompile while predicate instructions
			WhileProgramBlock wpb = (WhileProgramBlock)pb;
			WhileStatementBlock wsb = (WhileStatementBlock) pb.getStatementBlock();
			if( wsb!=null && wsb.getPredicateHops()!=null )
				wpb.setPredicate(recompileHopsDagInstructions(wsb.getPredicateHops()));
		}
		else if( pb instanceof IfProgramBlock ) {
			//recompile if predicate instructions
			IfProgramBlock ipb = (IfProgramBlock)pb;
			IfStatementBlock isb = (IfStatementBlock) pb.getStatementBlock();
			if( isb!=null && isb.getPredicateHops()!=null )
				ipb.setPredicate(recompileHopsDagInstructions(isb.getPredicateHops()));
		}
		else if( pb instanceof ForProgramBlock ) {
			//recompile for/parfor predicate instructions
			ForProgramBlock fpb = (ForProgramBlock)pb;
			ForStatementBlock fsb = (ForStatementBlock) pb.getStatementBlock();
			if( fsb!=null && fsb.getFromHops()!=null )
				fpb.setFromInstructions(recompileHopsDagInstructions(fsb.getFromHops()));
			if( fsb!=null && fsb.getToHops()!=null )
				fpb.setToInstructions(recompileHopsDagInstructions(fsb.getToHops()));
			if( fsb!=null && fsb.getIncrementHops()!=null )
				fpb.setIncrementInstructions(recompileHopsDagInstructions(fsb.getIncrementHops()));
		}
		else if( pb instanceof BasicProgramBlock ) {
			//recompile last-level program block instructions
			BasicProgramBlock bpb = (BasicProgramBlock) pb;
			StatementBlock sb = bpb.getStatementBlock();
			if( sb!=null && sb.getHops()!=null ) {
				bpb.setInstructions(recompileHopsDagInstructions(sb, sb.getHops()));
			}
		}
	}
	
	public static boolean requiresRecompilation( ArrayList<Hop> hops ) {
		if( hops == null )
			return false;
		synchronized( hops ) {
			Hop.resetVisitStatus(hops);
			return hops.stream()
				.anyMatch(h -> rRequiresRecompile(h));
		}
	}
	
	public static boolean requiresRecompilation( Hop hop ) {
		if( hop == null )
			return false;
		synchronized( hop ) {
			hop.resetVisitStatus();
			return rRequiresRecompile(hop);
		}
	}
	

	/**
	 * Deep copy of hops dags for parallel recompilation.
	 * 
	 * @param hops list of high-level operators
	 * @return list of high-level operators
	 */
	public static ArrayList<Hop> deepCopyHopsDag( List<Hop> hops )
	{
		ArrayList<Hop> ret = new ArrayList<>(hops.size());
		
		try {
			//note: need memo table over all independent DAGs in order to 
			//account for shared transient reads (otherwise more instructions generated)
			Map<Long, Hop> memo = _memoHop.get(); //orig ID, new clone
			for( Hop hopRoot : hops )
				ret.add(rDeepCopyHopsDag(hopRoot, memo));
		}
		catch(Exception ex) {
			throw new HopsException(ex);
		}
		
		return ret;
	}
	
	/**
	 * Deep copy of hops dags for parallel recompilation.
	 * 
	 * @param hops high-level operator
	 * @return high-level operator
	 */
	public static Hop deepCopyHopsDag( Hop hops ) {
		Hop ret = null;
		
		try {
			Map<Long, Hop> memo = _memoHop.get(); //orig ID, new clone
			ret = rDeepCopyHopsDag(hops, memo);
		}
		catch(Exception ex) {
			throw new HopsException(ex);
		}
		
		return ret;
	}
	
	private static Hop rDeepCopyHopsDag( Hop hop, Map<Long,Hop> memo ) 
		throws CloneNotSupportedException
	{
		Hop ret = memo.get(hop.getHopID());
	
		//create clone if required 
		if( ret == null ) {
			ret = (Hop) hop.clone();
			
			//create new childs and modify references
			for( Hop in : hop.getInput() ) {
				Hop tmp = rDeepCopyHopsDag(in, memo);
				ret.getInput().add(tmp);
				tmp.getParent().add(ret);
			}
			memo.put(hop.getHopID(), ret);
		}
		
		return ret;
	}
	

	public static void updateFunctionNames(List<Hop> hops, long pid) 
	{
		Hop.resetVisitStatus(hops);
		for( Hop hopRoot : hops  )
			rUpdateFunctionNames( hopRoot, pid );
	}
	
	public static void rUpdateFunctionNames( Hop hop, long pid )
	{
		if( hop.isVisited() )
			return;
		
		//update function names
		if( hop instanceof FunctionOp && ((FunctionOp)hop).getFunctionType() != FunctionType.MULTIRETURN_BUILTIN) {
			FunctionOp fop = (FunctionOp) hop;
			fop.setFunctionName( fop.getFunctionName() + Lop.CP_CHILD_THREAD + pid);
		}
		
		if( hop.getInput() != null )
			for( Hop c : hop.getInput() )
				rUpdateFunctionNames(c, pid);
		
		hop.setVisited();
	}
	
	
	//////////////////////////////
	// private helper functions //
	//////////////////////////////
	
	private static void rRecompileProgramBlock( ProgramBlock pb, LocalVariableMap vars, RecompileStatus status )
	{
		if (pb instanceof WhileProgramBlock) {
			WhileProgramBlock wpb = (WhileProgramBlock)pb;
			WhileStatementBlock wsb = (WhileStatementBlock) wpb.getStatementBlock();
			//recompile predicate
			recompileWhilePredicate(wpb, wsb, vars, status);
			//remove updated scalars because in loop
			removeUpdatedScalars(vars, wsb); 
			//copy vars for later compare
			LocalVariableMap oldVars = (LocalVariableMap) vars.clone();
			RecompileStatus oldStatus = (RecompileStatus) status.clone();
			for (ProgramBlock pb2 : wpb.getChildBlocks())
				rRecompileProgramBlock(pb2, vars, status);
			if( reconcileUpdatedCallVarsLoops(oldVars, vars, wsb) 
				| reconcileUpdatedCallVarsLoops(oldStatus, status, wsb) ) {
				//second pass with unknowns if required
				recompileWhilePredicate(wpb, wsb, vars, status);
				for (ProgramBlock pb2 : wpb.getChildBlocks())
					rRecompileProgramBlock(pb2, vars, status);
			}
			removeUpdatedScalars(vars, wsb);
		}
		else if (pb instanceof IfProgramBlock) {
			IfProgramBlock ipb = (IfProgramBlock)pb;
			IfStatementBlock isb = (IfStatementBlock)ipb.getStatementBlock();
			//recompile predicate
			recompileIfPredicate(ipb, isb, vars, status);
			//copy vars for later compare
			LocalVariableMap oldVars = (LocalVariableMap) vars.clone();
			LocalVariableMap varsElse = (LocalVariableMap) vars.clone();
			RecompileStatus oldStatus = (RecompileStatus)status.clone();
			RecompileStatus statusElse = (RecompileStatus)status.clone();
			for( ProgramBlock pb2 : ipb.getChildBlocksIfBody() )
				rRecompileProgramBlock(pb2, vars, status);
			for( ProgramBlock pb2 : ipb.getChildBlocksElseBody() )
				rRecompileProgramBlock(pb2, varsElse, statusElse);
			reconcileUpdatedCallVarsIf(oldVars, vars, varsElse, isb);
			reconcileUpdatedCallVarsIf(oldStatus, status, statusElse, isb);
			removeUpdatedScalars(vars, ipb.getStatementBlock());
		}
		else if (pb instanceof ForProgramBlock) { //includes ParFORProgramBlock
			ForProgramBlock fpb = (ForProgramBlock)pb;
			ForStatementBlock fsb = (ForStatementBlock) fpb.getStatementBlock();
			//recompile predicates
			recompileForPredicates(fpb, fsb, vars, status);
			//remove updated scalars because in loop
			removeUpdatedScalars(vars, fpb.getStatementBlock());
			//copy vars for later compare
			LocalVariableMap oldVars = (LocalVariableMap) vars.clone();
			RecompileStatus oldStatus = (RecompileStatus) status.clone();
			for( ProgramBlock pb2 : fpb.getChildBlocks() )
				rRecompileProgramBlock(pb2, vars, status);
			if( reconcileUpdatedCallVarsLoops(oldVars, vars, fsb) 
				| reconcileUpdatedCallVarsLoops(oldStatus, status, fsb)) {
				//second pass with unknowns if required
				recompileForPredicates(fpb, fsb, vars, status);
				for( ProgramBlock pb2 : fpb.getChildBlocks() )
					rRecompileProgramBlock(pb2, vars, status);
			}
			removeUpdatedScalars(vars, fpb.getStatementBlock());
		}
		else if (  pb instanceof FunctionProgramBlock ) //includes ExternalFunctionProgramBlock and ExternalFunctionProgramBlockCP
		{
			//do nothing
		}
		else if( pb instanceof BasicProgramBlock ) {
			StatementBlock sb = pb.getStatementBlock();
			BasicProgramBlock bpb = (BasicProgramBlock) pb;
			ArrayList<Instruction> tmp = bpb.getInstructions();
			if( sb == null ) 
				return;
			
			//recompile all for stats propagation and recompile flags
			tmp = Recompiler.recompileHopsDag(
				sb, sb.getHops(), vars, status, status.isInPlace(), false, status.getTID());
			bpb.setInstructions( tmp );
			
			//propagate stats across hops (should be executed on clone of vars)
			if( status.isInPlace() )
				Recompiler.extractDAGOutputStatistics(sb.getHops(), vars);
			
			//reset recompilation flags (w/ special handling functions)
			if( ParForProgramBlock.RESET_RECOMPILATION_FLAGs 
				&& !containsRootFunctionOp(sb.getHops())
				&& status.isReset() )
			{
				Hop.resetRecompilationFlag(sb.getHops(), ExecType.CP, status.getReset());
				sb.updateRecompilationFlag();
			}
			status.trackRecompile(sb.requiresRecompilation());
		}
	}
	
	public static boolean reconcileUpdatedCallVarsLoops( LocalVariableMap oldCallVars, LocalVariableMap callVars, StatementBlock sb )
	{
		boolean requiresRecompile = false;
		
		//handle matrices
		for( String varname : sb.variablesUpdated().getVariableNames() )
		{
			Data dat1 = oldCallVars.get(varname);
			Data dat2 = callVars.get(varname);
			if( dat1!=null && dat1 instanceof MatrixObject && dat2!=null && dat2 instanceof MatrixObject )
			{
				MatrixObject moOld = (MatrixObject) dat1;
				MatrixObject mo = (MatrixObject) dat2;
				DataCharacteristics mcOld = moOld.getDataCharacteristics();
				DataCharacteristics mc = mo.getDataCharacteristics();
				
				if( mcOld.getRows() != mc.getRows() 
					|| mcOld.getCols() != mc.getCols()
					|| mcOld.getNonZeros() != mc.getNonZeros() )
				{
					long ldim1 = mc.getRows(), ldim2 = mc.getCols(), lnnz = mc.getNonZeros();
					//handle row dimension change in body
					if( mcOld.getRows() != mc.getRows() ) {
						ldim1=-1; //unknown
						requiresRecompile = true;
					}
					//handle column dimension change in body
					if( mcOld.getCols() != mc.getCols() ) {
						ldim2=-1; //unknown
						requiresRecompile = true;
					}
					//handle sparsity change
					if( mcOld.getNonZeros() != mc.getNonZeros() ) {
						lnnz=-1; //unknown
						requiresRecompile = true;
					}
					
					MatrixObject moNew = createOutputMatrix(ldim1, ldim2, lnnz);
					callVars.put(varname, moNew);
				}
			}
		}
		
		return requiresRecompile;
	}

	public static boolean reconcileUpdatedCallVarsLoops( RecompileStatus oldCallStatus, RecompileStatus callStatus, StatementBlock sb )
	{
		boolean requiresRecompile = false;
		
		//handle matrices
		for( String varname : sb.variablesUpdated().getVariableNames() )
		{
			DataCharacteristics dat1 = oldCallStatus.getTWriteStats().get(varname);
			DataCharacteristics dat2 = callStatus.getTWriteStats().get(varname);
			if( dat1!=null  && dat2!=null  )
			{
				DataCharacteristics dcOld = dat1;
				DataCharacteristics dc = dat2;
				
				if( dcOld.getRows() != dc.getRows()
					|| dcOld.getCols() != dc.getCols()
					|| dcOld.getNonZeros() != dc.getNonZeros() )
				{
					long ldim1 = dc.getRows(), ldim2 = dc.getCols(), lnnz = dc.getNonZeros();
					//handle row dimension change in body
					if( dcOld.getRows() != dc.getRows() ) {
						ldim1 = -1;
						requiresRecompile = true;
					}
					//handle column dimension change in body
					if( dcOld.getCols() != dc.getCols() ) {
						ldim2 = -1;
						requiresRecompile = true;
					}
					//handle sparsity change
					if( dcOld.getNonZeros() != dc.getNonZeros() ) {
						lnnz = -1;
						requiresRecompile = true;
					}
					
					DataCharacteristics moNew = new MatrixCharacteristics(ldim1, ldim2, -1, lnnz);
					callStatus.getTWriteStats().put(varname, moNew);
				}
			}
		}
		
		return requiresRecompile;
	}
	
	public static LocalVariableMap reconcileUpdatedCallVarsIf( LocalVariableMap oldCallVars, LocalVariableMap callVarsIf, LocalVariableMap callVarsElse, StatementBlock sb )
	{
		for( String varname : sb.variablesUpdated().getVariableNames() )
		{	
			Data origVar = oldCallVars.get(varname);
			Data ifVar = callVarsIf.get(varname);
			Data elseVar = callVarsElse.get(varname);
			Data dat1 = null, dat2 = null;
			
			if( ifVar!=null && elseVar!=null ){ // both branches exists
				dat1 = ifVar;
				dat2 = elseVar;
			}
			else if( ifVar!=null && elseVar==null ){ //only if
				dat1 = origVar;
				dat2 = ifVar;
			}
			else { //only else
				dat1 = origVar;
				dat2 = elseVar;
			}
			
			//compare size and value information (note: by definition both dat1 and dat2 are of same type
			//because we do not allow data type changes)
			if( dat1 != null && dat1 instanceof MatrixObject && dat2!=null )
			{
				//handle matrices
				if( dat1 instanceof MatrixObject && dat2 instanceof MatrixObject )
				{
					MatrixObject moOld = (MatrixObject) dat1;
					MatrixObject mo = (MatrixObject) dat2;
					DataCharacteristics mcOld = moOld.getDataCharacteristics();
					DataCharacteristics mc = mo.getDataCharacteristics();
					
					if( mcOld.getRows() != mc.getRows() 
							|| mcOld.getCols() != mc.getCols()
							|| mcOld.getNonZeros() != mc.getNonZeros() )
					{
						long ldim1 =mc.getRows(), ldim2=mc.getCols(), lnnz=mc.getNonZeros();
						
						//handle row dimension change
						if( mcOld.getRows() != mc.getRows() ) {
							ldim1 = -1; //unknown
						}
						if( mcOld.getCols() != mc.getCols() ) {
							ldim2 = -1; //unknown
						}
						//handle sparsity change
						if( mcOld.getNonZeros() != mc.getNonZeros() ) {
							lnnz = -1; //unknown
						}
						
						MatrixObject moNew = createOutputMatrix(ldim1, ldim2, lnnz);
						callVarsIf.put(varname, moNew);
					}
				}
			}
		}
		
		return callVarsIf;
	}
	
	public static RecompileStatus reconcileUpdatedCallVarsIf( RecompileStatus oldStatus, RecompileStatus callStatusIf, RecompileStatus callStatusElse, StatementBlock sb )
	{
		for( String varname : sb.variablesUpdated().getVariableNames() )
		{	
			DataCharacteristics origVar = oldStatus.getTWriteStats().get(varname);
			DataCharacteristics ifVar = callStatusIf.getTWriteStats().get(varname);
			DataCharacteristics elseVar = callStatusElse.getTWriteStats().get(varname);
			DataCharacteristics dat1 = null, dat2 = null;
			
			if( ifVar!=null && elseVar!=null ){ // both branches exists
				dat1 = ifVar;
				dat2 = elseVar;
			}
			else if( ifVar!=null && elseVar==null ){ //only if
				dat1 = origVar;
				dat2 = ifVar;
			}
			else { //only else
				dat1 = origVar;
				dat2 = elseVar;
			}
			
			//compare size and value information (note: by definition both dat1 and dat2 are of same type
			//because we do not allow data type changes)
			if( dat1 != null && dat2!=null )
			{
				DataCharacteristics dcOld = dat1;
				DataCharacteristics dc = dat2;
					
				if( dcOld.getRows() != dc.getRows()
						|| dcOld.getCols() != dc.getCols()
						|| dcOld.getNonZeros() != dc.getNonZeros() )
				{
					long ldim1 = (dcOld.getRows()>=0 && dc.getRows()>=0) ?
							Math.max( dcOld.getRows(), dc.getRows() ) : -1;
					long ldim2 = (dcOld.getCols()>=0 && dc.getCols()>=0) ?
							Math.max( dcOld.getCols(), dc.getCols() ) : -1;
					long lnnz = (dcOld.getNonZeros()>=0 && dc.getNonZeros()>=0) ?
							Math.max( dcOld.getNonZeros(), dc.getNonZeros() ) : -1;
					
					DataCharacteristics mcNew = new MatrixCharacteristics(ldim1, ldim2, -1, lnnz);
					callStatusIf.getTWriteStats().put(varname, mcNew);
				}
			}
		}
		
		return callStatusIf;
	}
	
	private static boolean containsRootFunctionOp( ArrayList<Hop> hops )
	{
		boolean ret = false;
		for( Hop h : hops )
			if( h instanceof FunctionOp )
				ret |= true;
		
		return ret;
	}
	
	private static MatrixObject createOutputMatrix(long dim1, long dim2, long nnz) {
		MatrixObject moOut = new MatrixObject(ValueType.FP64, null);
		int blksz = ConfigurationManager.getBlocksize();
		DataCharacteristics mc = new MatrixCharacteristics(dim1, dim2, blksz, nnz);
		MetaDataFormat meta = new MetaDataFormat(mc,null);
		moOut.setMetaData(meta);
		return moOut;
	}
	
	
	//helper functions for predicate recompile
	
	private static void recompileIfPredicate( IfProgramBlock ipb, IfStatementBlock isb, LocalVariableMap vars, RecompileStatus status ) {
		if( isb == null || isb.getPredicateHops() == null )
			return;
		Hop hops = isb.getPredicateHops();
		ArrayList<Instruction> tmp = recompileHopsDag(
			hops, vars, status, status.isInPlace(), false, status.getTID());
		ipb.setPredicate( tmp );
		if( ParForProgramBlock.RESET_RECOMPILATION_FLAGs && status.isReset() ) {
			Hop.resetRecompilationFlag(hops, ExecType.CP, status.getReset());
			isb.updatePredicateRecompilationFlag();
		}
		status.trackRecompile(isb.requiresPredicateRecompilation());
	}
	
	private static void recompileWhilePredicate( WhileProgramBlock wpb, WhileStatementBlock wsb, LocalVariableMap vars, RecompileStatus status ) {
		if( wsb == null || wsb.getPredicateHops() == null )
			return;
		Hop hops = wsb.getPredicateHops();
		ArrayList<Instruction> tmp = recompileHopsDag(
			hops, vars, status, status.isInPlace(), false, status.getTID());
		wpb.setPredicate( tmp );
		if( ParForProgramBlock.RESET_RECOMPILATION_FLAGs && status.isReset() ) {
			Hop.resetRecompilationFlag(hops, ExecType.CP, status.getReset());
			wsb.updatePredicateRecompilationFlag();
		}
		status.trackRecompile(wsb.requiresPredicateRecompilation());
	}
	
	private static void recompileForPredicates( ForProgramBlock fpb, ForStatementBlock fsb, LocalVariableMap vars, RecompileStatus status ) {
		if( fsb == null )
			return;
		
		Hop fromHops = fsb.getFromHops();
		Hop toHops = fsb.getToHops();
		Hop incrHops = fsb.getIncrementHops();
		
		// recompile predicates
		if( fromHops != null ) {
			ArrayList<Instruction> tmp = recompileHopsDag(
				fromHops, vars, status, status.isInPlace(), false, status.getTID());
			fpb.setFromInstructions(tmp);
		}
		if( toHops != null ) {
			ArrayList<Instruction> tmp = recompileHopsDag(
				toHops, vars, status, status.isInPlace(), false, status.getTID());
			fpb.setToInstructions(tmp);
		}
		if( incrHops != null ) {
			ArrayList<Instruction> tmp = recompileHopsDag(
				incrHops, vars, status, status.isInPlace(), false, status.getTID());
			fpb.setIncrementInstructions(tmp);
		}
		
		//handle recompilation flags
		if( ParForProgramBlock.RESET_RECOMPILATION_FLAGs && status.isReset() ) {
			if( fromHops != null )
				Hop.resetRecompilationFlag(fromHops, ExecType.CP, status.getReset());
			if( toHops != null )
				Hop.resetRecompilationFlag(toHops, ExecType.CP, status.getReset());
			if( incrHops != null )
				Hop.resetRecompilationFlag(incrHops, ExecType.CP, status.getReset());
			fsb.updatePredicateRecompilationFlags();
		}
		status.trackRecompile(fsb.requiresPredicateRecompilation());
	}
	
	public static void rRecompileProgramBlock2Forced( ProgramBlock pb, long tid, Set<String> fnStack, ExecType et ) {
		if (pb instanceof WhileProgramBlock)
		{
			WhileProgramBlock pbTmp = (WhileProgramBlock)pb;
			WhileStatementBlock sbTmp = (WhileStatementBlock)pbTmp.getStatementBlock();
			//recompile predicate
			if( sbTmp!=null && !(et==ExecType.CP && !OptTreeConverter.containsSparkInstruction(pbTmp.getPredicate(), true)) )
				pbTmp.setPredicate( Recompiler.recompileHopsDag2Forced(sbTmp.getPredicateHops(), tid, et) );
			
			//recompile body
			for (ProgramBlock pb2 : pbTmp.getChildBlocks())
				rRecompileProgramBlock2Forced(pb2, tid, fnStack, et);
		}
		else if (pb instanceof IfProgramBlock)
		{
			IfProgramBlock pbTmp = (IfProgramBlock)pb;
			IfStatementBlock sbTmp = (IfStatementBlock)pbTmp.getStatementBlock();
			//recompile predicate
			if( sbTmp!=null &&!(et==ExecType.CP && !OptTreeConverter.containsSparkInstruction(pbTmp.getPredicate(), true)) )
				pbTmp.setPredicate( Recompiler.recompileHopsDag2Forced(sbTmp.getPredicateHops(), tid, et) );
			//recompile body
			for( ProgramBlock pb2 : pbTmp.getChildBlocksIfBody() )
				rRecompileProgramBlock2Forced(pb2, tid, fnStack, et);
			for( ProgramBlock pb2 : pbTmp.getChildBlocksElseBody() )
				rRecompileProgramBlock2Forced(pb2, tid, fnStack, et);
		}
		else if (pb instanceof ForProgramBlock) { //includes ParFORProgramBlock
			ForProgramBlock pbTmp = (ForProgramBlock)pb;
			ForStatementBlock sbTmp = (ForStatementBlock) pbTmp.getStatementBlock();
			//recompile predicate
			if( sbTmp!=null && sbTmp.getFromHops() != null && !(et==ExecType.CP && !OptTreeConverter.containsSparkInstruction(pbTmp.getFromInstructions(), true)) )
				pbTmp.setFromInstructions( Recompiler.recompileHopsDag2Forced(sbTmp.getFromHops(), tid, et) );
			if( sbTmp!=null && sbTmp.getToHops() != null && !(et==ExecType.CP && !OptTreeConverter.containsSparkInstruction(pbTmp.getToInstructions(), true)) )
				pbTmp.setToInstructions( Recompiler.recompileHopsDag2Forced(sbTmp.getToHops(), tid, et) );
			if( sbTmp!=null && sbTmp.getIncrementHops() != null && !(et==ExecType.CP && !OptTreeConverter.containsSparkInstruction(pbTmp.getIncrementInstructions(), true)) )
				pbTmp.setIncrementInstructions( Recompiler.recompileHopsDag2Forced(sbTmp.getIncrementHops(), tid, et) );
			//recompile body
			for( ProgramBlock pb2 : pbTmp.getChildBlocks() )
				rRecompileProgramBlock2Forced(pb2, tid, fnStack, et);
		}		
		else if (  pb instanceof FunctionProgramBlock ) {
			FunctionProgramBlock tmp = (FunctionProgramBlock)pb;
			for( ProgramBlock pb2 : tmp.getChildBlocks() )
				rRecompileProgramBlock2Forced(pb2, tid, fnStack, et);
		}
		else if( pb instanceof BasicProgramBlock )
		{
			BasicProgramBlock bpb = (BasicProgramBlock) pb;
			StatementBlock sb = bpb.getStatementBlock();
			
			//recompile hops dag to CP (note selective recompile 'if CP and no MR inst' 
			//would be invalid with permutation matrix mult across multiple dags)
			if( sb != null ) {
				ArrayList<Instruction> tmp = bpb.getInstructions();
				tmp = Recompiler.recompileHopsDag2Forced(sb, sb.getHops(), tid, et);
				bpb.setInstructions( tmp );
			}
			
			//recompile functions
			if( OptTreeConverter.containsFunctionCallInstruction(bpb) )
			{
				ArrayList<Instruction> tmp = bpb.getInstructions();
				for( Instruction inst : tmp ) {
					if( inst instanceof FunctionCallCPInstruction ) {
						FunctionCallCPInstruction func = (FunctionCallCPInstruction)inst;
						String fname = func.getFunctionName();
						String fnamespace = func.getNamespace();
						rRecompileProgramBlock2Forced(fnamespace, fname, pb.getProgram(), tid, fnStack, et);
					}
					else if( inst instanceof EvalNaryCPInstruction ) {
						CPOperand fname = ((EvalNaryCPInstruction)inst).getInputs()[0];
						if( fname.isLiteral() ) {
							rRecompileProgramBlock2Forced(DMLProgram.DEFAULT_NAMESPACE,
								fname.getName(), pb.getProgram(), tid, fnStack, et);
						}
						else {
							for( String fkey : pb.getProgram().getFunctionProgramBlocks().keySet() )
								if(!fkey.startsWith(DMLProgram.BUILTIN_NAMESPACE)) {
									String[] parts = DMLProgram.splitFunctionKey(fkey);
									rRecompileProgramBlock2Forced(
										parts[0], parts[1], pb.getProgram(), tid, fnStack, et);
								}
						}
					}
				}
			}
		}
	}

	private static void rRecompileProgramBlock2Forced(String fnamespace, String fname,
		Program prog, long tid, Set<String> fnStack, ExecType et)
	{
		String fKey = DMLProgram.constructFunctionKey(fnamespace, fname);
		if( !fnStack.contains(fKey) ) { //memoization for multiple calls, recursion
			fnStack.add(fKey);
			FunctionProgramBlock fpb = prog.getFunctionProgramBlock(fnamespace, fname, true);
			rRecompileProgramBlock2Forced(fpb, tid, fnStack, et); //recompile chains of functions
			if( prog.containsFunctionProgramBlock(fnamespace, fname, false) ) {
				FunctionProgramBlock fpb2 = prog.getFunctionProgramBlock(fnamespace, fname, false);
				rRecompileProgramBlock2Forced(fpb2, tid, fnStack, et); //recompile chains of functions
			}
		}
	}
	
	/**
	 * Remove any scalar variables from the variable map if the variable
	 * is updated in this block.
	 *
	 * @param callVars  Map of variables eligible for propagation.
	 * @param sb  DML statement block.
	 */
	public static void removeUpdatedScalars( LocalVariableMap callVars, StatementBlock sb ) {
		if( sb != null ) {
			//remove updated scalar variables from constants
			for( Entry<String, DataIdentifier> v : sb.variablesUpdated().getVariables().entrySet() )
				if( v.getValue().getDataType().isScalar() )
					callVars.remove(v.getKey());
		}
	}
	
	public static void extractDAGOutputStatistics(List<Hop> hops, LocalVariableMap vars) {
		extractDAGOutputStatistics(hops, vars, true);
	}
	
	public static void extractDAGOutputStatistics(List<Hop> hops, LocalVariableMap vars, boolean overwrite) {
		for( Hop hop : hops ) //for all hop roots
			extractDAGOutputStatistics(hop, vars, overwrite);
	}

	public static void extractDAGOutputStatistics(Hop hop, LocalVariableMap vars, boolean overwrite)
	{
		if( hop instanceof DataOp && ((DataOp)hop).getOp()==OpOpData.TRANSIENTWRITE ) //for all writes to symbol table
		{
			String varName = hop.getName();
			if( !vars.keySet().contains(varName) || overwrite ) //not existing so far
			{
				//extract matrix sizes for size propagation
				if( !hop.getDataType().isScalar()) //matrix/frame/list/tensor
				{
					MatrixObject mo = new MatrixObject(ValueType.FP64, null);
					DataCharacteristics mc = new MatrixCharacteristics(hop.getDim1(),
						hop.getDim2(), ConfigurationManager.getBlocksize(), hop.getNnz());
					MetaDataFormat meta = new MetaDataFormat(mc,null);
					mo.setMetaData(meta);	
					vars.put(varName, mo);
				}
				//extract scalar constants for second constant propagation
				else if( hop.getDataType()==DataType.SCALAR )
				{
					//extract literal assignments
					if( hop.getInput().size()==1 && hop.getInput().get(0) instanceof LiteralOp )
					{
						ScalarObject constant = HopRewriteUtils.getScalarObject((LiteralOp)hop.getInput().get(0));
						if( constant!=null )
							vars.put(varName, constant);
					}
					//extract constant variable assignments
					else if( hop.getInput().size()==1 && hop.getInput().get(0) instanceof DataOp)
					{
						DataOp dop = (DataOp) hop.getInput().get(0);
						String dopvarname = dop.getName();
						if( dop.isRead() && vars.keySet().contains(dopvarname) )
						{
							ScalarObject constant = (ScalarObject) vars.get(dopvarname);
							vars.put(varName, constant); //no clone because constant
						}
					}
					//extract ncol/nrow variable assignments
					else if( hop.getInput().size()==1 && hop.getInput().get(0) instanceof UnaryOp
							&& (((UnaryOp)hop.getInput().get(0)).getOp()==OpOp1.NROW ||
							    ((UnaryOp)hop.getInput().get(0)).getOp()==OpOp1.NCOL)   )
					{
						UnaryOp uop = (UnaryOp) hop.getInput().get(0);
						if( uop.getOp()==OpOp1.NROW && uop.getInput().get(0).getDim1()>0 )
							vars.put(varName, new IntObject(uop.getInput().get(0).getDim1()));
						else if( uop.getOp()==OpOp1.NCOL && uop.getInput().get(0).getDim2()>0 )
							vars.put(varName, new IntObject(uop.getInput().get(0).getDim2()));
					}
					//remove other updated scalars
					else
					{
						//we need to remove other updated scalars in order to ensure result
						//correctness of recompilation w/o being too conservative
						vars.remove(varName);
					}
				}
			}
			else //already existing: take largest   
			{
				Data dat = vars.get(varName);
				if( dat instanceof MatrixObject )
				{
					MatrixObject mo = (MatrixObject)dat;
					DataCharacteristics mc = mo.getDataCharacteristics();
					if( OptimizerUtils.estimateSizeExactSparsity(mc.getRows(), mc.getCols(), (mc.getNonZeros()>=0)?((double)mc.getNonZeros())/mc.getRows()/mc.getCols():1.0)	
					    < OptimizerUtils.estimateSize(hop.getDim1(), hop.getDim2()) )
					{
						//update statistics if necessary
						mc.setDimension(hop.getDim1(), hop.getDim2());
						mc.setNonZeros(hop.getNnz());
					}
				}
				else //scalar (just overwrite)
				{
					if( hop.getInput().size()==1 && hop.getInput().get(0) instanceof LiteralOp )
					{
						ScalarObject constant = HopRewriteUtils.getScalarObject((LiteralOp)hop.getInput().get(0));
						if( constant!=null )
							vars.put(varName, constant);
					}
				}
				
			}
		}
	}
	
	/**
	 * NOTE: no need for update visit status due to early abort
	 * 
	 * @param hop high-level operator
	 * @return true if requires recompile, false otherwise
	 */
	private static boolean rRequiresRecompile( Hop hop )
	{	
		boolean ret = hop.requiresRecompile();
		if( hop.isVisited() )
			return ret;
		
		if( hop.getInput() != null )
			for( Hop c : hop.getInput() )
			{
				ret |= rRequiresRecompile(c);
				if( ret ) break; // early abort
			}
		
		hop.setVisited();
		
		return ret;
	}
	
	/**
	 * Clearing lops for a given hops includes to (1) remove the reference
	 * to constructed lops and (2) clear the exec type (for consistency). 
	 * 
	 * The latter is important for advanced optimizers like parfor; otherwise subtle
	 * side-effects of program recompilation and hop-lop rewrites possible
	 * (e.g., see indexingop hop-lop rewrite in combination parfor rewrite set
	 * exec type that eventuelly might lead to unnecessary remote_parfor jobs).
	 * 
	 * @param hop high-level operator
	 */
	public static void rClearLops( Hop hop )
	{
		if( hop.isVisited() )
			return;
		
		//clear all relevant lops to allow for recompilation
		if( hop instanceof LiteralOp )
		{
			//for literal ops, we just clear parents because always constant
			if( hop.getLops() != null )
				hop.getLops().getOutputs().clear();	
		}
		else //GENERAL CASE
		{
			hop.resetExecType(); //remove exec type
			hop.setLops(null); //clear lops
			if( hop.getInput() != null )
				for( Hop c : hop.getInput() )
					rClearLops(c);
		}
		
		hop.setVisited();
	}
	
	public static void rUpdateStatistics( Hop hop, LocalVariableMap vars ) 
	{
		if( hop.isVisited() )
			return;

		//recursively process children
		if( hop.getInput() != null )
			for( Hop c : hop.getInput() )
				rUpdateStatistics(c, vars);
		
		//update statistics for transient reads according to current statistics
		//(with awareness not to override persistent reads to an existing name)
		if( HopRewriteUtils.isData(hop, OpOpData.TRANSIENTREAD) ) {
			DataOp d = (DataOp) hop;
			String varName = d.getName();
			if( vars.keySet().contains( varName ) ) {
				Data dat = vars.get(varName);
				if( dat instanceof MatrixObject ) {
					MatrixObject mo = (MatrixObject) dat;
					d.setDim1(mo.getNumRows());
					d.setDim2(mo.getNumColumns());
					d.setNnz(mo.getNnz());
					if(mo.isCompressed()){
						d.setCompressedSize(mo.getCompressedSize());
						d.setCompressedOutput(true);	
					}
				}
				else if( dat instanceof FrameObject ) {
					FrameObject fo = (FrameObject) dat;
					d.setDim1(fo.getNumRows());
					d.setDim2(fo.getNumColumns());
				}
				else if( dat instanceof ListObject ) {
					ListObject lo = (ListObject) dat;
					d.setDim1(lo.getLength());
					d.setDim2(1);
				}
				else if( dat instanceof TensorObject) {
					TensorObject to = (TensorObject) dat;
					// TODO: correct dimensions
					d.setDim1(to.getNumRows());
					d.setDim2(to.getNumColumns());
					d.setNnz(to.getNnz());
				}
				if( dat instanceof CacheableData<?> ) {
					CacheableData<?> cd = (CacheableData<?>) dat;
					d.setOnlyRDD(!cd.isCached(true) &&cd.getRDDHandle()!=null);
				}
			}
		}
		//special case for persistent reads with unknown size (read-after-write)
		else if( HopRewriteUtils.isData(hop, OpOpData.PERSISTENTREAD)
			&& !hop.dimsKnown() && ((DataOp)hop).getFileFormat()!=FileFormat.CSV
			&& !ConfigurationManager.getCompilerConfigFlag(ConfigType.IGNORE_READ_WRITE_METADATA) )
		{
			//update hop with read meta data
			DataOp dop = (DataOp) hop; 
			tryReadMetaDataFileDataCharacteristics(dop);
		}
		//update size expression for rand/seq according to symbol table entries
		else if ( hop instanceof DataGenOp )
		{
			DataGenOp d = (DataGenOp) hop;
			Map<String,Integer> params = d.getParamIndexMap();
			if (   d.getOp() == OpOpDG.RAND || d.getOp()==OpOpDG.SINIT 
				|| d.getOp() == OpOpDG.SAMPLE || d.getOp() == OpOpDG.FRAMEINIT ) 
			{
				boolean initUnknown = !d.dimsKnown();
				// TODO refresh tensor size information
				if (params.containsKey(DataExpression.RAND_ROWS) && params.containsKey(DataExpression.RAND_COLS)) {
					int ix1 = params.get(DataExpression.RAND_ROWS);
					int ix2 = params.get(DataExpression.RAND_COLS);
					//update rows/cols by evaluating simple expression of literals, nrow, ncol, scalars, binaryops
					Map<Long, Long> memo = new HashMap<>();
					d.refreshRowsParameterInformation(d.getInput().get(ix1), vars, memo);
					d.refreshColsParameterInformation(d.getInput().get(ix2), vars, memo);
					if( !(initUnknown & d.dimsKnown()) )
						d.refreshSizeInformation();
				}
			} 
			else if ( d.getOp() == OpOpDG.SEQ ) 
			{
				boolean initUnknown = !d.dimsKnown();
				int ix1 = params.get(Statement.SEQ_FROM);
				int ix2 = params.get(Statement.SEQ_TO);
				int ix3 = params.get(Statement.SEQ_INCR);
				Map<Long, Double> memo = new HashMap<>();
				double from = Hop.computeBoundsInformation(d.getInput().get(ix1), vars, memo);
				double to = Hop.computeBoundsInformation(d.getInput().get(ix2), vars, memo);
				double incr = Hop.computeBoundsInformation(d.getInput().get(ix3), vars, memo);
				
				//special case increment 
				if ( from!=Double.MAX_VALUE && to!=Double.MAX_VALUE ) {
					incr *= ((from > to && incr > 0) || (from < to && incr < 0)) ? -1.0 : 1.0;
				}
				
				if ( from!=Double.MAX_VALUE && to!=Double.MAX_VALUE && incr!=Double.MAX_VALUE ) {
					d.setDim1( UtilFunctions.getSeqLength(from, to, incr) );
					d.setDim2( 1 );
					d.setIncrementValue( incr );
				}
				if( !(initUnknown & d.dimsKnown()) )
					d.refreshSizeInformation();
			}
			else if (d.getOp() == OpOpDG.TIME) {
				d.refreshSizeInformation();
			}
			else {
				throw new DMLRuntimeException("Unexpected data generation method: " + d.getOp());
			}
		}
		//update size expression for reshape according to symbol table entries
		else if( HopRewriteUtils.isReorg(hop, ReOrgOp.RESHAPE) ) {
			if (hop.getDataType() != DataType.TENSOR) {
				hop.refreshSizeInformation(); //update incl reset
				if (!hop.dimsKnown()) {
					Map<Long, Long> memo = new HashMap<>();
					hop.refreshRowsParameterInformation(hop.getInput().get(1), vars, memo);
					hop.refreshColsParameterInformation(hop.getInput().get(2), vars, memo);
				}
			} else {
				//TODO tensor rewrite
			}
		}
		//update size expression for indexing according to symbol table entries
		else if( hop instanceof IndexingOp ) {
			hop.refreshSizeInformation(); //update, incl reset
			if( !hop.dimsKnown() ) {
				if( hop.getDataType().isList() 
					&& hop.getInput().get(1).getValueType() == ValueType.STRING ) {
					hop.setDim1(1);
					hop.setDim2(1);
				}
				else {
					Map<Long, Double> memo = new HashMap<>();
					double rl = Hop.computeBoundsInformation(hop.getInput().get(1), vars, memo);
					double ru = Hop.computeBoundsInformation(hop.getInput().get(2), vars, memo);
					double cl = Hop.computeBoundsInformation(hop.getInput().get(3), vars, memo);
					double cu = Hop.computeBoundsInformation(hop.getInput().get(4), vars, memo);
					if( rl!=Double.MAX_VALUE && ru!=Double.MAX_VALUE )
						hop.setDim1( (long)(ru-rl+1) );
					if( cl!=Double.MAX_VALUE && cu!=Double.MAX_VALUE )
						hop.setDim2( (long)(cu-cl+1) );
				}
			}
		}
		else if(HopRewriteUtils.isUnary(hop, OpOp1.CAST_AS_MATRIX)
			&& hop.getInput(0) instanceof IndexingOp && hop.getInput(0).getDataType().isList()
			&& HopRewriteUtils.isData(hop.getInput(0).getInput(0), OpOpData.TRANSIENTREAD) ) {
			Data ldat = vars.get(hop.getInput(0).getInput(0).getName()); //list, or matrix during IPA
			Hop rix = hop.getInput(0);
			if( ldat != null && ldat instanceof ListObject
				&& rix.getInput(1) instanceof LiteralOp
				&& rix.getInput(2) instanceof LiteralOp
				&& HopRewriteUtils.isEqualValue(rix.getInput(1), rix.getInput(2))) {
				ListObject list = (ListObject) ldat;
				MatrixObject mo = (MatrixObject) ((rix.getInput(1).getValueType() == ValueType.STRING) ? 
					list.getData(((LiteralOp)rix.getInput(1)).getStringValue()) :
					list.getData((int)HopRewriteUtils.getIntValueSafe(rix.getInput(1))-1));
				hop.setDim1(mo.getNumRows());
				hop.setDim2(mo.getNumColumns());
			}
		}
		else {
		//propagate statistics along inner nodes of DAG,
		//without overwriting inferred size expressions
			hop.refreshSizeInformation();
		}
		
		hop.setVisited();
	}

	/**
	 * public interface to package local literal replacement
	 * 
	 * @param hop high-level operator
	 * @param ec Execution context
	 * @param scalarsOnly if true, replace only scalar variables but no matrix operations;
	 *            if false, apply full literal replacement
	 */
	public static void rReplaceLiterals( Hop hop, ExecutionContext ec, boolean scalarsOnly ) {
		LiteralReplacement.rReplaceLiterals(hop, ec, scalarsOnly);
	}

	public static void rReplaceLiterals( Hop hop, LocalVariableMap vars, boolean scalarsOnly ) {
		LiteralReplacement.rReplaceLiterals(hop, new ExecutionContext(vars), scalarsOnly);
	}
	
	public static void rSetExecType( Hop hop, ExecType etype ) {
		if( hop.isVisited() )
			return;
		//update function names
		hop.setForcedExecType(etype);
		if( hop.getInput() != null )
			for( Hop c : hop.getInput() )
				rSetExecType(c, etype);
		hop.setVisited();
	}
	
	public static int rGetMaxParallelism(List<Hop> hops) {
		int ret = -1;
		for( Hop c : hops )
			ret = Math.max(ret, rGetMaxParallelism(c));
		return ret;
	}
	
	public static int rGetMaxParallelism(Hop hop) {
		if( hop.isVisited() )
			return -1;
		//recursively process children and
		int ret = rGetMaxParallelism(hop.getInput());
		//obtain max num thread constraints
		if( hop instanceof MultiThreadedHop )
			ret = Math.max(ret, ((MultiThreadedHop)hop).getMaxNumThreads());
		hop.setVisited();
		return ret;
	}
	
	public static void rSetMaxParallelism(List<Hop> hops, int k) {
		for( Hop c : hops )
			rSetMaxParallelism(c, k);
	}
	
	public static void rSetMaxParallelism(Hop hop, int k) {
		if( hop.isVisited() )
			return;
		//recursively process children
		rSetMaxParallelism(hop.getInput(), k);
		//set max num thread constraint
		if( hop instanceof MultiThreadedHop )
			((MultiThreadedHop)hop).setMaxNumThreads(k);
		hop.setVisited();
	}
	
	public static void recompileFunctionOnceIfNeeded(boolean recompileOnce,
		List<ProgramBlock> childBlocks, long tid, boolean inplace, ResetType reset, ExecutionContext ec)
	{
		try {
			if( ConfigurationManager.isDynamicRecompilation() 
				&& recompileOnce 
				&& ParForProgramBlock.RESET_RECOMPILATION_FLAGs )
			{
				long t0 = DMLScript.STATISTICS ? System.nanoTime() : 0;
				
				//note: it is important to reset the recompilation flags here
				// (1) it is safe to reset recompilation flags because a 'recompile_once'
				//     function will be recompiled for every execution.
				// (2) without reset, there would be no benefit in recompiling the entire function
				LocalVariableMap tmp = (LocalVariableMap) ec.getVariables().clone();
				Recompiler.recompileProgramBlockHierarchy(childBlocks, tmp, tid, inplace, reset);

				if( DMLScript.STATISTICS ){
					long t1 = System.nanoTime();
					Statistics.incrementFunRecompileTime(t1-t0);
					Statistics.incrementFunRecompiles();
				}
			}
		}
		catch(Exception ex) {
			throw new DMLRuntimeException("Error recompiling function body.", ex);
		}
	}

	/**
	 * CP Reblock check for spark instructions; in contrast to MR, we can not
	 * rely on the input file sizes because inputs might be passed via rdds. 
	 * 
	 * @param ec execution context
	 * @param varin variable
	 * @return true if CP reblock?
	 */
	public static boolean checkCPReblock(ExecutionContext ec, String varin) 
	{
		CacheableData<?> obj = ec.getCacheableData(varin);
		DataCharacteristics mc = ec.getDataCharacteristics(varin);
		
		long rows = mc.getRows();
		long cols = mc.getCols();
		long nnz = mc.getNonZeros();
		
		//check valid cp reblock recompilation hook
		if(    !ConfigurationManager.isDynamicRecompilation()
			|| !OptimizerUtils.isHybridExecutionMode() )
		{
			return false;
		}

		//robustness for usage through mlcontext (key/values of input rdds are 
		//not serializable for text; also bufferpool rdd read only supported for binary)
		MetaDataFormat iimd = (MetaDataFormat) obj.getMetaData();
		if( obj.getRDDHandle() != null && iimd.getFileFormat() != FileFormat.BINARY )
			return false;
		
		//robustness unknown dimensions, e.g., for csv reblock
		if( rows <= 0 || cols <= 0 ) {
			try {
				long size = HDFSTool.getFilesizeOnHDFS(new Path(obj.getFileName()));
				return (size < OptimizerUtils.getLocalMemBudget() &&
					size < CP_CSV_REBLOCK_UNKNOWN_THRESHOLD_SIZE * 
					OptimizerUtils.getParallelTextReadParallelism());
			} 
			catch(IllegalArgumentException | IOException ex) {
				throw new DMLRuntimeException(ex);
			}
		}

		//check valid dimensions and memory requirements
		double sp = OptimizerUtils.getSparsity(rows, cols, nnz);
		double mem = MatrixBlock.estimateSizeInMemory(rows, cols, sp);
		
		if(    !OptimizerUtils.isValidCPDimensions(rows, cols)
			|| !OptimizerUtils.isValidCPMatrixSize(rows, cols, sp)
			|| mem >= OptimizerUtils.getLocalMemBudget() ) 
		{
			return false;
		}
		
		//check in-memory reblock size threshold (preference: distributed)
		long estFilesize = (long)(3.5 * mem); //conservative estimate
		long cpThreshold = CP_REBLOCK_THRESHOLD_SIZE * 
			OptimizerUtils.getParallelTextReadParallelism();
		boolean ret = (iimd.getFileFormat() == FileFormat.BINARY
			|| estFilesize < cpThreshold); //for text conservative
		
		// for reading ultra-sparse in local mode (1 executor) always avoid spark reblock
		if( !ret && sp < MatrixBlock.ULTRA_SPARSITY_TURN_POINT ) // but qualifies
			ret = (SparkExecutionContext.getNumExecutors() == 1);
		
		return ret;
	}
	
	public static boolean checkCPCheckpoint(DataCharacteristics dc) {
		return OptimizerUtils.isHybridExecutionMode()
			&& OptimizerUtils.isValidCPDimensions(dc.getRows(), dc.getCols())
			&& !OptimizerUtils.exceedsCachingThreshold(dc.getCols(), OptimizerUtils.estimateSize(dc));
	}

	public static void executeInMemoryReblock(ExecutionContext ec, String varin, String varout) {
		executeInMemoryReblock(ec, varin, varout, null);
	}
	
	@SuppressWarnings("unchecked")
	public static void executeInMemoryReblock(ExecutionContext ec, String varin, String varout, LineageItem litem) {
		CacheableData<CacheBlock<?>> in = (CacheableData<CacheBlock<?>>) ec.getCacheableData(varin);
		CacheableData<CacheBlock<?>> out = (CacheableData<CacheBlock<?>>) ec.getCacheableData(varout);

		if( in.isFederated() ) {
			out.setMetaData(in.getMetaData());
			out.setFedMapping(in.getFedMapping());
		}
		else {
			//read text input matrix (through buffer pool, matrix object carries all relevant
			//information including additional arguments for csv reblock)
			CacheBlock<?> mb = in.acquireRead();
			
			//set output (incl update matrix characteristics)
			out.acquireModify(mb);
			out.setCacheLineage(litem);
			out.release();
			in.release();
		}
	}
	
	private static void tryReadMetaDataFileDataCharacteristics( DataOp dop )
	{
		try
		{
			//get meta data filename
			String mtdname = DataExpression.getMTDFileName(dop.getFileName());
			Path path = new Path(mtdname);
			FileSystem fs = IOUtilFunctions.getFileSystem(mtdname); //no auto-close
			if( fs.exists(path) ){
				try(BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(path)))) {
					MetaDataAll mtd = new MetaDataAll(br);
					DataType dt = mtd.getDataType();
					dop.setDataType(dt);
					if(dt != DataType.FRAME)
						dop.setValueType(mtd.getValueType());
					dop.setDim1((dt==DataType.MATRIX||dt==DataType.FRAME)? mtd.getDim1():0);
					dop.setDim2((dt==DataType.MATRIX||dt==DataType.FRAME)? mtd.getDim2():0);
				}
			}
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
	}
}
