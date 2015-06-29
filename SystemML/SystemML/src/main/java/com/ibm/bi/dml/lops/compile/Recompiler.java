/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.lops.compile;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.conf.ConfigurationManager;
import com.ibm.bi.dml.hops.BinaryOp;
import com.ibm.bi.dml.hops.DataOp;
import com.ibm.bi.dml.hops.FunctionOp;
import com.ibm.bi.dml.hops.FunctionOp.FunctionType;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.Hop.FileFormatTypes;
import com.ibm.bi.dml.hops.Hop.OpOp1;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.IndexingOp;
import com.ibm.bi.dml.hops.LiteralOp;
import com.ibm.bi.dml.hops.MemoTable;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.hops.DataGenOp;
import com.ibm.bi.dml.hops.ReorgOp;
import com.ibm.bi.dml.hops.Hop.DataGenMethod;
import com.ibm.bi.dml.hops.Hop.DataOpTypes;
import com.ibm.bi.dml.hops.Hop.VisitStatus;
import com.ibm.bi.dml.hops.UnaryOp;
import com.ibm.bi.dml.hops.rewrite.HopRewriteUtils;
import com.ibm.bi.dml.hops.rewrite.ProgramRewriter;
import com.ibm.bi.dml.lops.CSVReBlock;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.DataGen;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.LopsException;
import com.ibm.bi.dml.lops.ReBlock;
import com.ibm.bi.dml.parser.DMLProgram;
import com.ibm.bi.dml.parser.DMLTranslator;
import com.ibm.bi.dml.parser.DataExpression;
import com.ibm.bi.dml.parser.ForStatementBlock;
import com.ibm.bi.dml.parser.IfStatementBlock;
import com.ibm.bi.dml.parser.Statement;
import com.ibm.bi.dml.parser.StatementBlock;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.parser.WhileStatementBlock;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.ForProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.FunctionProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.IfProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.LocalVariableMap;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.WhileProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.controlprogram.parfor.ProgramConverter;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.OptTreeConverter;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.MRJobInstruction;
import com.ibm.bi.dml.runtime.instructions.cp.Data;
import com.ibm.bi.dml.runtime.instructions.cp.FunctionCallCPInstruction;
import com.ibm.bi.dml.runtime.instructions.cp.IntObject;
import com.ibm.bi.dml.runtime.instructions.cp.ScalarObject;
import com.ibm.bi.dml.runtime.instructions.mr.RandInstruction;
import com.ibm.bi.dml.runtime.instructions.mr.SeqInstruction;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixFormatMetaData;
import com.ibm.bi.dml.runtime.matrix.data.InputInfo;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.utils.Explain;
import com.ibm.bi.dml.utils.Explain.ExplainType;
import com.ibm.json.java.JSONObject;

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
public class Recompiler 
{	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final Log LOG = LogFactory.getLog(Recompiler.class.getName());
	
	//Max threshold for in-memory reblock of text input [in bytes]
	//reason: single-threaded text read at 20MB/s, 1GB input -> 50s (should exploit parallelism)
	//note that we scale this threshold up by the degree of available parallelism
	private static final long CP_REBLOCK_THRESHOLD_SIZE = (long)1024*1024*1024; 
	private static final long CP_CSV_REBLOCK_UNKNOWN_THRESHOLD_SIZE = (long)256*1024*1024;
	
	//reused rewriter for dynamic rewrites during recompile
	private static ProgramRewriter rewriter = new ProgramRewriter(false, true);
	
	/**
	 * Re-initializes the recompiler according to the current optimizer flags.
	 */
	public static void reinitRecompiler()
	{
		rewriter = new ProgramRewriter(false, true);
	}
	
	/**
	 * A) Recompile basic program block hop DAG.  
	 * 	
	 * We support to basic types inplace or via deep copy. Deep copy is the default and is required 
	 * in order to apply non-reversible rewrites. In-place is required in order to modify the existing
	 * hops (e.g., for parfor pre-recompilation). 
	 * 
	 * @param hops
	 * @param vars
	 * @return
	 * @throws DMLRuntimeException
	 * @throws HopsException
	 * @throws LopsException
	 * @throws DMLUnsupportedOperationException
	 * @throws IOException
	 */
	public static ArrayList<Instruction> recompileHopsDag( StatementBlock sb, ArrayList<Hop> hops, LocalVariableMap vars, RecompileStatus status, boolean inplace, long tid ) 
		throws DMLRuntimeException, HopsException, LopsException, DMLUnsupportedOperationException, IOException
	{
		ArrayList<Instruction> newInst = null;

		//need for synchronization as we do temp changes in shared hops/lops
		//however, we create deep copies for most dags to allow for concurrent recompile
		synchronized( hops ) 
		{	
			LOG.debug ("\n**************** Optimizer (Recompile) *************\nMemory Budget = " + 
					   OptimizerUtils.toMB(OptimizerUtils.getLocalMemBudget()) + " MB");
	
			// prepare hops dag for recompile
			if( !inplace ){ 
				// deep copy hop dag (for non-reversable rewrites)
				hops = deepCopyHopsDag(hops);
			}
			else {
				// clear existing lops
				Hop.resetVisitStatus(hops);
				for( Hop hopRoot : hops )
					rClearLops( hopRoot );
			}

			// replace scalar reads with literals 
			if( !inplace ) {
				Hop.resetVisitStatus(hops);
				for( Hop hopRoot : hops )
					rReplaceLiterals( hopRoot, vars );
			}
			
			// refresh matrix characteristics (update stats)			
			Hop.resetVisitStatus(hops);
			for( Hop hopRoot : hops )
				rUpdateStatistics( hopRoot, vars );
			
			// dynamic hop rewrites
			if( !inplace )
				rewriter.rewriteHopDAGs( hops, null );
			
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
			
			// construct lops			
			Dag<Lop> dag = new Dag<Lop>();
			for( Hop hopRoot : hops ){
				Lop lops = hopRoot.constructLops();
				lops.addToDag(dag);	
			}		
			
			// generate runtime instructions (incl piggybacking)
			newInst = dag.getJobs(sb, ConfigurationManager.getConfig());	
		}
		
		// replace thread ids in new instructions
		if( tid != 0 ) //only in parfor context
			newInst = ProgramConverter.createDeepCopyInstructionSet(newInst, tid, -1, null, null, null, false, false);
		
		// explain recompiled hops / instructions
		if( DMLScript.EXPLAIN == ExplainType.RECOMPILE_HOPS ){
			LOG.info("EXPLAIN RECOMPILE \nGENERIC (lines "+sb.getBeginLine()+"-"+sb.getEndLine()+"):\n" + 
		    Explain.explainHops(hops, 1));
		}
		if( DMLScript.EXPLAIN == ExplainType.RECOMPILE_RUNTIME ){
			LOG.info("EXPLAIN RECOMPILE \nGENERIC (lines "+sb.getBeginLine()+"-"+sb.getEndLine()+"):\n" + 
		    Explain.explain(newInst, 1));
		}
	
		return newInst;
	}

	/**
	 * B) Recompile predicate hop DAG (single root): 
	 * 
	 * Note: This overloaded method is required for predicate instructions because
	 * they have only a single hops DAG and we need to synchronize on the original 
	 * (shared) hops object. Hence, we cannot create any wrapper arraylist for each
	 * recompilation - this would result in race conditions for concurrent recompilation 
	 * in a parfor body. 	
	 * 
	 * Note: no statementblock passed because for predicate dags we dont have separate live variable analysis information.
	 * 
	 * @param hops
	 * @param vars
	 * @return
	 * @throws DMLRuntimeException
	 * @throws HopsException
	 * @throws LopsException
	 * @throws DMLUnsupportedOperationException
	 * @throws IOException
	 */
	public static ArrayList<Instruction> recompileHopsDag( Hop hops, LocalVariableMap vars, boolean inplace, long tid ) 
		throws DMLRuntimeException, HopsException, LopsException, DMLUnsupportedOperationException, IOException
	{
		ArrayList<Instruction> newInst = null;

		//need for synchronization as we do temp changes in shared hops/lops
		synchronized( hops ) 
		{	
			LOG.debug ("\n**************** Optimizer (Recompile) *************\nMemory Budget = " + 
					   OptimizerUtils.toMB(OptimizerUtils.getLocalMemBudget()) + " MB");

			// prepare hops dag for recompile
			if( !inplace ) {
				// deep copy hop dag (for non-reversable rewrites)
				//(this also clears existing lops in the created dag) 
				hops = deepCopyHopsDag(hops);	
			}
			else {
				// clear existing lops
				hops.resetVisitStatus();
				rClearLops( hops );	
			}
			
			// replace scalar reads with literals 
			if( !inplace ) {
				hops.resetVisitStatus();
				rReplaceLiterals( hops, vars );
			}
			
			// refresh matrix characteristics (update stats)			
			hops.resetVisitStatus();
			rUpdateStatistics( hops, vars );
			
			// dynamic hop rewrites
			if( !inplace )
				rewriter.rewriteHopDAG( hops, null );
			
			// refresh memory estimates (based on updated stats)
			hops.resetVisitStatus();
			hops.refreshMemEstimates(new MemoTable()); 		
			
			// construct lops			
			Dag<Lop> dag = new Dag<Lop>();
			Lop lops = hops.constructLops();
			lops.addToDag(dag);		
			
			// generate runtime instructions (incl piggybacking)
			newInst = dag.getJobs(null, ConfigurationManager.getConfig());
		}
		
		// replace thread ids in new instructions
		if( tid != 0 ) //only in parfor context
			newInst = ProgramConverter.createDeepCopyInstructionSet(newInst, tid, -1, null, null, null, false, false);
		
		// explain recompiled instructions
		if( DMLScript.EXPLAIN == ExplainType.RECOMPILE_HOPS )
			LOG.info("EXPLAIN RECOMPILE \nPRED (line "+hops.getBeginLine()+"):\n" + Explain.explain(hops,1));
		if( DMLScript.EXPLAIN == ExplainType.RECOMPILE_RUNTIME )
			LOG.info("EXPLAIN RECOMPILE \nPRED (line "+hops.getBeginLine()+"):\n" + Explain.explain(newInst,1));
		
		return newInst;
	}
	
	/**
	 * C) Recompile basic program block hop DAG, but forced to CP.  
	 * 
	 * This happens always 'inplace', without statistics updates, and 
	 * without dynamic rewrites.
	 * 
	 * @param hops
	 * @param tid
	 * @return
	 * @throws DMLRuntimeException
	 * @throws HopsException
	 * @throws LopsException
	 * @throws DMLUnsupportedOperationException
	 * @throws IOException
	 */
	public static ArrayList<Instruction> recompileHopsDag2Forced( StatementBlock sb, ArrayList<Hop> hops, long tid, ExecType et ) 
		throws DMLRuntimeException, HopsException, LopsException, DMLUnsupportedOperationException, IOException
	{
		ArrayList<Instruction> newInst = null;
		
		//need for synchronization as we do temp changes in shared hops/lops
		//however, we create deep copies for most dags to allow for concurrent recompile
		synchronized( hops ) 
		{	
			LOG.debug ("\n**************** Optimizer (Recompile) *************\nMemory Budget = " + 
					   OptimizerUtils.toMB(OptimizerUtils.getLocalMemBudget()) + " MB");
	
			// clear existing lops
			Hop.resetVisitStatus(hops);
			for( Hop hopRoot : hops )
				rClearLops( hopRoot );
			
			// update exec type
			Hop.resetVisitStatus(hops);
			for( Hop hopRoot : hops )
				rSetExecType( hopRoot, et );
			Hop.resetVisitStatus(hops);
			
			// construct lops			
			Dag<Lop> dag = new Dag<Lop>();
			for( Hop hopRoot : hops ){
				Lop lops = hopRoot.constructLops();
				lops.addToDag(dag);	
			}		
			
			// generate runtime instructions (incl piggybacking)
			newInst = dag.getJobs(sb, ConfigurationManager.getConfig());			
		}
		
		// replace thread ids in new instructions
		if( tid != 0 ) //only in parfor context
			newInst = ProgramConverter.createDeepCopyInstructionSet(newInst, tid, -1, null, null, null, false, false);
		
		return newInst;
	}

	/**
	 * D) Recompile predicate hop DAG (single root), but forced to CP. 
	 * 
	 * This happens always 'inplace', without statistics updates, and 
	 * without dynamic rewrites.
	 * 
	 * @param hops
	 * @param tid
	 * @param et
	 * @return
	 * @throws DMLRuntimeException
	 * @throws HopsException
	 * @throws LopsException
	 * @throws DMLUnsupportedOperationException
	 * @throws IOException
	 */
	public static ArrayList<Instruction> recompileHopsDag2Forced( Hop hops, long tid, ExecType et ) 
		throws DMLRuntimeException, HopsException, LopsException, DMLUnsupportedOperationException, IOException
	{
		ArrayList<Instruction> newInst = null;

		//need for synchronization as we do temp changes in shared hops/lops
		synchronized( hops ) 
		{	
			LOG.debug ("\n**************** Optimizer (Recompile) *************\nMemory Budget = " + 
					   OptimizerUtils.toMB(OptimizerUtils.getLocalMemBudget()) + " MB");

			// clear existing lops
			hops.resetVisitStatus();
			rClearLops( hops );	
			
			// update exec type
			hops.resetVisitStatus();
			rSetExecType( hops, et );
			hops.resetVisitStatus();
			
			// construct lops			
			Dag<Lop> dag = new Dag<Lop>();
			Lop lops = hops.constructLops();
			lops.addToDag(dag);		
			
			// generate runtime instructions (incl piggybacking)
			newInst = dag.getJobs(null, ConfigurationManager.getConfig());
		}
		
		// replace thread ids in new instructions
		if( tid != 0 ) //only in parfor context
			newInst = ProgramConverter.createDeepCopyInstructionSet(newInst, tid, -1, null, null, null, false, false);
		
		return newInst;
	}

	/**
	 * 
	 * @param sb
	 * @param hops
	 * @return
	 * @throws HopsException
	 * @throws LopsException
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 * @throws IOException
	 */
	public static ArrayList<Instruction> recompileHopsDagInstructions( StatementBlock sb, ArrayList<Hop> hops ) 
		throws HopsException, LopsException, DMLRuntimeException, DMLUnsupportedOperationException, IOException 
	{
		ArrayList<Instruction> newInst = null;

		//need for synchronization as we do temp changes in shared hops/lops
		//however, we create deep copies for most dags to allow for concurrent recompile
		synchronized( hops ) 
		{	
			LOG.debug ("\n**************** Optimizer (Recompile) *************\nMemory Budget = " + 
					   OptimizerUtils.toMB(OptimizerUtils.getLocalMemBudget()) + " MB");
	
			// clear existing lops
			Hop.resetVisitStatus(hops);
			for( Hop hopRoot : hops )
				rClearLops( hopRoot );
			
			// construct lops			
			Dag<Lop> dag = new Dag<Lop>();
			for( Hop hopRoot : hops ){
				Lop lops = hopRoot.constructLops();
				lops.addToDag(dag);	
			}		
			
			// generate runtime instructions (incl piggybacking)
			newInst = dag.getJobs(sb, ConfigurationManager.getConfig());	
		}
		
		// explain recompiled hops / instructions
		if( DMLScript.EXPLAIN == ExplainType.RECOMPILE_HOPS ){
			LOG.info("EXPLAIN RECOMPILE \nGENERIC (lines "+sb.getBeginLine()+"-"+sb.getEndLine()+"):\n" + 
		    Explain.explainHops(hops, 1));
		}
		if( DMLScript.EXPLAIN == ExplainType.RECOMPILE_RUNTIME ){
			LOG.info("EXPLAIN RECOMPILE \nGENERIC (lines "+sb.getBeginLine()+"-"+sb.getEndLine()+"):\n" + 
		    Explain.explain(newInst, 1));
		}
	
		return newInst;
	}

	/**
	 * 
	 * @param hops
	 * @return
	 * @throws DMLRuntimeException
	 * @throws HopsException
	 * @throws LopsException
	 * @throws DMLUnsupportedOperationException
	 * @throws IOException
	 */
	public static ArrayList<Instruction> recompileHopsDagInstructions( Hop hops ) 
		throws DMLRuntimeException, HopsException, LopsException, DMLUnsupportedOperationException, IOException
	{
		ArrayList<Instruction> newInst = null;

		//need for synchronization as we do temp changes in shared hops/lops
		synchronized( hops ) 
		{	
			LOG.debug ("\n**************** Optimizer (Recompile) *************\nMemory Budget = " + 
					   OptimizerUtils.toMB(OptimizerUtils.getLocalMemBudget()) + " MB");

			// clear existing lops
			hops.resetVisitStatus();
			rClearLops( hops );	

			// construct lops			
			Dag<Lop> dag = new Dag<Lop>();
			Lop lops = hops.constructLops();
			lops.addToDag(dag);		
			
			// generate runtime instructions (incl piggybacking)
			newInst = dag.getJobs(null, ConfigurationManager.getConfig());
		}

		// explain recompiled instructions
		if( DMLScript.EXPLAIN == ExplainType.RECOMPILE_HOPS )
			LOG.info("EXPLAIN RECOMPILE \nPRED (line "+hops.getBeginLine()+"):\n" + Explain.explain(hops,1));
		if( DMLScript.EXPLAIN == ExplainType.RECOMPILE_RUNTIME )
			LOG.info("EXPLAIN RECOMPILE \nPRED (line "+hops.getBeginLine()+"):\n" + Explain.explain(newInst,1));
		
		return newInst;
	}

	
	/**
	 * 
	 * @param pbs
	 * @param vars
	 * @param tid
	 * @throws DMLRuntimeException 
	 */
	public static void recompileProgramBlockHierarchy( ArrayList<ProgramBlock> pbs, LocalVariableMap vars, long tid, boolean resetRecompile ) 
		throws DMLRuntimeException
	{
		try 
		{
			RecompileStatus status = new RecompileStatus();
			
			synchronized( pbs )
			{
				for( ProgramBlock pb : pbs )
					rRecompileProgramBlock(pb, vars, status, tid, resetRecompile);
			}
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException("Unable to recompile program block hierarchy.", ex);
		}
	}
	
	/**
	 * Method to recompile program block hierarchy to forced execution time. This affects also
	 * referenced functions and chains of functions. Use et==null in order to release the forced 
	 * exec type.
	 * 
	 * @param pbs
	 * @param tid
	 * @throws DMLRuntimeException
	 */
	public static void recompileProgramBlockHierarchy2Forced( ArrayList<ProgramBlock> pbs, long tid, HashSet<String> fnStack, ExecType et ) 
		throws DMLRuntimeException
	{
		try 
		{
			synchronized( pbs )
			{
				for( ProgramBlock pb : pbs )
					rRecompileProgramBlock2Forced(pb, tid, fnStack, et);
			}
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException("Unable to recompile program block hierarchy to CP.", ex);
		}
	}
	
	/**
	 * This method does NO full program block recompile (no stats update, no rewrites, no recursion) but
	 * only regenerates lops and instructions. The primary use case is recompilation after are hop configuration 
	 * changes which allows to preserve statistics (e.g., propagated worst case stats from other program blocks)
	 * and better performance for recompiling individual program blocks.  
	 * 
	 * @param pb
	 * @throws IOException 
	 * @throws DMLUnsupportedOperationException 
	 * @throws DMLRuntimeException 
	 * @throws LopsException 
	 * @throws HopsException 
	 */
	public static void recompileProgramBlockInstructions(ProgramBlock pb) 
		throws HopsException, LopsException, DMLRuntimeException, DMLUnsupportedOperationException, IOException
	{
		if( pb instanceof WhileProgramBlock )
		{
			//recompile while predicate instructions
			WhileProgramBlock wpb = (WhileProgramBlock)pb;
			WhileStatementBlock wsb = (WhileStatementBlock) pb.getStatementBlock();
			if( wsb!=null && wsb.getPredicateHops()!=null )
				wpb.setPredicate(recompileHopsDagInstructions(wsb.getPredicateHops()));
		}
		else if( pb instanceof IfProgramBlock )
		{
			//recompile if predicate instructions
			IfProgramBlock ipb = (IfProgramBlock)pb;
			IfStatementBlock isb = (IfStatementBlock) pb.getStatementBlock();
			if( isb!=null && isb.getPredicateHops()!=null )
				ipb.setPredicate(recompileHopsDagInstructions(isb.getPredicateHops()));
		}
		else if( pb instanceof ForProgramBlock )
		{
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
		else
		{
			//recompile last-level program block instructions
			StatementBlock sb = pb.getStatementBlock();
			if( sb!=null && sb.get_hops()!=null ) {
				pb.setInstructions(recompileHopsDagInstructions(sb, sb.get_hops()));
			}
		}
	}
	
	/**
	 * 
	 * @param hops
	 * @return
	 */
	public static boolean requiresRecompilation( ArrayList<Hop> hops )
	{
		boolean ret = false;
		
		if( hops != null )
		{
			synchronized( hops )
			{
				Hop.resetVisitStatus(hops);
				for( Hop hop : hops )
				{
					ret |= rRequiresRecompile(hop);
					if( ret ) break; // early abort
				}
			}
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @param hops
	 * @return
	 */
	public static boolean requiresRecompilation( Hop hop )
	{
		boolean ret = false;
		
		if( hop != null )
		{
			synchronized( hop )
			{
				hop.resetVisitStatus();
				ret = rRequiresRecompile(hop);
			}
		}
		
		return ret;
	}
	

	/**
	 * Deep copy of hops dags for parallel recompilation.
	 * 
	 * @param hops
	 * @return
	 * @throws CloneNotSupportedException
	 */
	public static ArrayList<Hop> deepCopyHopsDag( ArrayList<Hop> hops ) 
		throws HopsException 
	{
		ArrayList<Hop> ret = new ArrayList<Hop>();
		
		try {
			//note: need memo table over all independent DAGs in order to 
			//account for shared transient reads (otherwise more instructions generated)
			HashMap<Long, Hop> memo = new HashMap<Long, Hop>(); //orig ID, new clone
			for( Hop hopRoot : hops )
				ret.add(rDeepCopyHopsDag(hopRoot, memo));
		}
		catch(Exception ex)
		{
			throw new HopsException(ex);
		}
		
		return ret;
	}
	
	/**
	 * Deep copy of hops dags for parallel recompilation.
	 * 
	 * @param hops
	 * @return
	 * @throws CloneNotSupportedException
	 */
	public static Hop deepCopyHopsDag( Hop hops ) 
		throws HopsException 
	{
		Hop ret = null;
		
		try {
			HashMap<Long, Hop> memo = new HashMap<Long, Hop>(); //orig ID, new clone
			ret = rDeepCopyHopsDag(hops, memo);
		}
		catch(Exception ex)
		{
			throw new HopsException(ex);
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @param hops
	 * @param memo
	 * @return
	 * @throws CloneNotSupportedException
	 */
	private static Hop rDeepCopyHopsDag( Hop hops, HashMap<Long,Hop> memo ) 
		throws CloneNotSupportedException
	{
		Hop ret = memo.get(hops.getHopID());
	
		//create clone if required 
		if( ret == null ) 
		{
			ret = (Hop) hops.clone();
			ArrayList<Hop> tmp = new ArrayList<Hop>();
			
			//create new childs
			for( Hop in : hops.getInput() )
			{
				Hop newIn = rDeepCopyHopsDag(in, memo);
				tmp.add(newIn);
			}
			//modify references of childs
			for( Hop in : tmp )
			{
				ret.getInput().add(in);
				in.getParent().add(ret);
			}
			
			memo.put(hops.getHopID(), ret);
		}
		
		return ret;
	}
	

	public static void updateFunctionNames(ArrayList<Hop> hops, long pid) 
	{
		Hop.resetVisitStatus(hops);
		for( Hop hopRoot : hops  )
			rUpdateFunctionNames( hopRoot, pid );
	}
	
	public static void rUpdateFunctionNames( Hop hop, long pid )
	{
		if( hop.getVisited() == VisitStatus.DONE )
			return;
		
		//update function names
		if( hop instanceof FunctionOp && ((FunctionOp)hop).getFunctionType() != FunctionType.MULTIRETURN_BUILTIN) {
			FunctionOp fop = (FunctionOp) hop;
			fop.setFunctionName( fop.getFunctionName() +
					             ProgramConverter.CP_CHILD_THREAD + pid);
		}
		
		if( hop.getInput() != null )
			for( Hop c : hop.getInput() )
				rUpdateFunctionNames(c, pid);
		
		hop.setVisited(VisitStatus.DONE);
	}
	
	
	//////////////////////////////
	// private helper functions //
	//////////////////////////////
	
	
	/**
	 * 
	 * @param pb
	 * @param vars
	 * @param tid
	 * @throws IOException 
	 * @throws DMLUnsupportedOperationException 
	 * @throws LopsException 
	 * @throws DMLRuntimeException 
	 * @throws HopsException 
	 */
	private static void rRecompileProgramBlock( ProgramBlock pb, LocalVariableMap vars, RecompileStatus status, long tid, boolean resetRecompile ) 
		throws HopsException, DMLRuntimeException, LopsException, DMLUnsupportedOperationException, IOException
	{
		if (pb instanceof WhileProgramBlock)
		{
			WhileProgramBlock wpb = (WhileProgramBlock)pb;
			WhileStatementBlock wsb = (WhileStatementBlock) wpb.getStatementBlock();
			//recompile predicate
			recompileWhilePredicate(wpb, wsb, vars, tid, resetRecompile);
			//remove updated scalars because in loop
			removeUpdatedScalars(vars, wsb); 
			//copy vars for later compare
			LocalVariableMap oldVars = (LocalVariableMap) vars.clone();
			status.clearStatus();
			for (ProgramBlock pb2 : wpb.getChildBlocks())
				rRecompileProgramBlock(pb2, vars, status, tid, resetRecompile);
			if( reconcileUpdatedCallVarsLoops(oldVars, vars, wsb) ) {
				//second pass with unknowns if required
				recompileWhilePredicate(wpb, wsb, vars, tid, resetRecompile);
				status.clearStatus();
				for (ProgramBlock pb2 : wpb.getChildBlocks())
					rRecompileProgramBlock(pb2, vars, status, tid, resetRecompile);
			}
			removeUpdatedScalars(vars, wsb);
		}
		else if (pb instanceof IfProgramBlock)
		{
			IfProgramBlock ipb = (IfProgramBlock)pb;	
			IfStatementBlock isb = (IfStatementBlock)ipb.getStatementBlock();
			//recompile predicate
			recompileIfPredicate(ipb, isb, vars, tid, resetRecompile);
			//copy vars for later compare
			LocalVariableMap oldVars = (LocalVariableMap) vars.clone();
			LocalVariableMap varsElse = (LocalVariableMap) vars.clone();
			status.clearStatus();
			for( ProgramBlock pb2 : ipb.getChildBlocksIfBody() )
				rRecompileProgramBlock(pb2, vars, status, tid, resetRecompile);
			status.clearStatus();
			for( ProgramBlock pb2 : ipb.getChildBlocksElseBody() )
				rRecompileProgramBlock(pb2, varsElse, status, tid, resetRecompile);
			status.clearStatus();
			reconcileUpdatedCallVarsIf(oldVars, vars, varsElse, isb);
			removeUpdatedScalars(vars, ipb.getStatementBlock());
		}
		else if (pb instanceof ForProgramBlock) //includes ParFORProgramBlock
		{ 
			ForProgramBlock fpb = (ForProgramBlock)pb;	
			ForStatementBlock fsb = (ForStatementBlock) fpb.getStatementBlock();
			//recompile predicates
			recompileForPredicates(fpb, fsb, vars, tid, resetRecompile);
			//remove updated scalars because in loop
			removeUpdatedScalars(vars, fpb.getStatementBlock()); 
			//copy vars for later compare
			LocalVariableMap oldVars = (LocalVariableMap) vars.clone();
			status.clearStatus();
			for( ProgramBlock pb2 : fpb.getChildBlocks() )
				rRecompileProgramBlock(pb2, vars, status, tid, resetRecompile);
			if( reconcileUpdatedCallVarsLoops(oldVars, vars, fsb) ) {
				//second pass with unknowns if required
				recompileForPredicates(fpb, fsb, vars, tid, resetRecompile);
				status.clearStatus();
				for( ProgramBlock pb2 : fpb.getChildBlocks() )
					rRecompileProgramBlock(pb2, vars, status, tid, resetRecompile);		
			}
			removeUpdatedScalars(vars, fpb.getStatementBlock());
		}		
		else if (  pb instanceof FunctionProgramBlock ) //includes ExternalFunctionProgramBlock and ExternalFunctionProgramBlockCP
		{
			//do nothing
		}
		else 
		{	
			StatementBlock sb = pb.getStatementBlock();
			ArrayList<Instruction> tmp = pb.getInstructions();

			if(	sb != null //recompile all for stats propagation and recompile flags
				//&& Recompiler.requiresRecompilation( sb.get_hops() ) 
				/*&& !Recompiler.containsNonRecompileInstructions(tmp)*/ )
			{
				tmp = Recompiler.recompileHopsDag(sb, sb.get_hops(), vars, status, true, tid);
				pb.setInstructions( tmp );
				
				//propagate stats across hops (should be executed on clone of vars)
				Recompiler.extractDAGOutputStatistics(sb.get_hops(), vars);
				
				//reset recompilation flags (w/ special handling functions)
				if(    ParForProgramBlock.RESET_RECOMPILATION_FLAGs 
					&& !containsRootFunctionOp(sb.get_hops())  
					&& resetRecompile ) 
				{
					Hop.resetRecompilationFlag(sb.get_hops(), ExecType.CP);
					sb.updateRecompilationFlag();
				}
			}
			
		}
		
	}
	

	/**
	 * 
	 * @param oldCallVars
	 * @param callVars
	 * @param sb
	 * @return
	 */
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
				MatrixCharacteristics mcOld = ((MatrixFormatMetaData)moOld.getMetaData()).getMatrixCharacteristics();
				MatrixCharacteristics mc = ((MatrixFormatMetaData)mo.getMetaData()).getMatrixCharacteristics();
				
				if( mcOld.getRows() != mc.getRows() 
					|| mcOld.getCols() != mc.getCols()
					|| mcOld.getNonZeros() != mc.getNonZeros() )
				{
					long ldim1 =mc.getRows(), ldim2=mc.getCols(), lnnz=mc.getNonZeros();
					//handle dimension change in body
					if(    mcOld.getRows() != mc.getRows() 
						|| mcOld.getCols() != mc.getCols() )
					{
						ldim1=-1;
						ldim2=-1; //unknown
						requiresRecompile = true;
					}
					//handle sparsity change
					if( mcOld.getNonZeros() != mc.getNonZeros() )
					{
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

	/**
	 * 
	 * @param oldCallVars
	 * @param callVarsIf
	 * @param callVarsElse
	 * @param sb
	 * @return
	 */
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
					MatrixCharacteristics mcOld = ((MatrixFormatMetaData)moOld.getMetaData()).getMatrixCharacteristics();
					MatrixCharacteristics mc = ((MatrixFormatMetaData)mo.getMetaData()).getMatrixCharacteristics();
					
					if( mcOld.getRows() != mc.getRows() 
							|| mcOld.getCols() != mc.getCols()
							|| mcOld.getNonZeros() != mc.getNonZeros() )
					{
						long ldim1 =mc.getRows(), ldim2=mc.getCols(), lnnz=mc.getNonZeros();
						
						//handle dimension change
						if(    mcOld.getRows() != mc.getRows() 
							|| mcOld.getCols() != mc.getCols() )
						{
							ldim1=-1; ldim2=-1; //unknown
						}
						//handle sparsity change
						if( mcOld.getNonZeros() != mc.getNonZeros() )
						{
							lnnz=-1; //unknown		
						}
						
						MatrixObject moNew = createOutputMatrix(ldim1, ldim2, lnnz);
						callVarsIf.put(varname, moNew);
					}
				}
			}
		}
		
		return callVarsIf;
	}
	
	/**
	 * 
	 * @param hops
	 * @return
	 */
	private static boolean containsRootFunctionOp( ArrayList<Hop> hops )
	{
		boolean ret = false;
		for( Hop h : hops )
			if( h instanceof FunctionOp )
				ret |= true;
		
		return ret;
	}
	
	/**
	 * 
	 * @param dim1
	 * @param dim2
	 * @param nnz
	 * @return
	 */
	private static MatrixObject createOutputMatrix( long dim1, long dim2, long nnz )
	{
		MatrixObject moOut = new MatrixObject(ValueType.DOUBLE, null);
		MatrixCharacteristics mc = new MatrixCharacteristics( 
									dim1, dim2,
									DMLTranslator.DMLBlockSize, DMLTranslator.DMLBlockSize,
									nnz);
		MatrixFormatMetaData meta = new MatrixFormatMetaData(mc,null,null);
		moOut.setMetaData(meta);
		
		return moOut;
	}
	
	
	//helper functions for predicate recompile
	
	/**
	 * 
	 * @param ipb
	 * @param isb
	 * @param vars
	 * @param tid
	 * @throws DMLRuntimeException
	 * @throws HopsException
	 * @throws LopsException
	 * @throws DMLUnsupportedOperationException
	 * @throws IOException
	 */
	private static void recompileIfPredicate( IfProgramBlock ipb, IfStatementBlock isb, LocalVariableMap vars, long tid, boolean resetRecompile ) 
		throws DMLRuntimeException, HopsException, LopsException, DMLUnsupportedOperationException, IOException
	{
		if( isb != null )
		{
			Hop hops = isb.getPredicateHops();
			if( hops != null ) {
				ArrayList<Instruction> tmp = recompileHopsDag(hops, vars, true, tid);
				ipb.setPredicate( tmp );
				if( ParForProgramBlock.RESET_RECOMPILATION_FLAGs
					&& resetRecompile ) 
				{
					Hop.resetRecompilationFlag(hops, ExecType.CP);
					isb.updatePredicateRecompilationFlag();
				}

				//update predicate vars (potentially after constant folding, e.g., in parfor)
				if( hops instanceof LiteralOp )
					ipb.setPredicateResultVar(((LiteralOp)hops).getName().toLowerCase());
			}
		}
	}
	
	/**
	 * 
	 * @param wpb
	 * @param wsb
	 * @param vars
	 * @param tid
	 * @throws DMLRuntimeException
	 * @throws HopsException
	 * @throws LopsException
	 * @throws DMLUnsupportedOperationException
	 * @throws IOException
	 */
	private static void recompileWhilePredicate( WhileProgramBlock wpb, WhileStatementBlock wsb, LocalVariableMap vars, long tid, boolean resetRecompile ) 
		throws DMLRuntimeException, HopsException, LopsException, DMLUnsupportedOperationException, IOException
	{
		if( wsb != null )
		{
			Hop hops = wsb.getPredicateHops();
			if( hops != null ) {
				ArrayList<Instruction> tmp = recompileHopsDag(hops, vars, true, tid);
				wpb.setPredicate( tmp );
				if( ParForProgramBlock.RESET_RECOMPILATION_FLAGs 
					&& resetRecompile ) 
				{
					Hop.resetRecompilationFlag(hops, ExecType.CP);
					wsb.updatePredicateRecompilationFlag();
				}
				
				//update predicate vars (potentially after constant folding, e.g., in parfor)
				if( hops instanceof LiteralOp )
					wpb.setPredicateResultVar(((LiteralOp)hops).getName().toLowerCase());
			}
		}
	}
	
	/**
	 * 
	 * @param fpb
	 * @param fsb
	 * @param vars
	 * @param tid
	 * @throws DMLRuntimeException
	 * @throws HopsException
	 * @throws LopsException
	 * @throws DMLUnsupportedOperationException
	 * @throws IOException
	 */
	private static void recompileForPredicates( ForProgramBlock fpb, ForStatementBlock fsb, LocalVariableMap vars, long tid, boolean resetRecompile ) 
		throws DMLRuntimeException, HopsException, LopsException, DMLUnsupportedOperationException, IOException
	{
		if( fsb != null )
		{
			Hop fromHops = fsb.getFromHops();
			Hop toHops = fsb.getToHops();
			Hop incrHops = fsb.getIncrementHops();
			
			//handle recompilation flags
			if( ParForProgramBlock.RESET_RECOMPILATION_FLAGs 
				&& resetRecompile ) 
			{
				if( fromHops != null ) {
					ArrayList<Instruction> tmp = recompileHopsDag(fromHops, vars, true, tid);
					fpb.setFromInstructions(tmp);
					Hop.resetRecompilationFlag(fromHops,ExecType.CP);
				}
				if( toHops != null ) {
					ArrayList<Instruction> tmp = recompileHopsDag(toHops, vars, true, tid);
					fpb.setToInstructions(tmp);
					Hop.resetRecompilationFlag(toHops,ExecType.CP);
				}
				if( incrHops != null ) {
					ArrayList<Instruction> tmp = recompileHopsDag(incrHops, vars, true, tid);
					fpb.setIncrementInstructions(tmp);
					Hop.resetRecompilationFlag(incrHops,ExecType.CP);
				}
				fsb.updatePredicateRecompilationFlags();
			}
			else //no reset of recompilation flags
			{
				if( fromHops != null ) {
					ArrayList<Instruction> tmp = recompileHopsDag(fromHops, vars, true, tid);
					fpb.setFromInstructions(tmp);
				}
				if( toHops != null ) {
					ArrayList<Instruction> tmp = recompileHopsDag(toHops, vars, true, tid);
					fpb.setToInstructions(tmp);
				}
				if( incrHops != null ) {
					ArrayList<Instruction> tmp = recompileHopsDag(incrHops, vars, true, tid);
					fpb.setIncrementInstructions(tmp);
				}
			}
			
			//update predicate vars (potentially after constant folding, e.g., in parfor)
			String[] itervars = fpb.getIterablePredicateVars();
			if( fromHops != null && fromHops instanceof LiteralOp )
				itervars[1] = ((LiteralOp)fromHops).getName();
			if( toHops != null && toHops instanceof LiteralOp )
				itervars[2] = ((LiteralOp)toHops).getName();
			if( incrHops != null && incrHops instanceof LiteralOp )
				itervars[3] = ((LiteralOp)incrHops).getName();	
		}
	}
	
	/**
	 * 
	 * @param pb
	 * @param tid
	 * @throws HopsException
	 * @throws DMLRuntimeException
	 * @throws LopsException
	 * @throws DMLUnsupportedOperationException
	 * @throws IOException
	 */
	private static void rRecompileProgramBlock2Forced( ProgramBlock pb, long tid, HashSet<String> fnStack, ExecType et ) 
		throws HopsException, DMLRuntimeException, LopsException, DMLUnsupportedOperationException, IOException
	{
		if (pb instanceof WhileProgramBlock)
		{
			WhileProgramBlock pbTmp = (WhileProgramBlock)pb;
			WhileStatementBlock sbTmp = (WhileStatementBlock)pbTmp.getStatementBlock();
			//recompile predicate
			if(	sbTmp!=null && !(et==ExecType.CP && !OptTreeConverter.containsMRJobInstruction(pbTmp.getPredicate(), true)) )			
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
			if( sbTmp!=null &&!(et==ExecType.CP && !OptTreeConverter.containsMRJobInstruction(pbTmp.getPredicate(), true)) )			
				pbTmp.setPredicate( Recompiler.recompileHopsDag2Forced(sbTmp.getPredicateHops(), tid, et) );				
			//recompile body
			for( ProgramBlock pb2 : pbTmp.getChildBlocksIfBody() )
				rRecompileProgramBlock2Forced(pb2, tid, fnStack, et);
			for( ProgramBlock pb2 : pbTmp.getChildBlocksElseBody() )
				rRecompileProgramBlock2Forced(pb2, tid, fnStack, et);
		}
		else if (pb instanceof ForProgramBlock) //includes ParFORProgramBlock
		{ 
			ForProgramBlock pbTmp = (ForProgramBlock)pb;	
			ForStatementBlock sbTmp = (ForStatementBlock) pbTmp.getStatementBlock();
			//recompile predicate
			if( sbTmp!=null &&!(et==ExecType.CP && !OptTreeConverter.containsMRJobInstruction(pbTmp.getFromInstructions(), true)) )			
				pbTmp.setFromInstructions( Recompiler.recompileHopsDag2Forced(sbTmp.getFromHops(), tid, et) );				
			if( sbTmp!=null &&!(et==ExecType.CP && !OptTreeConverter.containsMRJobInstruction(pbTmp.getToInstructions(), true)) )			
				pbTmp.setToInstructions( Recompiler.recompileHopsDag2Forced(sbTmp.getToHops(), tid, et) );				
			if( sbTmp!=null &&!(et==ExecType.CP && !OptTreeConverter.containsMRJobInstruction(pbTmp.getIncrementInstructions(), true)) )			
				pbTmp.setIncrementInstructions( Recompiler.recompileHopsDag2Forced(sbTmp.getIncrementHops(), tid, et) );				
			//recompile body
			for( ProgramBlock pb2 : pbTmp.getChildBlocks() )
				rRecompileProgramBlock2Forced(pb2, tid, fnStack, et);
		}		
		else if (  pb instanceof FunctionProgramBlock )//includes ExternalFunctionProgramBlock and ExternalFunctionProgramBlockCP
		{
			FunctionProgramBlock tmp = (FunctionProgramBlock)pb;
			for( ProgramBlock pb2 : tmp.getChildBlocks() )
				rRecompileProgramBlock2Forced(pb2, tid, fnStack, et);
		}
		else 
		{	
			StatementBlock sb = pb.getStatementBlock();
			
			//recompile hops dag to CP (opt: don't recompile if CP and no MR inst)
			if(	sb != null && !(et==ExecType.CP && !OptTreeConverter.containsMRJobInstruction(pb, true)) )
			{
				ArrayList<Instruction> tmp = pb.getInstructions();
				tmp = Recompiler.recompileHopsDag2Forced(sb, sb.get_hops(), tid, et);
				pb.setInstructions( tmp );
			}
			
			//recompile functions
			if( OptTreeConverter.containsFunctionCallInstruction(pb) )
			{
				ArrayList<Instruction> tmp = pb.getInstructions();
				for( Instruction inst : tmp )
					if( inst instanceof FunctionCallCPInstruction )
					{
						FunctionCallCPInstruction func = (FunctionCallCPInstruction)inst;
						String fname = func.getFunctionName();
						String fnamespace = func.getNamespace();
						String fKey = DMLProgram.constructFunctionKey(fnamespace, fname);
						
						if( !fnStack.contains(fKey) ) //memoization for multiple calls, recursion
						{
							fnStack.add(fKey);
							
							FunctionProgramBlock fpb = pb.getProgram().getFunctionProgramBlock(fnamespace, fname);
							rRecompileProgramBlock2Forced(fpb, tid, fnStack, et); //recompile chains of functions
						}
					}
			}
		}
		
	}
	
	/**
	 * 
	 * @param callVars
	 * @param sb
	 */
	public static void removeUpdatedScalars( LocalVariableMap callVars, StatementBlock sb )
	{
		if( sb != null )
		{
			//remove update scalar variables from constants
			for( String varname : sb.variablesUpdated().getVariables().keySet() )
			{
				Data dat = callVars.get(varname);
				if( dat != null && dat.getDataType() == DataType.SCALAR )
				{
					callVars.remove(varname);
				}
			}
		}
	}
	
	/**
	 * 
	 * @param hops
	 * @param vars
	 */
	public static void extractDAGOutputStatistics(ArrayList<Hop> hops, LocalVariableMap vars)
	{
		extractDAGOutputStatistics(hops, vars, true);
	}
	
	/**
	 * 
	 * @param hops
	 * @param vars
	 */
	public static void extractDAGOutputStatistics(ArrayList<Hop> hops, LocalVariableMap vars, boolean overwrite)
	{
		for( Hop hop : hops ) //for all hop roots
			extractDAGOutputStatistics(hop, vars, overwrite);
	}
	
	/**
	 * 
	 * @param hop
	 * @param vars
	 */
	public static void extractDAGOutputStatistics(Hop hop, LocalVariableMap vars)
	{
		extractDAGOutputStatistics(hop, vars, true);
	}
	
	/**
	 * 
	 * @param hop
	 * @param vars
	 * @param overwrite
	 */
	public static void extractDAGOutputStatistics(Hop hop, LocalVariableMap vars, boolean overwrite)
	{
		if(    hop instanceof DataOp && ((DataOp)hop).getDataOpType()==DataOpTypes.TRANSIENTWRITE ) //for all writes to symbol table
			//&& hop.getDim1()>0 && hop.getDim2()>0  ) //matrix with known dims 
		{
			String varName = hop.getName();
			if( !vars.keySet().contains(varName) || overwrite ) //not existing so far
			{
				//extract matrix sizes for size propagation
				if( hop.getDataType()==DataType.MATRIX )
				{
					MatrixObject mo = new MatrixObject(ValueType.DOUBLE, null);
					MatrixCharacteristics mc = new MatrixCharacteristics( 
												hop.getDim1(), hop.getDim2(), 
												DMLTranslator.DMLBlockSize, DMLTranslator.DMLBlockSize,
												hop.getNnz());
					MatrixFormatMetaData meta = new MatrixFormatMetaData(mc,null,null);
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
					MatrixCharacteristics mc = ((MatrixFormatMetaData)mo.getMetaData()).getMatrixCharacteristics();
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
	 * @param hop
	 * @return
	 */
	private static boolean rRequiresRecompile( Hop hop )
	{	
		boolean ret = hop.requiresRecompile();
		if( hop.getVisited() == VisitStatus.DONE )
			return ret;
		
		if( hop.getInput() != null )
			for( Hop c : hop.getInput() )
			{
				ret |= rRequiresRecompile(c);
				if( ret ) break; // early abort
			}
		
		hop.setVisited(VisitStatus.DONE);
		
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
	 * @param hop
	 */
	public static void rClearLops( Hop hop )
	{
		if( hop.getVisited() == VisitStatus.DONE )
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
		
		hop.setVisited(VisitStatus.DONE);
	}
	
	/**
	 * 
	 * @param hop
	 * @param vars
	 * @throws DMLRuntimeException
	 */
	public static void rUpdateStatistics( Hop hop, LocalVariableMap vars ) 
		throws DMLRuntimeException
	{
		if( hop.getVisited() == VisitStatus.DONE )
			return;

		if( hop.getInput() != null )
			for( Hop c : hop.getInput() )
				rUpdateStatistics(c, vars);	
		
		//update statistics for transient reads according to current statistics
		//(with awareness not to override persistent reads to an existing name)
		if(     hop instanceof DataOp 
			&& ((DataOp)hop).getDataOpType() != DataOpTypes.PERSISTENTREAD )
		{
			DataOp d = (DataOp) hop;
			String varName = d.getName();
			if( vars.keySet().contains( varName ) )
			{
				Data dat = vars.get(varName);
				if( dat instanceof MatrixObject )
				{
					MatrixObject mo = (MatrixObject) dat;
					d.setDim1(mo.getNumRows());
					d.setDim2(mo.getNumColumns());
					d.setNnz(mo.getNnz());
				}
			}
		}
		//special case for persistent reads with unknown size (read-after-write)
		else if( hop instanceof DataOp 
				&& ((DataOp)hop).getDataOpType() == DataOpTypes.PERSISTENTREAD
				&& !hop.dimsKnown() && ((DataOp)hop).getInputFormatType()!=FileFormatTypes.CSV )
		{
			//update hop with read meta data
			DataOp dop = (DataOp) hop; 
			tryReadMetaDataFileMatrixCharacteristics(dop);
		}
		//update size expression for rand/seq according to symbol table entries
		else if ( hop instanceof DataGenOp )
		{
			DataGenOp d = (DataGenOp) hop;
			HashMap<String,Integer> params = d.getParamIndexMap();
			if (   d.getOp() == DataGenMethod.RAND || d.getOp()==DataGenMethod.SINIT 
				|| d.getOp() == DataGenMethod.SAMPLE ) 
			{
				int ix1 = params.get(DataExpression.RAND_ROWS);
				int ix2 = params.get(DataExpression.RAND_COLS);
				//update rows/cols by evaluating simple expression of literals, nrow, ncol, scalars, binaryops
				d.refreshRowsParameterInformation(d.getInput().get(ix1), vars);
				d.refreshColsParameterInformation(d.getInput().get(ix2), vars);
			} 
			else if ( d.getOp() == DataGenMethod.SEQ ) 
			{
				int ix1 = params.get(Statement.SEQ_FROM);
				int ix2 = params.get(Statement.SEQ_TO);
				int ix3 = params.get(Statement.SEQ_INCR);
				double from = d.computeBoundsInformation(d.getInput().get(ix1), vars);
				double to = d.computeBoundsInformation(d.getInput().get(ix2), vars);
				double incr = d.computeBoundsInformation(d.getInput().get(ix3), vars);
				
				//special case increment 
				Hop input3 = d.getInput().get(ix3);
				if ( input3 instanceof BinaryOp && ((BinaryOp)input3).getOp() == Hop.OpOp2.SEQINCR 
					&& from!=Double.MAX_VALUE && to!=Double.MAX_VALUE ) 
				{
					incr =( from >= to )? -1.0 : 1.0;
				}
				
				if ( from!=Double.MAX_VALUE && to!=Double.MAX_VALUE && incr!=Double.MAX_VALUE ) {
					d.setDim1( 1 + (long)Math.floor((to-from)/incr) );
					d.setDim2( 1 );
					d.setIncrementValue( incr );
				}
			}
			else {
				throw new DMLRuntimeException("Unexpected data generation method: " + d.getOp());
			}
		}
		//update size expression for reshape according to symbol table entries
		else if (    hop instanceof ReorgOp 
				 && ((ReorgOp)(hop)).getOp()==Hop.ReOrgOp.RESHAPE )
		{
			ReorgOp d = (ReorgOp) hop;
			d.refreshRowsParameterInformation(d.getInput().get(1), vars);
			d.refreshColsParameterInformation(d.getInput().get(2), vars);
		}
		//update size expression for indexing according to symbol table entries
		else if( hop instanceof IndexingOp )
		{
			IndexingOp iop = (IndexingOp)hop;
			Hop input2 = iop.getInput().get(1); //inpRowL
			Hop input3 = iop.getInput().get(2); //inpRowU
			Hop input4 = iop.getInput().get(3); //inpColL
			Hop input5 = iop.getInput().get(4); //inpColU
			double rl = iop.computeBoundsInformation(input2, vars);
			double ru = iop.computeBoundsInformation(input3, vars);
			double cl = iop.computeBoundsInformation(input4, vars);
			double cu = iop.computeBoundsInformation(input5, vars);
			if( rl!=Double.MAX_VALUE && ru!=Double.MAX_VALUE )
				iop.setDim1( (long)(ru-rl+1) );
			if( cl!=Double.MAX_VALUE && cu!=Double.MAX_VALUE )
				iop.setDim2( (long)(cu-cl+1) );
		}
		
		//propagate statistics along inner nodes of DAG
		hop.refreshSizeInformation();
		
		hop.setVisited(VisitStatus.DONE);
	}

	/**
	 * 
	 * @param hop
	 * @param vars
	 * @throws DMLRuntimeException
	 */
	public static void rReplaceLiterals( Hop hop, LocalVariableMap vars ) 
		throws DMLRuntimeException
	{
		if( hop.getVisited() == VisitStatus.DONE )
			return;

		if( hop.getInput() != null )
			for( int i=0; i<hop.getInput().size(); i++ )
			{
				Hop c = hop.getInput().get(i);
				
				//scalar read - literal replacement
				if( c instanceof DataOp && ((DataOp)c).getDataOpType() != DataOpTypes.PERSISTENTREAD 
					&& c.getDataType()==DataType.SCALAR )
				{
					Data dat = vars.get(c.getName());
					if( dat != null ) //required for selective constant propagation
					{
						ScalarObject sdat = (ScalarObject)dat;
						Hop literal = null;
						switch( sdat.getValueType() ) {
							case INT:
								literal = new LiteralOp(String.valueOf(sdat.getLongValue()), sdat.getLongValue());		
								break;
							case DOUBLE:
								literal = new LiteralOp(String.valueOf(sdat.getDoubleValue()), sdat.getDoubleValue());		
								break;						
							case BOOLEAN:
								literal = new LiteralOp(String.valueOf(sdat.getBooleanValue()), sdat.getBooleanValue());		
								break;
							default:	
								//otherwise: do nothing
						}
						
						//replace on demand 
						if( literal != null )
						{
							HopRewriteUtils.removeChildReference(hop, c);
							HopRewriteUtils.addChildReference(hop, literal, i);
						}
					}
				}
				//as.double/as.integer/as.boolean over scalar read - literal replacement
				else if( c instanceof UnaryOp && (((UnaryOp)c).getOp() == OpOp1.CAST_AS_DOUBLE
					|| ((UnaryOp)c).getOp() == OpOp1.CAST_AS_INT || ((UnaryOp)c).getOp() == OpOp1.CAST_AS_BOOLEAN )	
						&& c.getInput().get(0) instanceof DataOp && c.getDataType()==DataType.SCALAR )
				{
					Data dat = vars.get(c.getInput().get(0).getName());
					if( dat != null ) //required for selective constant propagation
					{
						ScalarObject sdat = (ScalarObject)dat;
						UnaryOp cast = (UnaryOp) c;
						Hop literal = null;
						switch( cast.getOp() ) {
							case CAST_AS_INT:
								literal = new LiteralOp(String.valueOf(sdat.getLongValue()), sdat.getLongValue());		
								break;
							case CAST_AS_DOUBLE:
								literal = new LiteralOp(String.valueOf(sdat.getDoubleValue()), sdat.getDoubleValue());		
								break;						
							case CAST_AS_BOOLEAN:
								literal = new LiteralOp(String.valueOf(sdat.getBooleanValue()), sdat.getBooleanValue());		
								break;
							default:	
								//otherwise: do nothing
						}
						
						//replace on demand 
						if( literal != null )
						{
							HopRewriteUtils.removeChildReference(hop, c);
							HopRewriteUtils.addChildReference(hop, literal, i);
						}
					}
				}
				//as.scalar/matrix read - literal replacement
				else if( c instanceof UnaryOp && ((UnaryOp)c).getOp() == OpOp1.CAST_AS_SCALAR 
					&& c.getInput().get(0) instanceof DataOp )
				{
					Data dat = vars.get(c.getInput().get(0).getName());
					if( dat != null ) //required for selective constant propagation
					{
						//cast as scalar (see VariableCPInstruction)
						MatrixObject mo = (MatrixObject)dat;
						MatrixBlock mBlock = mo.acquireRead();
						if( mBlock.getNumRows()!=1 || mBlock.getNumColumns()!=1 )
							throw new DMLRuntimeException("Dimension mismatch - unable to cast matrix of dimension ("+mBlock.getNumRows()+" x "+mBlock.getNumColumns()+") to scalar.");
						double value = mBlock.getValue(0,0);
						mo.release();
						
						//literal substitution (always double)
						Hop literal = new LiteralOp(String.valueOf(value), value);
						
						//replace on demand 
						HopRewriteUtils.removeChildReference(hop, c);
						HopRewriteUtils.addChildReference(hop, literal, i);	
					}
				}
				//as.scalar/right indexing w/ literals/vars and matrix less than 10^6 cells
				else if( c instanceof UnaryOp && ((UnaryOp)c).getOp() == OpOp1.CAST_AS_SCALAR 
						&& c.getInput().get(0) instanceof IndexingOp )
				{
					IndexingOp rix = (IndexingOp)c.getInput().get(0);
					Hop data = rix.getInput().get(0);
					Hop rl = rix.getInput().get(1);
					Hop ru = rix.getInput().get(2);
					Hop cl = rix.getInput().get(3);
					Hop cu = rix.getInput().get(4);
					if( rix.dimsKnown() && rix.getDim1()==1 && rix.getDim2()==1
						&& data instanceof DataOp && vars.keySet().contains(data.getName())
						&& ((rl instanceof DataOp && vars.keySet().contains(rl.getName())) || rl instanceof LiteralOp) 
						&& ((ru instanceof DataOp && vars.keySet().contains(ru.getName())) || ru instanceof LiteralOp)
						&& ((cl instanceof DataOp && vars.keySet().contains(cl.getName())) || cl instanceof LiteralOp)
						&& ((cu instanceof DataOp && vars.keySet().contains(cu.getName())) || cu instanceof LiteralOp)
						&& data.dimsKnown() && data.getDim1()*data.getDim2()<1000000 ) //8MB
					{
						long rlvalue = getIntValueDataLiteral(rl, vars);
						long clvalue = getIntValueDataLiteral(cl, vars);
	
						MatrixObject mo = (MatrixObject)vars.get(data.getName());
						MatrixBlock mBlock = mo.acquireRead();
						double value = mBlock.getValue((int)rlvalue-1,(int)clvalue-1);
						mo.release();
						
						
						//literal substitution (always double)
						Hop literal = new LiteralOp(String.valueOf(value), value);
						
						//replace on demand 
						HopRewriteUtils.removeChildReference(hop, c);
						HopRewriteUtils.addChildReference(hop, literal, i);	
					
					}
				}
				//recursively process childs
				else 
				{
					rReplaceLiterals(c, vars);	
				}
			}
		
		hop.setVisited(VisitStatus.DONE);
	}
	
	/**
	 * 
	 * @param hop
	 * @param vars
	 * @return
	 * @throws DMLRuntimeException 
	 */
	public static long getIntValueDataLiteral(Hop hop, LocalVariableMap vars) 
		throws DMLRuntimeException
	{
		long value = -1;
		
		try 
		{
			if( hop instanceof LiteralOp )
			{
				value = HopRewriteUtils.getIntValue((LiteralOp)hop);
			}
			else
			{
				ScalarObject sdat = (ScalarObject) vars.get(hop.getName());
				value = sdat.getLongValue();
			}
		}
		catch(HopsException ex)
		{
			throw new DMLRuntimeException("Failed to get int value for literal replacement", ex);
		}
		
		return value;
	}
	
	/**
	 * 
	 * @param hop
	 * @param pid
	 */
	public static void rSetExecType( Hop hop, ExecType etype )
	{
		if( hop.getVisited() == VisitStatus.DONE )
			return;
		
		//update function names
		hop.setForcedExecType(etype);
		
		if( hop.getInput() != null )
			for( Hop c : hop.getInput() )
				rSetExecType(c, etype);
		
		hop.setVisited(VisitStatus.DONE);
	}
	

	/**
	 * Returns true iff (1) all instruction are reblock instructions and (2) all
	 * individual reblock operations fit in the current memory budget.
	 * 
	 * @param inst
	 * @param pb
	 * @return
	 * @throws DMLRuntimeException 
	 * @throws IOException 
	 */
	public static boolean checkCPReblock(MRJobInstruction inst, MatrixObject[] inputs) 
		throws DMLRuntimeException, IOException 
	{
		boolean ret = true;
		
		boolean localMode = InfrastructureAnalyzer.isLocalMode();
		
		//check only shuffle inst
		String rdInst = inst.getIv_randInstructions();
		String rrInst = inst.getIv_recordReaderInstructions();
		String mapInst = inst.getIv_instructionsInMapper();
		String aggInst = inst.getIv_aggInstructions();
		String otherInst = inst.getIv_otherInstructions();
		if(    (rdInst != null && rdInst.length()>0)
			|| (rrInst != null && rrInst.length()>0)
			|| (mapInst != null && mapInst.length()>0)
			|| (aggInst != null && aggInst.length()>0)
			|| (otherInst != null && otherInst.length()>0)  )
		{
			ret = false;
		}
		
		//check only reblock inst
		if( ret ) {
			String shuffleInst = inst.getIv_shuffleInstructions();
			String[] instParts = shuffleInst.split( Lop.INSTRUCTION_DELIMITOR );
			for( String rblk : instParts )
				if(    !InstructionUtils.getOpCode(rblk).equals(ReBlock.OPCODE) 
					&& !InstructionUtils.getOpCode(rblk).equals(CSVReBlock.OPCODE) )
				{
					ret = false;
					break;
				}
		}
		
		//check output empty blocks (for outputEmptyBlocks=false, a CP reblock can be 
		//counter-productive because any export from CP would reintroduce the empty blocks)
		if( ret ){
			String shuffleInst = inst.getIv_shuffleInstructions();
			String[] instParts = shuffleInst.split( Lop.INSTRUCTION_DELIMITOR );
			for( String rblk : instParts )
				if(    InstructionUtils.getOpCode(rblk).equals(ReBlock.OPCODE) 
				    && rblk.endsWith("false") ) //no output of empty blocks
				{
					ret = false;
					break;
				}
		}
		
		//check recompile memory budget
		if( ret ) {
			for( MatrixObject mo : inputs )
			{
				long rows = mo.getNumRows();
				long cols = mo.getNumColumns();
				
				// If the dimensions are unknown then reblock can not be recompiled into CP
				// Note: unknown dimensions at this point can only happen for CSV files.
				// however, we do a conservative check with the CSV filesize
				if ( rows == -1 || cols == -1 ) 
				{
					Path path = new Path(mo.getFileName());
					long size = MapReduceTool.getFilesizeOnHDFS(path);
					if( size > CP_CSV_REBLOCK_UNKNOWN_THRESHOLD_SIZE || CP_CSV_REBLOCK_UNKNOWN_THRESHOLD_SIZE > OptimizerUtils.getLocalMemBudget() )
					{
						ret = false;
						break;
					}			
				}
				//default case (known dimensions)
				else
				{
					long nnz = mo.getNnz();
					double sp = OptimizerUtils.getSparsity(rows, cols, nnz);
					double mem = MatrixBlock.estimateSizeInMemory(rows, cols, sp);			
					if(    !OptimizerUtils.isValidCPDimensions(rows, cols)
						|| !OptimizerUtils.isValidCPMatrixSize(rows, cols, sp)
						|| mem >= OptimizerUtils.getLocalMemBudget() ) 
					{
						ret = false;
						break;
					}
				}
			}
		}
		
		//check in-memory reblock size threshold (prevent long single-threaded text read)
		//NOTE: this does not apply to local mode because there text read single-threaded as well
		if( ret && !localMode ) {
			for( MatrixObject mo : inputs )
			{
				MatrixFormatMetaData iimd = (MatrixFormatMetaData) mo.getMetaData();
				if((   iimd.getInputInfo()==InputInfo.TextCellInputInfo
					|| iimd.getInputInfo()==InputInfo.MatrixMarketInputInfo
					|| iimd.getInputInfo()==InputInfo.CSVInputInfo
					|| iimd.getInputInfo()==InputInfo.BinaryCellInputInfo)
					&& !mo.isDirty() )
				{
					//get file size on hdfs (as indicator for estimated read time)
					Path path = new Path(mo.getFileName());
					long fileSize = MapReduceTool.getFilesizeOnHDFS(path);
					//compute cp reblock size threshold based on available parallelism
					long cpThreshold = CP_REBLOCK_THRESHOLD_SIZE * 
							           OptimizerUtils.getParallelTextReadParallelism();
					
					if( fileSize > cpThreshold ) {
						ret = false;
						break;
					}
				}
			}
		}
	
		return ret;
	}

	/**
	 * 
	 * @param inst
	 * @param updatedRandInst
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static boolean checkCPDataGen( MRJobInstruction inst, String updatedRandInst ) 
		throws DMLRuntimeException 
	{
		boolean ret = true;
		
		//check only shuffle inst
		String shuffleInst = inst.getIv_shuffleInstructions();
		String rrInst = inst.getIv_recordReaderInstructions();
		String mapInst = inst.getIv_instructionsInMapper();
		String aggInst = inst.getIv_aggInstructions();
		String otherInst = inst.getIv_otherInstructions();
		if(    (shuffleInst != null && shuffleInst.length()>0)
			|| (rrInst != null && rrInst.length()>0)
			|| (mapInst != null && mapInst.length()>0)
			|| (aggInst != null && aggInst.length()>0)
			|| (otherInst != null && otherInst.length()>0)  )
		{
			ret = false;
		}
		
		//check only rand inst
		if( ret ) {
			String[] instParts = updatedRandInst.split( Lop.INSTRUCTION_DELIMITOR );
			for( String lrandStr : instParts ) {
				if( InstructionUtils.getOpCode(lrandStr).equals(DataGen.RAND_OPCODE) )
				{
					//check recompile memory budget
					RandInstruction lrandInst = (RandInstruction) RandInstruction.parseInstruction(lrandStr);
					long rows = lrandInst.getRows();
					long cols = lrandInst.getCols();
					double sparsity = lrandInst.getSparsity();
					double mem = MatrixBlock.estimateSizeInMemory(rows, cols, sparsity);				
					if(    !OptimizerUtils.isValidCPDimensions(rows, cols)
						|| !OptimizerUtils.isValidCPMatrixSize(rows, cols, sparsity)	
						|| mem >= OptimizerUtils.getLocalMemBudget() )
					{
						ret = false;
						break;
					}
				}
				else if( InstructionUtils.getOpCode(lrandStr).equals(DataGen.SEQ_OPCODE) )
				{
					//check recompile memory budget
					//(don't account for sparsity because always dense)
					SeqInstruction lrandInst = (SeqInstruction) SeqInstruction.parseInstruction(lrandStr);
					long rows = lrandInst.getRows();
					long cols = lrandInst.getCols();
					double mem = MatrixBlock.estimateSizeInMemory(rows, cols, 1.0d);				
					if(    !OptimizerUtils.isValidCPDimensions(rows, cols)
					    || !OptimizerUtils.isValidCPMatrixSize(rows, cols, 1.0d)	
						|| mem >= OptimizerUtils.getLocalMemBudget() )
					{
						ret = false;
						break;
					}
				}
				else
				{
					ret = false;
					break;
				}
			}
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @param fname
	 * @return
	 * @throws DMLRuntimeException
	 */
	private static void tryReadMetaDataFileMatrixCharacteristics( DataOp dop )
		throws DMLRuntimeException
	{
		try
		{
			//get meta data filename
			String mtdname = DataExpression.getMTDFileName(dop.getFileName());
			
			JobConf job = ConfigurationManager.getCachedJobConf();
			FileSystem fs = FileSystem.get(job);
			Path path = new Path(mtdname);
			if( fs.exists(path) ){
				BufferedReader br = null;
				try
				{
					br = new BufferedReader(new InputStreamReader(fs.open(path)));
					JSONObject mtd = JSONObject.parse(br);
					
					DataType dt = DataType.valueOf(String.valueOf(mtd.get(DataExpression.DATATYPEPARAM)).toUpperCase());
					dop.setDataType(dt);
					dop.setValueType(ValueType.valueOf(String.valueOf(mtd.get(DataExpression.VALUETYPEPARAM)).toUpperCase()));
					dop.setDim1((dt==DataType.MATRIX)?Long.parseLong(mtd.get(DataExpression.READROWPARAM).toString()):0);
					dop.setDim2((dt==DataType.MATRIX)?Long.parseLong(mtd.get(DataExpression.READCOLPARAM).toString()):0);
				}
				finally {
					if( br != null )
						br.close();
				}
			}
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException(ex);
		}
	}
}
