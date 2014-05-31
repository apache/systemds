/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.lops.compile;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;

import com.ibm.bi.dml.conf.ConfigurationManager;
import com.ibm.bi.dml.hops.BinaryOp;
import com.ibm.bi.dml.hops.DataOp;
import com.ibm.bi.dml.hops.FunctionOp;
import com.ibm.bi.dml.hops.FunctionOp.FunctionType;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.IndexingOp;
import com.ibm.bi.dml.hops.LiteralOp;
import com.ibm.bi.dml.hops.MemoTable;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.hops.DataGenOp;
import com.ibm.bi.dml.hops.ReorgOp;
import com.ibm.bi.dml.hops.Hop.DataGenMethod;
import com.ibm.bi.dml.hops.Hop.DataOpTypes;
import com.ibm.bi.dml.hops.Hop.Kind;
import com.ibm.bi.dml.hops.Hop.VISIT_STATUS;
import com.ibm.bi.dml.hops.rewrite.HopRewriteUtils;
import com.ibm.bi.dml.lops.CSVReBlock;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.DataGen;
import com.ibm.bi.dml.lops.Lop;
import com.ibm.bi.dml.lops.LopsException;
import com.ibm.bi.dml.lops.ReBlock;
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
import com.ibm.bi.dml.runtime.controlprogram.CVProgramBlock;
//import com.ibm.bi.dml.runtime.controlprogram.ELProgramBlock;
//import com.ibm.bi.dml.runtime.controlprogram.ELUseProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.ForProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.FunctionProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.IfProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.LocalVariableMap;
import com.ibm.bi.dml.runtime.controlprogram.Program;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.WhileProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.controlprogram.parfor.ProgramConverter;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.OptTreeConverter;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.MRJobInstruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.Data;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.FunctionCallCPInstruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.ScalarObject;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.RandInstruction;
import com.ibm.bi.dml.runtime.instructions.MRInstructions.SeqInstruction;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.MatrixFormatMetaData;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;

/**
 * 
 */
public class Recompiler 
{	
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final Log LOG = LogFactory.getLog(Recompiler.class.getName());
	
	//max threshold for in-memory reblock of text input [in bytes]
	//reason: single-threaded text read at 20MB/s, 1GB input -> 50s (should exploit parallelism)
	private static final long CP_REBLOCK_THRESHOLD_SIZE = 1024*1024*1024; 
	private static final double CP_CSV_REBLOCK_FILESIZE_RATIO = 0.4;
	
	/**
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
	public static ArrayList<Instruction> recompileHopsDag( ArrayList<Hop> hops, LocalVariableMap vars, long tid ) 
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

			// update statistics if unknown
			Hop.resetVisitStatus(hops);
			for( Hop hopRoot : hops )
				rUpdateStatistics( hopRoot, vars );
			Hop.resetVisitStatus(hops);
			MemoTable memo = new MemoTable();
			for( Hop hopRoot : hops )
				hopRoot.refreshMemEstimates(memo); 
						
			// construct lops			
			Dag<Lop> dag = new Dag<Lop>();
			for( Hop hopRoot : hops )
			{
				Lop lops = hopRoot.constructLops();
				lops.addToDag(dag);	
			}		
			
			// construct instructions
			newInst = dag.getJobs(ConfigurationManager.getConfig());			
		}
		
		// replace thread ids in new instructions
		if( tid != 0 ) //only in parfor context
			newInst = ProgramConverter.createDeepCopyInstructionSet(newInst, tid, -1, null, null, false, false);
		
		return newInst;
	}

	/**
	 * Note: This overloaded method is required for predicate instructions because
	 * they have only a single hops DAG and we need to synchronize on the original 
	 * (shared) hops object. Hence, we cannot create any wrapper arraylist for each
	 * recompilation - this would result in race conditions for concurrent recompilation 
	 * in a parfor body. 	
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
	public static ArrayList<Instruction> recompileHopsDag( Hop hops, LocalVariableMap vars, long tid ) 
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

			// update statistics if unknown
			hops.resetVisitStatus();
			rUpdateStatistics( hops, vars );
			hops.resetVisitStatus();
			hops.refreshMemEstimates(new MemoTable()); 		
			
			// construct lops
			Dag<Lop> dag = new Dag<Lop>();
			Lop lops = hops.constructLops();
			lops.addToDag(dag);		
			
			// construct instructions
			newInst = dag.getJobs(ConfigurationManager.getConfig());
		}
		
		// replace thread ids in new instructions
		if( tid != 0 ) //only in parfor context
			newInst = ProgramConverter.createDeepCopyInstructionSet(newInst, tid, -1, null, null, false, false);
		
		return newInst;
	}
	
	/**
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
	public static ArrayList<Instruction> recompileHopsDag2Forced( ArrayList<Hop> hops, long tid, ExecType et ) 
		throws DMLRuntimeException, HopsException, LopsException, DMLUnsupportedOperationException, IOException
	{
		ArrayList<Instruction> newInst = null;

		//long begin = System.nanoTime();
		synchronized( hops ) //need for synchronization as we do temp changes in shared hops/lops
		{	
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
			for( Hop hopRoot : hops )
			{
				Lop lops = hopRoot.constructLops();
				lops.addToDag(dag);
			}		

			// construct instructions
			newInst = dag.getJobs(ConfigurationManager.getConfig());
		}
		
		// replace thread ids in new instructions
		if( tid != 0 ) //only in parfor context
			newInst = ProgramConverter.createDeepCopyInstructionSet(newInst, tid, -1, null, null, false, false);
		
		//recompileTime += (System.nanoTime()-begin);
		//recompileCount++;
		return newInst;
	}

	/**
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

		//long begin = System.nanoTime();
		synchronized( hops ) //need for synchronization as we do temp changes in shared hops/lops
		{	
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
			
			// construct instructions
			newInst = dag.getJobs(ConfigurationManager.getConfig());
		}
		
		// replace thread ids in new instructions
		if( tid != 0 ) //only in parfor context
			newInst = ProgramConverter.createDeepCopyInstructionSet(newInst, tid, -1, null, null, false, false);
		
		//recompileTime += (System.nanoTime()-begin);
		//recompileCount++;
		return newInst;
	}

	/**
	 * 
	 * @param pbs
	 * @param vars
	 * @param tid
	 * @throws DMLRuntimeException 
	 */
	public static void recompileProgramBlockHierarchy( ArrayList<ProgramBlock> pbs, LocalVariableMap vars, long tid ) 
		throws DMLRuntimeException
	{
		try 
		{
			synchronized( pbs )
			{
				for( ProgramBlock pb : pbs )
					rRecompileProgramBlock(pb, vars, tid);
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
	 * 
	 * @param insts
	 * @return
	 */
	@Deprecated
	public static boolean containsNonRecompileInstructions( ArrayList<Instruction> insts )
	{
		boolean ret = false;
		
		for( Instruction inst : insts )
		{
			//function call instuctions because those are currently generated outside
			//hops/lops generation
			if( inst instanceof FunctionCallCPInstruction ) {
				ret = true;
				break;
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
		throws CloneNotSupportedException 
	{
		ArrayList<Hop> ret = new ArrayList<Hop>();
		
		//note: need memo table over all independent DAGs in order to 
		//account for shared transient reads (otherwise more instructions generated)
		HashMap<Long, Hop> memo = new HashMap<Long, Hop>(); //orig ID, new clone
		for( Hop hopRoot : hops )
			ret.add(rDeepCopyHopsDag(hopRoot, memo));
		
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
		throws CloneNotSupportedException 
	{
		HashMap<Long, Hop> memo = new HashMap<Long, Hop>(); //orig ID, new clone
		return rDeepCopyHopsDag(hops, memo);
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
		if( hop.get_visited() == VISIT_STATUS.DONE )
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
		
		hop.set_visited(VISIT_STATUS.DONE);
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
	private static void rRecompileProgramBlock( ProgramBlock pb, LocalVariableMap vars, long tid ) 
		throws HopsException, DMLRuntimeException, LopsException, DMLUnsupportedOperationException, IOException
	{
		if (pb instanceof WhileProgramBlock)
		{
			WhileProgramBlock tmp = (WhileProgramBlock)pb;
			removeUpdatedScalars(vars, tmp.getStatementBlock());
			for (ProgramBlock pb2 : tmp.getChildBlocks())
				rRecompileProgramBlock(pb2, vars, tid);
			removeUpdatedScalars(vars, tmp.getStatementBlock());
		}
		else if (pb instanceof IfProgramBlock)
		{
			IfProgramBlock tmp = (IfProgramBlock)pb;	
			for( ProgramBlock pb2 : tmp.getChildBlocksIfBody() )
				rRecompileProgramBlock(pb2, vars, tid);
			for( ProgramBlock pb2 : tmp.getChildBlocksElseBody() )
				rRecompileProgramBlock(pb2, vars, tid);
			removeUpdatedScalars(vars, tmp.getStatementBlock());
		}
		else if (pb instanceof ForProgramBlock) //includes ParFORProgramBlock
		{ 
			ForProgramBlock tmp = (ForProgramBlock)pb;	
			removeUpdatedScalars(vars, tmp.getStatementBlock());
			for( ProgramBlock pb2 : tmp.getChildBlocks() )
				rRecompileProgramBlock(pb2, vars, tid);
			removeUpdatedScalars(vars, tmp.getStatementBlock());
		}		
		else if (  pb instanceof FunctionProgramBlock //includes ExternalFunctionProgramBlock and ExternalFunctionProgramBlockCP
			    || pb instanceof CVProgramBlock
				//|| pb instanceof ELProgramBlock
				//|| pb instanceof ELUseProgramBlock
				)
		{
			//do nothing
		}
		else 
		{	
			StatementBlock sb = pb.getStatementBlock();
			ArrayList<Instruction> tmp = pb.getInstructions();

			if(	sb != null 
				&& Recompiler.requiresRecompilation( sb.get_hops() ) 
				/*&& !Recompiler.containsNonRecompileInstructions(tmp)*/ )
			{
				tmp = Recompiler.recompileHopsDag(sb.get_hops(), vars, tid);
				pb.setInstructions( tmp );
				
				//propagate stats across hops (should be executed on clone of vars)
				Recompiler.extractDAGOutputStatistics(sb.get_hops(), vars);
			}
			
			removeUpdatedScalars(vars, sb);			
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
		else if	( pb instanceof CVProgramBlock
				//|| pb instanceof ELProgramBlock
				//|| pb instanceof ELUseProgramBlock
				)
		{
			//do nothing
		}
		else 
		{	
			StatementBlock sb = pb.getStatementBlock();
			
			//recompile hops dag to CP (opt: don't recompile if CP and no MR inst)
			if(	sb != null && !(et==ExecType.CP && !OptTreeConverter.containsMRJobInstruction(pb, true)) )
			{
				ArrayList<Instruction> tmp = pb.getInstructions();
				tmp = Recompiler.recompileHopsDag2Forced(sb.get_hops(), tid, et);
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
						String fKey = fnamespace+Program.KEY_DELIM+fname;
						
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
			//&& hop.get_dim1()>0 && hop.get_dim2()>0  ) //matrix with known dims 
		{
			String varName = hop.get_name();
			if( !vars.keySet().contains(varName) || overwrite ) //not existing so far
			{
				//extract matrix sizes for size propagation
				if( hop.get_dataType()==DataType.MATRIX )
				{
					MatrixObject mo = new MatrixObject(ValueType.DOUBLE, null);
					MatrixCharacteristics mc = new MatrixCharacteristics( 
												hop.get_dim1(), hop.get_dim2(), 
												DMLTranslator.DMLBlockSize, DMLTranslator.DMLBlockSize,
												hop.getNnz());
					MatrixFormatMetaData meta = new MatrixFormatMetaData(mc,null,null);
					mo.setMetaData(meta);	
					vars.put(varName, mo);
				}
				//extract scalar constants for second constant propagation
				else if( hop.get_dataType()==DataType.SCALAR )
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
						String dopvarname = dop.get_name();
						if( dop.isRead() && vars.keySet().contains(dopvarname) )
						{
							ScalarObject constant = (ScalarObject) vars.get(dopvarname);
							vars.put(varName, constant); //no clone because constant
						}
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
					if( OptimizerUtils.estimateSizeExactSparsity(mc.get_rows(), mc.get_cols(), (mc.getNonZeros()>0)?((double)mc.getNonZeros())/mc.get_rows()/mc.get_cols():1.0)	
					    < OptimizerUtils.estimateSize(hop.get_dim1(), hop.get_dim2(), 1.0d) )
					{
						//update statistics if necessary
						mc.setDimension(hop.get_dim1(), hop.get_dim2());
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
		if( hop.get_visited() == VISIT_STATUS.DONE )
			return ret;
		
		if( hop.getInput() != null )
			for( Hop c : hop.getInput() )
			{
				ret |= rRequiresRecompile(c);
				if( ret ) break; // early abort
			}
		
		hop.set_visited(VISIT_STATUS.DONE);
		
		return ret;
	}
	
	/**
	 * 
	 * @param hop
	 */
	public static void rClearLops( Hop hop )
	{
		if( hop.get_visited() == VISIT_STATUS.DONE )
			return;
		
		//clear all relevant lops to allow for recompilation
		if( hop instanceof LiteralOp )
		{
			//for literal ops, we just clear parents because always constant
			if( hop.get_lops() != null )
				hop.get_lops().getOutputs().clear();	
		}
		else //GENERAL CASE
		{
			hop.set_lops(null);
			if( hop.getInput() != null )
				for( Hop c : hop.getInput() )
					rClearLops(c);
		}
		
		hop.set_visited(VISIT_STATUS.DONE);
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
		if( hop.get_visited() == VISIT_STATUS.DONE )
			return;

		if( hop.getInput() != null )
			for( Hop c : hop.getInput() )
				rUpdateStatistics(c, vars);	
		
		//update statitics for transient reads according to current statistics
		//(with awareness not to override persistent reads to an existing name)
		if(     hop instanceof DataOp 
			&& ((DataOp)hop).get_dataop() != DataOpTypes.PERSISTENTREAD )
		{
			DataOp d = (DataOp) hop;
			String varName = d.get_name();
			if( vars.keySet().contains( varName ) )
			{
				Data dat = vars.get(varName);
				if( dat instanceof MatrixObject )
				{
					MatrixObject mo = (MatrixObject) dat;
					d.set_dim1(mo.getNumRows());
					d.set_dim2(mo.getNumColumns());
					d.setNnz(mo.getNnz());
				}
			}
		}
		else if ( hop instanceof DataGenOp )
		{
			DataGenOp d = (DataGenOp) hop;
			HashMap<String,Integer> params = d.getParamIndexMap();
			if ( d.getDataGenMethod() == DataGenMethod.RAND ) {
				int ix1 = params.get(DataExpression.RAND_ROWS);
				int ix2 = params.get(DataExpression.RAND_COLS);
				//update rows/cols by evaluating simple expression of literals, nrow, ncol, scalars, binaryops
				d.refreshRowsParameterInformation(d.getInput().get(ix1), vars);
				d.refreshColsParameterInformation(d.getInput().get(ix2), vars);
			} 
			else if ( d.getDataGenMethod() == DataGenMethod.SEQ ) {
				int ix1 = params.get(Statement.SEQ_FROM);
				int ix2 = params.get(Statement.SEQ_TO);
				int ix3 = params.get(Statement.SEQ_INCR);
				double from = d.computeBoundsInformation(d.getInput().get(ix1), vars);
				double to = d.computeBoundsInformation(d.getInput().get(ix2), vars);
				double incr = d.computeBoundsInformation(d.getInput().get(ix3), vars);
				
				//special case increment 
				Hop input3 = d.getInput().get(ix3);
				if ( input3.getKind() == Kind.BinaryOp && ((BinaryOp)input3).getOp() == Hop.OpOp2.SEQINCR 
					&& from!=Double.MAX_VALUE && to!=Double.MAX_VALUE ) 
				{
					incr =( from >= to )? -1.0 : 1.0;
				}
				
				if ( from!=Double.MAX_VALUE && to!=Double.MAX_VALUE && incr!=Double.MAX_VALUE ) {
					d.set_dim1( 1 + (long)Math.floor((to-from)/incr) );
					d.set_dim2( 1 );
					d.setIncrementValue( incr );
				}
			}
			else {
				throw new DMLRuntimeException("Unexpect data generation method: " + d.getDataGenMethod());
			}
		}
		else if (    hop instanceof ReorgOp 
				 && ((ReorgOp)(hop)).getOp()==Hop.ReOrgOp.RESHAPE )
		{
			ReorgOp d = (ReorgOp) hop;
			d.refreshRowsParameterInformation(d.getInput().get(1), vars);
			d.refreshColsParameterInformation(d.getInput().get(2), vars);
		}
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
				iop.set_dim1( (long)(ru-rl+1) );
			if( cl!=Double.MAX_VALUE && cu!=Double.MAX_VALUE )
				iop.set_dim2( (long)(cu-cl+1) );
		}
		
		
		hop.refreshSizeInformation();
		
		hop.set_visited(VISIT_STATUS.DONE);
	}
	
	
	/**
	 * 
	 * @param hop
	 * @param pid
	 */
	public static void rSetExecType( Hop hop, ExecType etype )
	{
		if( hop.get_visited() == VISIT_STATUS.DONE )
			return;
		
		//update function names
		hop.setForcedExecType(etype);
		
		if( hop.getInput() != null )
			for( Hop c : hop.getInput() )
				rSetExecType(c, etype);
		
		hop.set_visited(VISIT_STATUS.DONE);
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
				if( !InstructionUtils.getOpCode(rblk).equals(ReBlock.OPCODE) && !InstructionUtils.getOpCode(rblk).equals(CSVReBlock.OPCODE) )
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
					/* TODO for this feature we need csv read with unknown size 
					JobConf job = ConfigurationManager.getCachedJobConf();
					FileSystem fs = FileSystem.get(job);
					FileStatus fstatus = fs.getFileStatus(new Path(mo.getFileName()));
					if( fstatus.getLen() > CP_CSV_REBLOCK_FILESIZE_RATIO * OptimizerUtils.getLocalMemBudget() )
					{
						ret = false;
						break;
					}			
					*/
					
					ret = false;
					break;
				}
				//default case (known dimensions)
				else
				{
					long nnz = mo.getNnz();
					double sp = OptimizerUtils.getSparsity(rows, cols, nnz);
					double mem = MatrixBlock.estimateSizeInMemory(rows, cols, sp);			
					if( mem >= OptimizerUtils.getLocalMemBudget() )
					{
						ret = false;
						break;
					}
				}
			}
		}
		
		//check in-memory reblock size threshold
		//(prevent long single-threaded text read)
		if( ret ) {
			for( MatrixObject mo : inputs )
			{
				MatrixFormatMetaData iimd = (MatrixFormatMetaData) mo.getMetaData();
				if((   iimd.getInputInfo()==InputInfo.TextCellInputInfo
					|| iimd.getInputInfo()==InputInfo.MatrixMarketInputInfo
					|| iimd.getInputInfo()==InputInfo.CSVInputInfo)
					&& !mo.isDirty() )
				{
					JobConf job = ConfigurationManager.getCachedJobConf();
					FileSystem fs = FileSystem.get(job);
					FileStatus fstatus = fs.getFileStatus(new Path(mo.getFileName()));
					if( fstatus.getLen() > CP_REBLOCK_THRESHOLD_SIZE )
					{
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
					long rows = lrandInst.rows;
					long cols = lrandInst.cols;
					double sparsity = lrandInst.sparsity;
					double mem = MatrixBlock.estimateSizeInMemory(rows, cols, sparsity);				
					if( mem >= OptimizerUtils.getLocalMemBudget() )
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
					long rows = lrandInst.rows;
					long cols = lrandInst.cols;
					double mem = MatrixBlock.estimateSizeInMemory(rows, cols, 1.0d);				
					if( mem >= OptimizerUtils.getLocalMemBudget() )
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
}
