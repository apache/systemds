package com.ibm.bi.dml.lops.compile;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import com.ibm.bi.dml.hops.DataOp;
import com.ibm.bi.dml.hops.Hops;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.hops.RandOp;
import com.ibm.bi.dml.hops.Hops.VISIT_STATUS;
import com.ibm.bi.dml.lops.Lops;
import com.ibm.bi.dml.lops.ReBlock;
import com.ibm.bi.dml.parser.RandStatement;
import com.ibm.bi.dml.parser.StatementBlock;
import com.ibm.bi.dml.runtime.controlprogram.CVProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.ELProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.ELUseProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.ForProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.FunctionProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.IfProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.LocalVariableMap;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.WhileProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.controlprogram.parfor.ProgramConverter;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.ConfigurationManager;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.InstructionUtils;
import com.ibm.bi.dml.runtime.instructions.MRJobInstruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.Data;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.FunctionCallCPInstruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.ScalarObject;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlockDSM;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;
import com.ibm.bi.dml.utils.HopsException;
import com.ibm.bi.dml.utils.LopsException;

/**
 * 
 */
public class Recompiler 
{
	private static final Log LOG = LogFactory.getLog(Recompiler.class.getName());
	
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
	public static ArrayList<Instruction> recompileHopsDag( ArrayList<Hops> hops, LocalVariableMap vars, long tid ) 
		throws DMLRuntimeException, HopsException, LopsException, DMLUnsupportedOperationException, IOException
	{
		ArrayList<Instruction> newInst = null;

		synchronized( hops ) //need for synchronization as we do temp changes in shared hops/lops
		{	
			LOG.debug ("\n**************** Optimizer (Recompile) *************\nMemory Budget = " + OptimizerUtils.toMB(Hops.getMemBudget(true)) + " MB");
			
			// clear existing lops
			for( Hops hopRoot : hops )
			{
				hopRoot.resetVisitStatus();
				rClearLops( hopRoot );
			}

			// update statistics if unknown
			for( Hops hopRoot : hops )
			{
				hopRoot.resetVisitStatus();
				rUpdateStatistics( hopRoot, vars );
				
				hopRoot.resetVisitStatus();
				hopRoot.refreshMemEstimates(); 
			}			
			
			// construct lops
			Dag<Lops> dag = new Dag<Lops>();
			for( Hops hopRoot : hops )
			{
				Lops lops = hopRoot.constructLops();
				lops.addToDag(dag);
			}		

			// construct instructions
			newInst = dag.getJobs(ConfigurationManager.getConfig());
		}
		
		// replace thread ids in new instructions
		if( tid != 0 ) //only in parfor context
			newInst = ProgramConverter.createDeepCopyInstructionSet(newInst, tid, -1, null, null, false);

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
	public static ArrayList<Instruction> recompileHopsDag( Hops hops, LocalVariableMap vars, long tid ) 
		throws DMLRuntimeException, HopsException, LopsException, DMLUnsupportedOperationException, IOException
	{
		ArrayList<Instruction> newInst = null;

		synchronized( hops ) //need for synchronization as we do temp changes in shared hops/lops
		{	
			LOG.debug ("\n**************** Optimizer (Recompile) *************\nMemory Budget = " + OptimizerUtils.toMB(Hops.getMemBudget(true)) + " MB");
			
			// clear existing lops
			hops.resetVisitStatus();
			rClearLops( hops );

			// update statistics if unknown
			hops.resetVisitStatus();
			rUpdateStatistics( hops, vars );
			hops.resetVisitStatus();
			hops.refreshMemEstimates(); 		
			
			// construct lops
			Dag<Lops> dag = new Dag<Lops>();
			Lops lops = hops.constructLops();
			lops.addToDag(dag);		

			// construct instructions
			newInst = dag.getJobs(ConfigurationManager.getConfig());
		}
		
		// replace thread ids in new instructions
		if( tid != 0 ) //only in parfor context
			newInst = ProgramConverter.createDeepCopyInstructionSet(newInst, tid, -1, null, null, false);

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
	 * 
	 * @param hops
	 * @return
	 */
	public static boolean requiresRecompilation( ArrayList<Hops> hops )
	{
		boolean ret = false;
		
		if( hops != null )
			for( Hops hop : hops )
			{
				ret |= rRequiresRecompile(hop);
				if( ret ) break; // early abort
			}
		
		return ret;
	}
	
	/**
	 * 
	 * @param hops
	 * @return
	 */
	public static boolean requiresRecompilation( Hops hops )
	{
		boolean ret = false;
		
		if( hops != null )
			ret = rRequiresRecompile(hops);
	
		return ret;
	}
	
	/**
	 * 
	 * @param insts
	 * @return
	 */
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
	private static void rRecompileProgramBlock( ProgramBlock pb, LocalVariableMap vars, long tid ) throws HopsException, DMLRuntimeException, LopsException, DMLUnsupportedOperationException, IOException
	{
		if (pb instanceof WhileProgramBlock)
		{
			WhileProgramBlock tmp = (WhileProgramBlock)pb;
			for (ProgramBlock pb2 : tmp.getChildBlocks())
				rRecompileProgramBlock(pb2, vars, tid);
		}
		else if (pb instanceof IfProgramBlock)
		{
			IfProgramBlock tmp = (IfProgramBlock)pb;	
			for( ProgramBlock pb2 : tmp.getChildBlocksIfBody() )
				rRecompileProgramBlock(pb2, vars, tid);
			for( ProgramBlock pb2 : tmp.getChildBlocksElseBody() )
				rRecompileProgramBlock(pb2, vars, tid);
		}
		else if (pb instanceof ForProgramBlock) //includes ParFORProgramBlock
		{ 
			ForProgramBlock tmp = (ForProgramBlock)pb;	
			for( ProgramBlock pb2 : tmp.getChildBlocks() )
				rRecompileProgramBlock(pb2, vars, tid);
		}		
		else if (  pb instanceof FunctionProgramBlock //includes ExternalFunctionProgramBlock and ExternalFunctionProgramBlockCP
			    || pb instanceof CVProgramBlock
				|| pb instanceof ELProgramBlock
				|| pb instanceof ELUseProgramBlock)
		{
			//do nothing
		}
		else 
		{	
			StatementBlock sb = pb.getStatementBlock();
			ArrayList<Instruction> tmp = pb.getInstructions();

			if(	sb != null 
				&& Recompiler.requiresRecompilation( sb.get_hops() ) 
				&& !Recompiler.containsNonRecompileInstructions(tmp) )
			{
				tmp = Recompiler.recompileHopsDag(sb.get_hops(), vars, tid);
				pb.setInstructions( tmp );
			}
		}
		
	}
	
	/**
	 * 
	 * @param hop
	 * @return
	 */
	private static boolean rRequiresRecompile( Hops hop )
	{	
		boolean ret = hop.requiresRecompile();
		
		if( hop.getInput() != null )
			for( Hops c : hop.getInput() )
			{
				ret |= rRequiresRecompile(c);
				if( ret ) break; // early abort
			}
		
		return ret;
	}
	
	/**
	 * 
	 * @param hop
	 */
	public static void rClearLops( Hops hop )
	{
		if( hop.get_visited() == VISIT_STATUS.DONE )
			return;
		
		//preserve Treads to prevent wrong rmfilevar instructions
		if( hop.getOpString().equals("TRead") )
			return; 
		
		//clear all relevant lops to allow for recompilation
		hop.set_lops(null);
		if( hop.getInput() != null )
			for( Hops c : hop.getInput() )
				rClearLops(c);
		
		hop.set_visited(VISIT_STATUS.DONE);
	}
	
	/**
	 * 
	 * @param hop
	 * @param vars
	 * @throws DMLRuntimeException
	 */
	private static void rUpdateStatistics( Hops hop, LocalVariableMap vars ) 
		throws DMLRuntimeException
	{
		if( hop.get_visited() == VISIT_STATUS.DONE )
			return;
		
		//output
		//System.out.println(" HOP STATISTICS "+hop.getOpString());
		//System.out.println("  name = "+hop.get_name());
		//System.out.println("  rows = "+hop.get_dim1());
		//System.out.println("  cols = "+hop.get_dim2());
		//System.out.println("  nnz = "+hop.getNnz());

		if( hop.getInput() != null )
			for( Hops c : hop.getInput() )
				rUpdateStatistics(c, vars);	
		
		if( hop instanceof DataOp )
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
		else if ( hop instanceof RandOp )
		{
			RandOp d = (RandOp) hop;
			HashMap<String,Integer> params = d.getParamIndexMap();
			int ix1 = params.get(RandStatement.RAND_ROWS);
			int ix2 = params.get(RandStatement.RAND_COLS);
			String name1 = d.getInput().get(ix1).get_name();
			String name2 = d.getInput().get(ix2).get_name();
			Data dat1 = vars.get(name1);
			Data dat2 = vars.get(name2);
			if( dat1!=null && dat1 instanceof ScalarObject )
				d.set_dim1( ((ScalarObject)dat1).getLongValue() );
			if( dat2!=null && dat2 instanceof ScalarObject )
				d.set_dim2( ((ScalarObject)dat2).getLongValue() );
		}
		
		hop.refreshSizeInformation();
		
		//if( hop.getExecType()==ExecType.MR )
		//	System.out.println("HOP with exec type MR after recompilation "+hop.getOpString());
		
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
	 */
	public static boolean checkCPReblock(MRJobInstruction inst, MatrixObject[] inputs) 
		throws DMLRuntimeException 
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
			String[] instParts = shuffleInst.split( Lops.INSTRUCTION_DELIMITOR );
			for( String rblk : instParts )
				if( !InstructionUtils.getOpCode(rblk).equals(ReBlock.OPCODE) )
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
				long nnz = mo.getNnz();
				double mem = MatrixBlockDSM.estimateSize(rows, cols, (nnz>0) ? ((double)nnz)/rows/cols : 1.0d);				
				if( mem >= Hops.getMemBudget(true) )
				{
					ret = false;
					break;
				}
			}
		}

		return ret;
	}
}
