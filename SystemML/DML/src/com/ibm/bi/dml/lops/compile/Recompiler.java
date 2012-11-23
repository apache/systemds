package com.ibm.bi.dml.lops.compile;

import java.io.IOException;
import java.util.ArrayList;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

import com.ibm.bi.dml.hops.DataOp;
import com.ibm.bi.dml.hops.Hops;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.hops.Hops.VISIT_STATUS;
import com.ibm.bi.dml.lops.Lops;
import com.ibm.bi.dml.runtime.controlprogram.LocalVariableMap;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.controlprogram.parfor.ProgramConverter;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.ConfigurationManager;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.Data;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.FunctionCallCPInstruction;
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
		
		//Timing timeOut = new Timing();
		//timeOut.start();
		
		synchronized( hops ) //need for synchronization as we do temp changes in shared hops/lops
		{	
			LOG.debug ("\n**************** Optimizer (Recompile) *************\nMemory Budget = " + OptimizerUtils.toMB(Hops.getMemBudget(true)) + " MB");
			
			
			//Timing time = new Timing();
			//time.start();
			
			// clear existing lops
			for( Hops hopRoot : hops )
			{
				hopRoot.resetVisitStatus();
				rClearLops( hopRoot );
			}
			
			//System.out.println("Clear existing lops in "+time.stop()+"ms");
			
			// update statistics if unknown
			for( Hops hopRoot : hops )
			{
				hopRoot.resetVisitStatus();
				rUpdateStatistics( hopRoot, vars );
				
				hopRoot.resetVisitStatus();
				hopRoot.refreshMemEstimates(); 
			}			
			//System.out.println("Update hop statistics in "+time.stop()+"ms");
			
			// construct lops
			Dag<Lops> dag = new Dag<Lops>();
			for( Hops hopRoot : hops )
			{
				Lops lops = hopRoot.constructLops();
				lops.addToDag(dag);
			}		
			//System.out.println("Construct lops in "+time.stop()+"ms");
			
			// construct instructions
			newInst = dag.getJobs(ConfigurationManager.getConfig());
			//System.out.println("Construct instructions in "+time.stop()+"ms");
		}
		
		// replace thread ids in new instructions
		if( tid != 0 ) //only in parfor context
			newInst = ProgramConverter.createDeepCopyInstructionSet(newInst, tid, -1, null);
		
		//System.out.println("Program block recompiled in "+timeOut.stop()+"ms.");
		
		return newInst;
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
		
		hop.refreshSizeInformation();
		
		//if( hop.getExecType()==ExecType.MR )
		//	System.out.println("HOP with exec type MR after recompilation "+hop.getOpString());
		
		hop.set_visited(VISIT_STATUS.DONE);
	}
}
