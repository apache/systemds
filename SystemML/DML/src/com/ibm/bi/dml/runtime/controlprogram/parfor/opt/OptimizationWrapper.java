package com.ibm.bi.dml.runtime.controlprogram.parfor.opt;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map.Entry;

import com.ibm.bi.dml.parser.DMLProgram;
import com.ibm.bi.dml.parser.ForStatement;
import com.ibm.bi.dml.parser.ForStatementBlock;
import com.ibm.bi.dml.parser.IfStatement;
import com.ibm.bi.dml.parser.IfStatementBlock;
import com.ibm.bi.dml.parser.ParForStatementBlock;
import com.ibm.bi.dml.parser.StatementBlock;
import com.ibm.bi.dml.parser.WhileStatement;
import com.ibm.bi.dml.parser.WhileStatementBlock;
import com.ibm.bi.dml.runtime.controlprogram.ForProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.FunctionProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.IfProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.Program;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.WhileProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.POptMode;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.Stat;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.StatisticMonitor;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.Timing;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;
import com.ibm.bi.dml.utils.LanguageException;


/**
 * Wrapper to ParFOR cost estimation and optimizer. This is intended to be the 
 * only public access to the optimizer package.
 * 
 * NOTE: There are two main alternatives for invocation of this OptimizationWrapper:
 * (1) During compilation (after creating rtprog), (2) on execute of all top-level ParFOR PBs.
 * We decided to use (2) (and carry the SBs during execution) due to the following advantages
 *   - Known Statistics: problem size of top-level parfor known, in general, less unknown statistics
 *   - No Overhead: preventing overhead for non-parfor scripts (finding top-level parfors)
 *   - Simplicity: no need of finding top-level parfors 
 * 
 */
public class OptimizationWrapper 
{
	/**
	 * 
	 * 
	 * @param prog
	 * @param rtprog
	 * @throws DMLRuntimeException 
	 * @throws LanguageException 
	 * @throws DMLUnsupportedOperationException 
	 */
	public static void optimize(DMLProgram prog, Program rtprog) 
		throws DMLRuntimeException, LanguageException, DMLUnsupportedOperationException 
	{
		//init internal structures 
		HashMap<Long, ParForStatementBlock> sbs = new HashMap<Long, ParForStatementBlock>();
		HashMap<Long, ParForProgramBlock> pbs = new HashMap<Long, ParForProgramBlock>();	
		
		//find all top-level paror pbs
		findParForProgramBlocks(prog, rtprog, sbs, pbs);
		
		//optimize each top-level parfor pb independently
		for( Entry<Long, ParForProgramBlock> entry : pbs.entrySet() )
		{
			long key = entry.getKey();
			ParForStatementBlock sb = sbs.get(key);
			ParForProgramBlock pb = entry.getValue();
			
			//optimize (and implicit exchange)
			POptMode type = pb.getOptimizationMode(); //known to be >0
			optimize( type, sb, pb );
		}		
	}

	/**
	 * 
	 * 
	 * @param type
	 * @param sb
	 * @param pb
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException 
	 */
	public static void optimize( POptMode type, ParForStatementBlock sb, ParForProgramBlock pb ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		Timing time = new Timing();
		time.start();
		
		System.out.println("ParFOR: run optimizer for ParFOR("+pb.getID()+")");
		
		
		//set max contraints if not specified
		int ck = 2 * Math.max( InfrastructureAnalyzer.getCkMaxCP(),
							   InfrastructureAnalyzer.getCkMaxMR() );
		double cm = InfrastructureAnalyzer.getCmMax();
		
		//execute optimizer
		optimize( type, ck, cm, sb, pb );
		
		
		System.out.println("INFO: PARFOR("+pb.getID()+"): optimization finished in "+time.stop()+"ms.");
	}
	
	
	/**
	 * 
	 * @param type
	 * @param ck
	 * @param cm
	 * @param sb
	 * @param pb
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException 
	 * @throws  
	 */
	private static void optimize( POptMode type, int ck, double cm, ParForStatementBlock sb, ParForProgramBlock pb ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		Timing time = null;
		if( ParForProgramBlock.MONITOR )
		{
			time = new Timing();
			time.start();
		}
		
		//create opt tree
		OptTree tree = OptTreeConverter.createOptTree(ck, cm, pb); 
		//if( DMLScript.DEBUG )
			System.out.println(tree.explain(false));
		
		
		//optimize
		Optimizer opt = createOptimizer( type );
		opt.optimize( sb, pb, tree );
		//if( DMLScript.DEBUG )
			System.out.println(tree.explain(false));
		
		//cleanup phase
		OptTreeConverter.clear();
		
		if( ParForProgramBlock.MONITOR )
		{
			System.out.println("Optimization finished in "+time.stop()+" ms."); 
			
			StatisticMonitor.putPFStat( pb.getID() , Stat.OPT_T, time.stop());
			StatisticMonitor.putPFStat( pb.getID() , Stat.OPT_OPTIMIZER, type.ordinal());
			StatisticMonitor.putPFStat( pb.getID() , Stat.OPT_NUMTPLANS, opt.getNumTotalPlans());
			StatisticMonitor.putPFStat( pb.getID() , Stat.OPT_NUMEPLANS, opt.getNumEvaluatedPlans());
		}
	}

	/**
	 * 
	 * @param prog
	 * @param rtprog
	 * @throws LanguageException 
	 */
	private static void findParForProgramBlocks( DMLProgram prog, Program rtprog, 
			HashMap<Long, ParForStatementBlock> sbs, HashMap<Long, ParForProgramBlock> pbs ) 
		throws LanguageException
	{
		//handle function program blocks
		HashMap<String,FunctionProgramBlock> fpbs = rtprog.getFunctionProgramBlocks();
		for( Entry<String, FunctionProgramBlock> entry : fpbs.entrySet() )
		{
			String[] keypart = entry.getKey().split( Program.KEY_DELIM );
			String namespace = keypart[0];
			String name      = keypart[1]; 
			
			ProgramBlock pb = entry.getValue();
			StatementBlock sb = prog.getFunctionStatementBlock(namespace, name);
			
			//recursive find 
			rfindParForProgramBlocks(sb, pb, sbs, pbs);	
		}
		
		//handle actual program blocks
		ArrayList<ProgramBlock> tpbs = rtprog.getProgramBlocks();
		for( int i=0; i<tpbs.size(); i++ )
		{
			ProgramBlock pb = tpbs.get(i);
			StatementBlock sb = prog.getStatementBlock(i);
			
			//recursive find
			rfindParForProgramBlocks(sb, pb, sbs, pbs);
		}	
	}
	
	/**
	 * 
	 * @param sb
	 * @param pb
	 */
	private static void rfindParForProgramBlocks( StatementBlock sb, ProgramBlock pb,
			HashMap<Long, ParForStatementBlock> sbs, HashMap<Long, ParForProgramBlock> pbs )
	{
		if( pb instanceof ParForProgramBlock  ) 
		{
			//put top-level parfor into map, but no recursion
			ParForProgramBlock pfpb = (ParForProgramBlock) pb;
			ParForStatementBlock pfsb = (ParForStatementBlock) sb;
			
			//if( DMLScript.DEBUG )
				System.out.println("ParFOR: found ParForProgramBlock with POptMode="+pfpb.getOptimizationMode().toString());
			
			if( pfpb.getOptimizationMode() != POptMode.NONE )
			{
				//register programblock tree for optimization
				long pfid = pfpb.getID();
				pbs.put(pfid, pfpb);
				sbs.put(pfid, pfsb);
			}
		}
		else if( pb instanceof ForProgramBlock )
		{
			//recursive find
			ArrayList<ProgramBlock> fpbs = ((ForProgramBlock) pb).getChildBlocks();
			ArrayList<StatementBlock> fsbs = ((ForStatement)((ForStatementBlock) sb).getStatement(0)).getBody();
			for( int i=0;  i< fpbs.size(); i++ )
				rfindParForProgramBlocks(fsbs.get(i), fpbs.get(i), sbs, pbs);
		}
		else if( pb instanceof WhileProgramBlock )
		{
			//recursive find
			ArrayList<ProgramBlock> wpbs = ((WhileProgramBlock) pb).getChildBlocks();
			ArrayList<StatementBlock> wsbs = ((WhileStatement)((WhileStatementBlock) sb).getStatement(0)).getBody();
			for( int i=0;  i< wpbs.size(); i++ )
				rfindParForProgramBlocks(wsbs.get(i), wpbs.get(i), sbs, pbs);	
		}
		else if( pb instanceof IfProgramBlock  )
		{
			//recursive find
			IfProgramBlock ifpb = (IfProgramBlock) pb;
			IfStatement ifs = (IfStatement) ((IfStatementBlock) sb).getStatement(0);			
			ArrayList<ProgramBlock> ipbs1 = ifpb.getChildBlocksIfBody();
			ArrayList<ProgramBlock> ipbs2 = ifpb.getChildBlocksElseBody();
			ArrayList<StatementBlock> isbs1 = ifs.getIfBody();
			ArrayList<StatementBlock> isbs2 = ifs.getElseBody();			
			for( int i=0;  i< ipbs1.size(); i++ )
				rfindParForProgramBlocks(isbs1.get(i), ipbs1.get(i), sbs, pbs);				
			for( int i=0;  i< ipbs2.size(); i++ )
				rfindParForProgramBlocks(isbs2.get(i), ipbs2.get(i), sbs, pbs);								
		}
	}
	
	/**
	 * 
	 * @param otype
	 * @return
	 * @throws DMLRuntimeException
	 */
	private static Optimizer createOptimizer( POptMode otype ) 
		throws DMLRuntimeException
	{
		Optimizer opt = null;
		
		switch( otype )
		{
			case FULL_DP:
				//TODO: implement dynamic programming optimizer
//				opt = new OptimizerDPEnum();
//				break;
			case GREEDY:
				opt = new OptimizerGreedyEnum();
				break;
			case HEURISTIC:
				opt = new OptimizerHeuristic();
				break;
			default:
				throw new DMLRuntimeException("Undefined optimizer: '"+otype+"'.");
		}
		
		return opt;
	}

}
