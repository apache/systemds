package com.ibm.bi.dml.runtime.controlprogram.parfor.opt;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map.Entry;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.hops.OptimizerUtils;
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
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.Optimizer.CostModelType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.Stat;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.StatisticMonitor;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.Timing;
import com.ibm.bi.dml.runtime.util.UtilFunctions;
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
	public static final boolean LDEBUG = DMLScript.DEBUG || true;
	public static final double PAR_FACTOR_INFRASTRUCTURE = 1.0;
	
	//internal parameters
	private static final boolean ALLOW_RUNTIME_COSTMODEL = false;
	
	/**
	 * Called once per DML script (during program compile time) 
	 * in order to optimize all top-level parfor program blocks.
	 * 
	 * NOTE: currently note used at all.
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
		Timing time = null;
		if( LDEBUG )
		{
			System.out.println("ParFOR Opt: Running optimize all on DML program "+DMLScript.getUUID());
			time = new Timing();
			time.start();
		}
		
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
		
		if( LDEBUG )
			System.out.println("ParFOR Opt: Finished optimization for DML program "+DMLScript.getUUID()+" in "+time.stop()+"ms.");
	}

	/**
	 * Called once per top-level parfor (during runtime, on parfor execute)
	 * in order to optimize the specific parfor program block.
	 * 
	 * NOTE: this is the default way to invoke parfor optimizers.
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
		Timing time = null;		
		if( LDEBUG )
		{
			System.out.println("ParFOR Opt: Running optimization for ParFOR("+pb.getID()+")");
			time = new Timing();
			time.start();
		}
		
		//set max contraints if not specified
		int ck = UtilFunctions.toInt( Math.max( InfrastructureAnalyzer.getCkMaxCP(),
						                        InfrastructureAnalyzer.getCkMaxMR() ) * PAR_FACTOR_INFRASTRUCTURE );
		double cm = InfrastructureAnalyzer.getCmMax() * OptimizerUtils.MEM_UTIL_FACTOR; 
		
		//execute optimizer
		optimize( type, ck, cm, sb, pb );
		
		if( LDEBUG )
		{
			double timeVal = time.stop();
			
			System.out.println("ParFOR Opt: Finished optimization for PARFOR("+pb.getID()+") in "+timeVal+"ms.");
			if( ParForProgramBlock.MONITOR )
				StatisticMonitor.putPFStat( pb.getID() , Stat.OPT_T, timeVal);
		}
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
	private static void optimize( POptMode otype, int ck, double cm, ParForStatementBlock sb, ParForProgramBlock pb ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		Timing time = null;
		if( LDEBUG )
		{
			time = new Timing();
			time.start();
		}
		
		//create specified optimizer
		Optimizer opt = createOptimizer( otype );
		CostModelType cmtype = opt.getCostModelType();
		if( LDEBUG )
			System.out.println("ParFOR Opt: Created optimizer ("+otype+","+opt.getPlanInputType()+","+opt.getCostModelType()+") in "+time.stop()+"ms.");
		
		if( cmtype == CostModelType.RUNTIME_METRICS  //TODO remove check when perftesttool supported
			&& !ALLOW_RUNTIME_COSTMODEL )
		{
			throw new DMLRuntimeException("ParFOR Optimizer "+otype+" requires cost model "+cmtype+" that is not suported yet.");
		}
		
		//create opt tree
		OptTree tree = null;
		try
		{
			tree = OptTreeConverter.createOptTree(ck, cm, opt.getPlanInputType(), sb, pb); 
			if( LDEBUG )
			{
				System.out.println("ParFOR Opt: Created plan in "+time.stop()+"ms.");
				System.out.println(tree.explain(false));
			}
		}
		catch(Exception ex)
		{
			throw new DMLRuntimeException("Unable to create opt tree.", ex);
		}
		
		//create cost estimator
		CostEstimator est = createCostEstimator( cmtype );
		if( LDEBUG )
			System.out.println("ParFOR Opt: Created cost estimator ("+cmtype+") in "+time.stop()+"ms.");
		
		
		//core optimize
		opt.optimize( sb, pb, tree, est );
		if( LDEBUG )
		{
			System.out.println("ParFOR Opt: Optimized plan in "+time.stop()+"ms.");
			System.out.println(tree.explain(false));
		}
		
		//cleanup phase
		OptTreeConverter.clear();
		
		//monitor stats
		if( ParForProgramBlock.MONITOR )
		{
			StatisticMonitor.putPFStat( pb.getID() , Stat.OPT_OPTIMIZER, otype.ordinal());
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
				opt = new OptimizerDPEnum();
				break;
			case GREEDY:
				opt = new OptimizerGreedyEnum();
				break;
			case HEURISTIC:
				opt = new OptimizerHeuristic();
				break;
			case RULEBASED:
				opt = new OptimizerRuleBased();
				break;	
			default:
				throw new DMLRuntimeException("Undefined optimizer: '"+otype+"'.");
		}
		
		return opt;
	}

	/**
	 * 
	 * @param cmtype
	 * @return
	 * @throws DMLRuntimeException
	 */
	private static CostEstimator createCostEstimator( CostModelType cmtype ) 
		throws DMLRuntimeException
	{
		CostEstimator est = null;
		
		switch( cmtype )
		{
			case STATIC_MEM_METRIC:
				est = new CostEstimatorHops( OptTreeConverter.getAbstractPlanMapping() );
				break;
			case RUNTIME_METRICS:
				est = new CostEstimatorRuntime();
				break;
			default:
				throw new DMLRuntimeException("Undefined cost model type: '"+cmtype+"'.");
		}
		
		return est;
	}
}
