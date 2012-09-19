package com.ibm.bi.dml.runtime.controlprogram.parfor.opt;

import java.util.ArrayList;
import java.util.HashMap;

import com.ibm.bi.dml.parser.ParForStatementBlock;
import com.ibm.bi.dml.runtime.controlprogram.ForProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.Program;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PExecMode;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PTaskPartitioner;
import com.ibm.bi.dml.runtime.controlprogram.parfor.ProgramConverter;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.OptNode.ExecType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.OptNode.NodeType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.OptNode.ParamType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.PerfTestTool.TestMeasure;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;

/**
 * Heuristic ParFor Optimizer (time: O(n)):
 * 
 * This optimizer applies a set of heuristic rewrites to a given hierarchy of programblocks and
 * instructions, where the root is a parfor program block. All rewrites are only applied
 * to parfor program blocks, while all other program blocks and instructions remain unchanged. The
 * rewrites are cost-based decisions in order to guarantee main memory and parallelization constraints 
 * and in order to account rules of thumb with regards to execution strategy and degree of parallelism. 
 * 
 * 
 * TODO guarantee memory constraints
 * TODO guarantee parallelism constraints  
 * TODO enhancement rewrite exec strategy (more fine-grained decision)
 *  
 * TODO taskpartitioner: in general range, yes but binary
 *  
 */
public class OptimizerHeuristic extends Optimizer
{
	public static final double EXEC_TIME_THRESHOLD = 120000; //in ms
	public static final double PAR_MEM_FACTOR      = 0.8; //% of jvm mem
	public static final double PAR_K_FACTOR        = 2.0; //3.0
	public static final double PAR_K_MR_FACTOR     = 2.0; //3.0;

	//problem and infrastructure properties
	private int _N    = -1; //problemsize
	private int _Nmax = -1; //max problemsize (including subproblems)
	private int _lk   = -1; //local par
	private int _lkmax = -1; //local max par
	private int _rnk  = -1; //remote num nodes
	private int _rk   = -1; //remote par
	private int _rkmax = -1; //remote max par
	
	//constraints
	@SuppressWarnings("unused")
	private int    _ck  = -1; //general par constraint 
	private double _cmx = -1; //general memory constraint
	
	

	@Override
	public CostModelType getCostModelType() 
	{
		return CostModelType.RUNTIME_METRICS;
	}

	@Override
	public PlanInputType getPlanInputType() 
	{
		return PlanInputType.RUNTIME_PLAN;
	}
	
	
	/**
	 * transformation-based heuristic optimization
	 * (no use of sb, direct change of pb)
	 */
	@Override
	public boolean optimize(ParForStatementBlock sb, ParForProgramBlock pb, OptTree plan, CostEstimator est) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		System.out.println("--- HEURISTIC OPTIMIZER -------");
		
		
		OptNode pn = plan.getRoot();
		
		_N     = Integer.parseInt(pn.getParam(ParamType.NUM_ITERATIONS)); 
		_Nmax  = pn.getMaxProblemSize(); 
		_lk    = InfrastructureAnalyzer.getLocalParallelism();
		_lkmax = (int) Math.ceil( PAR_K_FACTOR * _lk ); 
		_rnk   = InfrastructureAnalyzer.getRemoteParallelNodes();  
		_rk    = InfrastructureAnalyzer.getRemoteParallelMapTasks(); 
		_rkmax = (int) Math.ceil( PAR_K_MR_FACTOR * _rk ); 
		_ck    = plan.getCK(); 
		_cmx   = PAR_MEM_FACTOR * InfrastructureAnalyzer.getCmMax(); 
		
		// intial cost estimation
		pn.setSerialParFor(); //for basic mem consumption
		double T = est.getEstimate(TestMeasure.EXEC_TIME, pn);  
		double M = est.getEstimate(TestMeasure.MEMORY_USAGE, pn);
		System.out.println("HEURISTIC OPTIMIZER: exec time estimate = "+T+" ms");
		System.out.println("HEURISTIC OPTIMIZER: mem estimate = "+M+" bytes");
		
		// rewrite 1: execution strategy
		rewriteSetExecutionStategy( pn, T, M );

		//exec-type-specific rewrites
		if( pn.getExecType() == ExecType.MR )
		{
			// rewrite 2: nested parallelism (incl exec types)	
			if( _N >= _rnk )
				rewriteNestedParallelism( pn );
			
			// rewrite 3: determine parallelism
			rewriteSetDegreeOfParallelism( pn, M, true );
			
			// rewrite 4: task partitioning
			rewriteSetTaskPartitioner( pn, PTaskPartitioner.STATIC );
			rewriteSetTaskPartitioner( pn.getChilds().get(0), PTaskPartitioner.FACTORING );
		}
		else //if( pn.getExecType() == ExecType.CP )
		{
			// rewrite 3: determine parallelism
			rewriteSetDegreeOfParallelism( pn, M, false );
			
			// rewrite 4: task partitioning
			rewriteSetTaskPartitioner( pn, PTaskPartitioner.FACTORING );
			//rewriteSetTaskPartitioner( pn, PTaskPartitioner.FIXED );
		}	
		
		///////
		//Final rewrites for cleanup / minor improvements
		
		// rewrite 0: parfor(par=1) to for 
		rewriteRemoveNestedParallelism( pn );
		
		
		//info optimization result
		_numEvaluatedPlans = 1;
		return true;
	}
	
	private void rewriteSetExecutionStategy(OptNode n, double T, double M)
	{
		if(    n.isCPOnly()          //all instruction can be be executed in CP
 			&& T > EXEC_TIME_THRESHOLD        //total exec time is large enough to compensate latency
			&& _Nmax>=_lkmax && _N >= _rnk //problem large enough to exploit full parallelism  
			&& M <= _cmx                     ) //cp inst fit into mem per node
		{
			n.setExecType( ExecType.MR ); //remote parfor
		}
		else
		{
			n.setExecType( ExecType.CP ); //local parfor
		}
		
		//actual modification
		long id = n.getID();
		ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
		                             .getRuntimePlanMapping().getMappedObject(id);
		pfpb.setExecMode( (n.getExecType()==ExecType.CP)? PExecMode.LOCAL : PExecMode.REMOTE_MR );	
	}

	private void rewriteRemoveNestedParallelism(OptNode n) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		if( !n.isLeaf() )
		{
			for( OptNode sub : n.getChilds() )
			{
				if( sub.getNodeType() == NodeType.PARFOR && sub.getK() == 1 )
				{
					long id = sub.getID();
					ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
					                             .getRuntimePlanMapping().getMappedObject(id);
					
					//create for pb as replacement
					Program prog = pfpb.getProgram();
					ForProgramBlock fpb = ProgramConverter.createShallowCopyForProgramBlock(pfpb, prog);
					
					//replace parfor with for, and update objectmapping
					OptTreeConverter.replaceProgramBlock(n, sub, pfpb, fpb, true);
					
					//update node
					sub.setNodeType(NodeType.FOR);
					sub.setK(1);
				}
				
				rewriteRemoveNestedParallelism(sub);
			}
		}		
	}

	@SuppressWarnings("unchecked")
	private void rewriteNestedParallelism(OptNode n ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		//modify tree
		ArrayList<OptNode> tmpOld = n.getChilds();
		OptNode nest = new OptNode(NodeType.PARFOR, ExecType.CP);
		ArrayList<OptNode> tmpNew = new ArrayList<OptNode>();
		tmpNew.add(nest);
		n.setChilds(tmpNew);
		nest.setChilds(tmpOld);
		
		//modify rtprog
		long id = n.getID();
		ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
		                            .getRuntimePlanMapping().getMappedObject(id);
		ArrayList<ProgramBlock> tmpPBOld = pfpb.getChildBlocks();
		
		//create new program block structure and modify parameters (from, to, incr, types,)
		String[] iterVars = pfpb.getIterablePredicateVars(); //from, to stay original
		String[] iterVars2 = iterVars.clone();  //itervar, incr stay original
		int outIncr = (int)Math.ceil(((double)_N)/_rnk);
		iterVars[ 0 ] = ParForStatementBlock.INTERAL_FN_INDEX_ROW; // already checked for uniqueness in ParForStatementBlock
		iterVars[ 3 ] = String.valueOf(outIncr); 		
		iterVars2[ 1 ] = ParForStatementBlock.INTERAL_FN_INDEX_ROW; //sub start
		iterVars2[ 2 ] = null;
		HashMap<String,String> params = pfpb.getParForParams();
		HashMap<String,String> params2 = (HashMap<String,String>)params.clone();	
		ParForProgramBlock pfpb2 = new ParForProgramBlock(pfpb.getProgram(),iterVars2, params2);
		OptTreeConverter.getRuntimePlanMapping().putMapping(pfpb2, nest);
		
		ArrayList<ProgramBlock> tmpPBNew = new ArrayList<ProgramBlock>();
		tmpPBNew.add(pfpb2);
		pfpb.setChildBlocks(tmpPBNew);
		pfpb.setIterablePredicateVars(iterVars);
		pfpb.setIncrementInstructions(new ArrayList<Instruction>());
		pfpb.setExecMode(PExecMode.REMOTE_MR);
		pfpb2.setChildBlocks(tmpPBOld);
		pfpb2.setResultVariables(pfpb.getResultVariables());
		pfpb2.setFromInstructions(new ArrayList<Instruction>());
		pfpb2.setToInstructions(ProgramRecompiler.createNestedParallelismToInstructionSet( ParForStatementBlock.INTERAL_FN_INDEX_ROW, String.valueOf(outIncr-1) ));
		pfpb2.setIncrementInstructions(new ArrayList<Instruction>());
		pfpb2.setExecMode(PExecMode.LOCAL);
	}
	
	/**
	 * 
	 * @param n
	 * @param M
	 * @param kMax
	 * @param mMax  (per node)
	 * @param nested
	 */
	private void rewriteSetDegreeOfParallelism(OptNode n, double M, boolean nested) 
	{
		long id = n.getID();
		int kMax = (n.getExecType()==ExecType.CP) ? _lkmax : _rkmax;
		kMax = Math.min(kMax, (int)Math.floor(_cmx/M));//ensure mem constraints 
		
		ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
        						  .getRuntimePlanMapping().getMappedObject(id);
		
		if( nested ) // remote,local
		{
			pfpb.setDegreeOfParallelism(_rnk); //only nested if N>rnk
			n.setK(_rnk);
			
			OptNode n2 = n.getChilds().get(0);
			ParForProgramBlock pfpb2 = (ParForProgramBlock) OptTreeConverter
			                            .getRuntimePlanMapping().getMappedObject(n2.getID());
			
			int k2 = Math.min((int)Math.ceil(((double)_N)/_rnk), (int)Math.ceil(((double)kMax)/_rnk));
			pfpb2.setDegreeOfParallelism(k2);  
			n2.setK(k2);
			
			
			int tmpK = (_N<kMax)? _N : kMax;
			assignRemainingParallelism( n2,(int)Math.ceil(((double)(kMax-tmpK+1))/tmpK) );
		}
		else // local or remote
		{
			int tmpK = (_N<kMax)? _N : kMax;
			pfpb.setDegreeOfParallelism(tmpK);
			n.setK(tmpK);	
			assignRemainingParallelism( n,(int)Math.ceil(((double)(kMax-tmpK+1))/tmpK) ); //1 if tmpK=kMax, otherwise larger
		}
	}
	
	private void assignRemainingParallelism(OptNode n, int par) 
	{		
		ArrayList<OptNode> childs = n.getChilds();
		if( childs != null )
			for( OptNode c : childs )
			{
				if( par == 1 )
					c.setSerialParFor();
				else if( c.getNodeType() == NodeType.PARFOR )
				{
					int tmpN = Integer.parseInt(c.getParam(ParamType.NUM_ITERATIONS));
					int tmpK = (tmpN<par)? tmpN : par;
					long id = c.getID();
					c.setK(tmpK);
					ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
					  							.getRuntimePlanMapping().getMappedObject(id);
					pfpb.setDegreeOfParallelism(tmpK);
					assignRemainingParallelism(c,(int)Math.ceil(((double)(par-tmpK+1))/tmpK));
				}
				else
					assignRemainingParallelism(c, par);
			}
	}

	private void rewriteSetTaskPartitioner(OptNode n, PTaskPartitioner partitioner) 
	{
		if( n.getNodeType() != NodeType.PARFOR )
			System.out.println("Warning: Task partitioner can only be set for a ParFor node.");
		
		long id = n.getID();
		
		// modify rtprog
		ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
        							.getRuntimePlanMapping().getMappedObject(id);
		pfpb.setTaskPartitioner(partitioner);
		
		// modify plan
		n.addParam(ParamType.TASK_PARTITIONER, partitioner.toString());
	}

}
