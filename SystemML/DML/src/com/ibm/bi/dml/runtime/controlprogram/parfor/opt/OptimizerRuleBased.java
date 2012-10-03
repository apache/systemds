package com.ibm.bi.dml.runtime.controlprogram.parfor.opt;

import java.util.ArrayList;
import java.util.HashMap;

import com.ibm.bi.dml.hops.Hops;
import com.ibm.bi.dml.hops.IndexingOp;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.parser.ParForStatementBlock;
import com.ibm.bi.dml.runtime.controlprogram.ForProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.LocalVariableMap;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.Program;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PDataPartitioner;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PExecMode;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PTaskPartitioner;
import com.ibm.bi.dml.runtime.controlprogram.parfor.ProgramConverter;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.OptNode.ExecType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.OptNode.NodeType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.OptNode.ParamType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.PerfTestTool.TestMeasure;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.MatrixObjectNew;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;

/**
 * Rule-Based ParFor Optimizer (time: O(n)):
 * 
 * Applied rule-based rewrites
 * - 1) rewrite set data partitioner
 * - 2) rewrite set execution strategy		 
 * - 3) rewrite nested parallelism 
 * - 4) rewrite set degree of parallelism
 * - 5) rewrite set task partitioner		 		 
 * - 6) remove unnecessary parfor 			 
 * - ( 7) blockwise partitioning )
 * 
 * TODO blockwise partitioning
 * TODO result partitioning 
 *  
 */
public class OptimizerRuleBased extends Optimizer
{
	public static final double PROB_SIZE_THRESHOLD = 100; //wrt # top-level iterations
	public static final boolean APPLY_REWRITE_NESTED_PARALLELISM = false;
	
	public static final double PAR_K_FACTOR        = OptimizationWrapper.PAR_FACTOR_INFRASTRUCTURE; 
	public static final double PAR_K_MR_FACTOR     = 2.0 * OptimizationWrapper.PAR_FACTOR_INFRASTRUCTURE; 

	//problem and infrastructure properties
	private int _N    = -1; //problemsize
	private int _Nmax = -1; //max problemsize (including subproblems)
	private int _lk   = -1; //local par
	private int _lkmaxCP = -1; //local max par (if only CP inst)
	private int _lkmaxMR = -1; //local max par (if also MR inst)
	private int _rnk  = -1; //remote num nodes
	private int _rk   = -1; //remote par
	private int _rkmax = -1; //remote max par
	private double _lm = -1; //general memory constraint
	private double _rm = -1; //global memory constraint

	@Override
	public CostModelType getCostModelType() 
	{
		return CostModelType.STATIC_MEM_METRIC;
	}


	@Override
	public PlanInputType getPlanInputType() 
	{
		return PlanInputType.ABSTRACT_PLAN;
	}
	
	
	/**
	 * Main optimization procedure.
	 * 
	 * Transformation-based heuristic (rule-based) optimization
	 * (no use of sb, direct change of pb).
	 */
	@Override
	public boolean optimize(ParForStatementBlock sb, ParForProgramBlock pb, OptTree plan, CostEstimator est) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		if( OptimizationWrapper.LDEBUG )
			System.out.println("--- RULEBASED OPTIMIZER -------");

		//ANALYZE infrastructure properties
		if( OptimizationWrapper.LDEBUG )
			System.out.println("RULEBASED OPTIMIZER: Analyze infrastructure properties." );
		
		OptNode pn = plan.getRoot();
		_N     = Integer.parseInt(pn.getParam(ParamType.NUM_ITERATIONS)); 
		_Nmax  = pn.getMaxProblemSize(); 
		_lk    = InfrastructureAnalyzer.getLocalParallelism();
		_lkmaxCP = (int) Math.ceil( PAR_K_FACTOR * _lk ); 
		_lkmaxMR = (int) Math.ceil( PAR_K_MR_FACTOR * _lk );
		_rnk   = InfrastructureAnalyzer.getRemoteParallelNodes();  
		_rk    = InfrastructureAnalyzer.getRemoteParallelMapTasks(); 
		_rkmax = (int) Math.ceil( PAR_K_FACTOR * _rk ); 
		_lm   = Hops.getMemBudget(true);
		_rm   = OptimizerUtils.MEM_UTIL_FACTOR * InfrastructureAnalyzer.getRemoteMaxMemory(); //Hops.getMemBudget(false); 
		
		if( OptimizationWrapper.LDEBUG )
			System.out.println("RULEBASED OPTIMIZER: Optimize with local_max_mem="+toMB(_lm)+" and remote_max_mem="+toMB(_rm)+")" );
		
		
		//ESTIMATE memory consumption 
		pn.setSerialParFor(); //for basic mem consumption 
		double M = est.getEstimate(TestMeasure.MEMORY_USAGE, pn);

		if( OptimizationWrapper.LDEBUG )
			System.out.println("RULEBASED OPTIMIZER: estimated mem (serial exec) M="+toMB(M) );
		
		//OPTIMIZE PARFOR PLAN
		
		// rewrite 1: data partitioning
		// (needs to be before exec type rewrite for taking recompilation into account)
		rewriteSetDataPartitioner( pn, pb.getVariables() );
		M = est.getEstimate(TestMeasure.MEMORY_USAGE, pn); //reestimate due to potential recompile in rewrite data partitioner
		
		// rewrite 2: execution strategy
		rewriteSetExecutionStategy( pn, M );

		//exec-type-specific rewrites
		if( pn.getExecType() == ExecType.MR )
		{
			// rewrite 3: nested parallelism (incl exec types)	
			boolean nested = rewriteNestedParallelism( pn, M );
			
			// rewrite 4: determine parallelism
			rewriteSetDegreeOfParallelism( pn, M, nested );
			
			// rewrite 5: task partitioning
			if( nested ){
				rewriteSetTaskPartitioner( pn, PTaskPartitioner.STATIC );
				rewriteSetTaskPartitioner( pn.getChilds().get(0), PTaskPartitioner.FACTORING );
			}
			else
				rewriteSetTaskPartitioner( pn, PTaskPartitioner.FACTORING );
		}
		else //if( pn.getExecType() == ExecType.CP )
		{
			// rewrite 4: determine parallelism
			rewriteSetDegreeOfParallelism( pn, M, false );
			
			// rewrite 5: task partitioning
			rewriteSetTaskPartitioner( pn, PTaskPartitioner.FACTORING );
		}	
		
		///////
		//Final rewrites for cleanup / minor improvements
		
		// rewrite 6: parfor (par=1) to for 
		rewriteRemoveUnnecessaryParFor( pn );
		
		//info optimization result
		_numEvaluatedPlans = 1;
		return true;
	}

	
	///////
	//REWRITE set data partitioner
	///
	
	/**
	 * 
	 * @param n
	 * @throws DMLRuntimeException 
	 */
	private void rewriteSetDataPartitioner(OptNode n, LocalVariableMap vars) 
		throws DMLRuntimeException
	{
		if( n.getNodeType() != NodeType.PARFOR )
			System.out.println("Warning: Data partitioner can only be set for a ParFor node.");
		
		//preparations
		long id = n.getID();
		Object[] o = OptTreeConverter.getAbstractPlanMapping().getMappedProg(id);
		ParForStatementBlock pfsb = (ParForStatementBlock) o[0];
		ParForProgramBlock pfpb = (ParForProgramBlock) o[1];
		
		//search for candidates
		ArrayList<String> cand = pfsb.getReadOnlyParentVars();
		HashMap<String, PDataPartitionFormat> cand2 = new HashMap<String, PDataPartitionFormat>();
		for( String c : cand )
		{
			PDataPartitionFormat dpf= pfsb.determineDataPartitionFormat( c );
			if( dpf != PDataPartitionFormat.NONE )
				cand2.put( c, dpf );
		}
		boolean apply = rFindDataPartitioningCandidates(n, cand2, vars);
		PDataPartitioner pdp = (apply)? PDataPartitioner.REMOTE_MR : PDataPartitioner.NONE;
		
		// modify rtprog 
		pfpb.setDataPartitioner( pdp );
		// modify plan
		n.addParam(ParamType.DATA_PARTITIONER, pdp.toString());
	
		if( OptimizationWrapper.LDEBUG )
			System.out.println("RULEBASED OPTIMIZER: rewrite 'set data partitioner' - result="+pdp.toString() );
	}
	
	/**
	 * 
	 * @param n
	 * @param cand
	 * @return
	 * @throws DMLRuntimeException 
	 */
	private boolean rFindDataPartitioningCandidates( OptNode n, HashMap<String, PDataPartitionFormat> cand, LocalVariableMap vars ) 
		throws DMLRuntimeException
	{
		boolean ret = false;

		if( !n.isLeaf() )
		{
			for( OptNode cn : n.getChilds() )
				ret |= rFindDataPartitioningCandidates( cn, cand, vars );
		}
		else if( n.getNodeType()== NodeType.HOP
			     && n.getParam(ParamType.OPSTRING).equals(IndexingOp.OPSTRING) 
			     && n.getExecType() == ExecType.MR )
		{
			Hops h = OptTreeConverter.getAbstractPlanMapping().getMappedHop(n.getID());
			String inMatrix = h.getInput().get(0).get_name();
			if( cand.containsKey(inMatrix) )
			{
				PDataPartitionFormat dpf = cand.get(inMatrix);
				double mnew = getNewMemoryEstimate( n, inMatrix, dpf, vars );
				if( mnew < _lm ) //apply rewrite if partitions fit into memory
				{
					n.setExecType(ExecType.CP);
					h.setMemEstimate( mnew );
					ret = true;
				}
			}
		}
		
		return ret;
	}
	
	/**
	 * TODO synchronize mem estimation with Indexing Hop
	 * 
	 * NOTE: Using the dimensions without sparsity is a conservative worst-case consideration.
	 * 
	 * @param n
	 * @param varName
	 * @param dpf
	 * @return
	 * @throws DMLRuntimeException 
	 */
	private double getNewMemoryEstimate( OptNode n, String varName, PDataPartitionFormat dpf, LocalVariableMap vars ) 
		throws DMLRuntimeException
	{
		double mem = -1;
		
		MatrixObjectNew mo = (MatrixObjectNew) vars.get( varName );
		
		//those are worst-case (dense) estimates
		switch( dpf )
		{
			case COLUMN_WISE:
				mem = mo.getNumRows() * 8; 
				break;
			case ROW_WISE:
				mem = mo.getNumColumns() * 8;
				break;
		}
		
		return mem;
	}
	

	///////
	//REWRITE set execution strategy
	///
	
	/**
	 * 
	 * 
	 * NOTES:
	 * 	- checking cm2 (min lJVM, rJVM) is sufficient because otherwise MR jobs generated anyway
	 * 
	 * @param n
	 * @param M
	 */
	private void rewriteSetExecutionStategy(OptNode n, double M)
	{
		//deciding on the execution strategy
		if(    n.isCPOnly()   //Required: all instruction can be be executed in CP
			&& M <= _rm     ) //Required: cp inst fit into JVM mem per node
		{
			//at this point all required conditions for REMOTE_MR given, now its an opt decision
			int cpk = (int) Math.min( _lk, Math.floor( _lm / M ) ); //estimated local exploited par  
			
			//MR if local par cannot be exploited due to mem constraints (this implies that we work on large data)
			if( cpk < _lk && cpk < _N && cpk < _rk )
			{
				n.setExecType( ExecType.MR ); //remote parfor
			}
			//MR if problem is large enough and remote parallelism is larger than local   
			else if( _lk < _N && _lk < _rk && (_N >= PROB_SIZE_THRESHOLD || _Nmax >= 10 * PROB_SIZE_THRESHOLD ) )
			{
				n.setExecType( ExecType.MR ); //remote parfor
			}
			//otherwise CP
			else
			{
				n.setExecType( ExecType.CP ); //local parfor	
			}			
		}
		else
		{
			n.setExecType( ExecType.CP ); //local parfor
		}
		
		//actual programblock modification
		long id = n.getID();
		ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
		                             .getAbstractPlanMapping().getMappedProg(id)[1];
		PExecMode mode = (n.getExecType()==ExecType.CP)? PExecMode.LOCAL : PExecMode.REMOTE_MR;
		pfpb.setExecMode( mode );	
		
		if( OptimizationWrapper.LDEBUG )
			System.out.println("RULEBASED OPTIMIZER: rewrite 'set execution strategy' - result="+mode );
	}


	///////
	//REWRITE nested parallelism
	///
	
	/**
	 * 
	 * @param n
	 * @param M
	 * @return
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	@SuppressWarnings("all")
	private boolean rewriteNestedParallelism(OptNode n, double M ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		boolean nested = false;
	
		if( APPLY_REWRITE_NESTED_PARALLELISM
			&& _N >= _rnk 					 // at least exploit all nodes
			&& !n.hasNestedParallelism()     // only for 1D problems, otherwise potentially bad load balance
			&& M * _lkmaxCP <= _rm  )        // only if we can exploit full local parallelism in the map task JVM memory
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
	                                    .getAbstractPlanMapping().getMappedProg(id)[1];
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
			OptTreeConverter.getAbstractPlanMapping().putProgMapping(null, pfpb2, nest);
			
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
		
			nested = true;
		}

		if( OptimizationWrapper.LDEBUG )
			System.out.println("RULEBASED OPTIMIZER: rewrite 'MR nested parallelism' - result="+nested );
		
		return nested;
	}

	
	///////
	//REWRITE set degree of parallelism
	///
		
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
		ExecType type = n.getExecType();
		long id = n.getID();
				
		//special handling for different exec models (CP, MR, MR nested)
		ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
										.getAbstractPlanMapping().getMappedProg(id)[1];
		
		if( type == ExecType.CP ) 
		{
			//determine local max parallelism constraint
			int kMax = -1;
			if( n.isCPOnly() )
				kMax = _lkmaxCP;
			else
				kMax = _lkmaxMR;
			
			//ensure local memory constraint
			kMax = Math.min( kMax, (int)Math.floor( _lm / M ) );
			if( kMax < 1 )
				kMax = 1;
			
			//distribute remaining parallelism 
			int tmpK = (_N<kMax)? _N : kMax;
			pfpb.setDegreeOfParallelism(tmpK);
			n.setK(tmpK);	
			rAssignRemainingParallelism( n,(int)Math.ceil(((double)(kMax-tmpK+1))/tmpK) ); //1 if tmpK=kMax, otherwise larger
		}
		else // ExecType.MR
		{
			int kMax = -1;
			if( nested )
			{
				//determine remote max parallelism constraint
				pfpb.setDegreeOfParallelism( _rnk ); //guaranteed <= _N (see nested)
				n.setK( _rnk );	
			
				kMax = _rkmax / _rnk; //per node (CP only inside)
			}
			else //not nested (default)
			{
				//determine remote max parallelism constraint
				int tmpK = (_N<_rk)? _N : _rk;
				pfpb.setDegreeOfParallelism(tmpK);
				n.setK(tmpK);	
				
				kMax = _rkmax / tmpK; //per node (CP only inside)
			}
			
			//ensure remote memory constraint
			kMax = Math.min( kMax, (int)Math.floor( _rm / M ) ); //guaranteed >= 1 (see exec strategy)
			if( kMax < 1 )
				kMax = 1;
				
			//distribute remaining parallelism
			rAssignRemainingParallelism( n, kMax ); 
		}		
		
		if( OptimizationWrapper.LDEBUG )
			System.out.println("RULEBASED OPTIMIZER: rewrite 'set degree of parallelism' - result=(see EXPLAIN)" );
	}
	
	/**
	 * TODO assign remaining parallelism according to mem in order 
	 *      to exploit higher degree of parallelism
	 *      (currently conservative because total max par determined 
	 *       by max mem operation of entire tree)
	 * 
	 * @param n
	 * @param par
	 */
	private void rAssignRemainingParallelism(OptNode n, int par) 
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
                                                  .getAbstractPlanMapping().getMappedProg(id)[1];
					pfpb.setDegreeOfParallelism(tmpK);
					rAssignRemainingParallelism(c,(int)Math.ceil(((double)(par-tmpK+1))/tmpK));
				}
				else
					rAssignRemainingParallelism(c, par);
			}
	}

	
	///////
	//REWRITE set task partitioner
	///
	
	/**
	 * 
	 * @param n
	 * @param partitioner
	 */
	private void rewriteSetTaskPartitioner(OptNode n, PTaskPartitioner partitioner) 
	{
		if( n.getNodeType() != NodeType.PARFOR )
			System.out.println("Warning: Task partitioner can only be set for a ParFor node.");
		
		long id = n.getID();
		
		// modify rtprog
		ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
                                     .getAbstractPlanMapping().getMappedProg(id)[1];
		pfpb.setTaskPartitioner(partitioner);
		
		// modify plan
		n.addParam(ParamType.TASK_PARTITIONER, partitioner.toString());
		
		if( OptimizationWrapper.LDEBUG )
			System.out.println("RULEBASED OPTIMIZER: rewrite 'set task partitioner' - result="+partitioner );
	}
	
	
	///////
	//REWRITE remove unnecessary parfor
	///
	
	/**
	 * 
	 * @param n
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	private void rewriteRemoveUnnecessaryParFor(OptNode n) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		int count = removeUnnecessaryParFor( n );
		
		if( OptimizationWrapper.LDEBUG )
			System.out.println("RULEBASED OPTIMIZER: rewrite 'remove unnecessary parfor' - result="+count );
	}
	
	private int removeUnnecessaryParFor( OptNode n ) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		int count = 0;
		
		if( !n.isLeaf() )
		{
			for( OptNode sub : n.getChilds() )
			{
				if( sub.getNodeType() == NodeType.PARFOR && sub.getK() == 1 )
				{
					long id = sub.getID();
					ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
                                                .getAbstractPlanMapping().getMappedProg(id)[1];
					
					//create for pb as replacement
					Program prog = pfpb.getProgram();
					ForProgramBlock fpb = ProgramConverter.createShallowCopyForProgramBlock(pfpb, prog);
					
					//replace parfor with for, and update objectmapping
					OptTreeConverter.replaceProgramBlock(n, sub, pfpb, fpb, false);
					
					//update node
					sub.setNodeType(NodeType.FOR);
					sub.setK(1);
					
					count++;
				}
				
				count += removeUnnecessaryParFor(sub);
			}
		}
		
		return count;
	}

	
	////////////////////////
	//   Helper methods   //
	////////////////////////
	
	public static String toMB( double inB )
	{
		return String.valueOf( (inB/(1024*1024)) ) + "MB";
		//return String.valueOf( (int)(inB/(1024*1024)) ) + "MB";
	}
}
