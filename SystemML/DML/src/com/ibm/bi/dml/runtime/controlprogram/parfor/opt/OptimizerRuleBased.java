package com.ibm.bi.dml.runtime.controlprogram.parfor.opt;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.hops.DataOp;
import com.ibm.bi.dml.hops.Hops;
import com.ibm.bi.dml.hops.IndexingOp;
import com.ibm.bi.dml.hops.LeftIndexingOp;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.lops.LopProperties;
import com.ibm.bi.dml.lops.compile.Recompiler;
import com.ibm.bi.dml.parser.ParForStatementBlock;
import com.ibm.bi.dml.parser.StatementBlock;
import com.ibm.bi.dml.runtime.controlprogram.ForProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.LocalVariableMap;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.Program;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PDataPartitioner;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PExecMode;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PResultMerge;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PTaskPartitioner;
import com.ibm.bi.dml.runtime.controlprogram.caching.MatrixObject;
import com.ibm.bi.dml.runtime.controlprogram.parfor.ProgramConverter;
import com.ibm.bi.dml.runtime.controlprogram.parfor.ResultMergeLocalFile;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.OptNode.ExecType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.OptNode.NodeType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.OptNode.ParamType;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.PerfTestTool.TestMeasure;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.Data;
import com.ibm.bi.dml.runtime.matrix.MatrixFormatMetaData;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.io.SparseRow;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;
import com.ibm.bi.dml.utils.HopsException;
import com.ibm.bi.dml.utils.LopsException;

/**
 * Rule-Based ParFor Optimizer (time: O(n)):
 * 
 * Applied rule-based rewrites
 * - 1) rewrite set data partitioner (incl. recompile RIX)
 * - 2) rewrite result partitioning (incl. recompile LIX)
 * - 3) rewrite set execution strategy
 * - 4) rewrite use data colocation		 
 * - 5) rewrite set partition replication factor 
 * - 6) rewrite use nested parallelism 
 * - 7) rewrite set degree of parallelism
 * - 8) rewrite set task partitioner
 * - 9) rewrite set result merge 		 		 
 * - 10) rewrite set recompile memory budget
 * - 11) remove unnecessary parfor		
 * 
 * 	 
 * 
 * 
 * TODO blockwise partitioning
 *  
 */
public class OptimizerRuleBased extends Optimizer
{
	public static final double PROB_SIZE_THRESHOLD_REMOTE = 100; //wrt # top-level iterations
	public static final double PROB_SIZE_THRESHOLD_PARTITIONING = 2; //wrt # top-level iterations
	public static final int MAX_REPLICATION_FACTOR_PARTITIONING = 4;    
	public static final boolean APPLY_REWRITE_NESTED_PARALLELISM = false;
	
	public static final double PAR_K_FACTOR        = OptimizationWrapper.PAR_FACTOR_INFRASTRUCTURE; 
	public static final double PAR_K_MR_FACTOR     = 1.0 * OptimizationWrapper.PAR_FACTOR_INFRASTRUCTURE; 
	
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
		LOG.debug("--- RULEBASED OPTIMIZER -------");

		//ANALYZE infrastructure properties
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
		
		LOG.debug("RULEBASED OPT: Optimize with local_max_mem="+toMB(_lm)+" and remote_max_mem="+toMB(_rm)+")" );
		
		
		//ESTIMATE memory consumption 
		pn.setSerialParFor(); //for basic mem consumption 
		double M = est.getEstimate(TestMeasure.MEMORY_USAGE, pn);
		LOG.debug("RULEBASED OPT: estimated mem (serial exec) M="+toMB(M) );
		
		//OPTIMIZE PARFOR PLAN
		
		// rewrite 1: data partitioning (incl. log. recompile RIX)
		rewriteSetDataPartitioner( pn, pb.getVariables() );
		M = est.getEstimate(TestMeasure.MEMORY_USAGE, pn); //reestimate
		
		// rewrite 2: rewrite result partitioning (incl. log/phy recompile LIX) 
		boolean flagLIX = rewriteSetResultPartitioning( pn, M );
		M = est.getEstimate(TestMeasure.MEMORY_USAGE, pn); //reestimate 
		
		// rewrite 3: execution strategy
		rewriteSetExecutionStategy( pn, M, flagLIX );
		
		//exec-type-specific rewrites
		if( pn.getExecType() == ExecType.MR )
		{
			// rewrite 4: data colocation
			rewriteDataColocation( pn, pb.getVariables() );
			
			// rewrite 5: rewrite set partition replication factor
			rewriteSetPartitionReplicationFactor( pn, pb.getVariables() );
			
			// rewrite 6: nested parallelism (incl exec types)	
			boolean flagNested = rewriteNestedParallelism( pn, M, flagLIX );
			
			// rewrite 7: determine parallelism
			rewriteSetDegreeOfParallelism( pn, M, flagNested );
			
			// rewrite 8: task partitioning 
			rewriteSetTaskPartitioner( pn, flagNested, flagLIX );
		}
		else //if( pn.getExecType() == ExecType.CP )
		{
			// rewrite 7: determine parallelism
			rewriteSetDegreeOfParallelism( pn, M, false );
			
			// rewrite 8: task partitioning
			rewriteSetTaskPartitioner( pn, false, false ); //flagLIX always false 
		}	
		
		//rewrite 9: set result merge
		rewriteSetResultMerge( pn, pb.getVariables() );
		
		//rewrite 10: set local recompile memory budget
		rewriteSetRecompileMemoryBudget( pn );
		
		///////
		//Final rewrites for cleanup / minor improvements
		
		// rewrite 11: parfor (par=1) to for 
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
			LOG.warn("RULEBASED OPT: Data partitioner can only be set for a ParFor node.");
		
		//preparations
		long id = n.getID();
		Object[] o = OptTreeConverter.getAbstractPlanMapping().getMappedProg(id);
		ParForStatementBlock pfsb = (ParForStatementBlock) o[0];
		ParForProgramBlock pfpb = (ParForProgramBlock) o[1];
		
		//search for candidates
		boolean apply = false;
		if(    DMLScript.rtplatform == RUNTIME_PLATFORM.HYBRID  //only if we are allowed to recompile
			&& (_N >= PROB_SIZE_THRESHOLD_PARTITIONING || _Nmax >= PROB_SIZE_THRESHOLD_PARTITIONING) ) //only if beneficial wrt problem size
		{
			ArrayList<String> cand = pfsb.getReadOnlyParentVars();
			HashMap<String, PDataPartitionFormat> cand2 = new HashMap<String, PDataPartitionFormat>();
			for( String c : cand )
			{
				PDataPartitionFormat dpf = pfsb.determineDataPartitionFormat( c );
				if( dpf != PDataPartitionFormat.NONE )
				{
					cand2.put( c, dpf );
					//System.out.println("Candidate "+c+": "+dpf);
				}
					
			}
			apply = rFindDataPartitioningCandidates(n, cand2, vars);
		}
		PDataPartitioner pdp = (apply)? PDataPartitioner.REMOTE_MR : PDataPartitioner.NONE;		
		//NOTE: since partitioning is only applied in case of MR index access, we assume a large
		//      matrix and hence always apply REMOTE_MR (the benefit for large matrices outweigths
		//      potentially unnecessary MR jobs for smaller matrices)
		
		// modify rtprog 
		pfpb.setDataPartitioner( pdp );
		// modify plan
		n.addParam(ParamType.DATA_PARTITIONER, pdp.toString());
	
		LOG.debug("RULEBASED OPT: rewrite 'set data partitioner' - result="+pdp.toString() );
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
				if( cn.getNodeType() != NodeType.FUNCCALL ) //prevent conflicts with aliases
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
				//NOTE: subsequent rewrites will still use the MR mem estimate
				//(guarded by subsequent operations that have at least the memory req of one partition)
				PDataPartitionFormat dpf = cand.get(inMatrix);
				double mnew = getNewRIXMemoryEstimate( n, inMatrix, dpf, vars );
				if( mnew < _lm ) //apply rewrite if partitions fit into memory
					n.setExecType(ExecType.CP);
				else
					n.setExecType(ExecType.CP); //CP_FILE, but hop still in MR 
				n.addParam(ParamType.DATA_PARTITION_FORMAT, dpf.toString());
				h.setMemEstimate( mnew ); //CP vs CP_FILE in ProgramRecompiler bases on mem_estimate
				ret = true;
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
	private double getNewRIXMemoryEstimate( OptNode n, String varName, PDataPartitionFormat dpf, LocalVariableMap vars ) 
		throws DMLRuntimeException
	{
		double mem = -1;
		
		MatrixObject mo = (MatrixObject) vars.get( varName );
		
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
	
	/**
	 * 
	 * @param mo
	 * @param dpf
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static LopProperties.ExecType getRIXExecType( MatrixObject mo, PDataPartitionFormat dpf ) 
		throws DMLRuntimeException
	{
		double mem = -1;
		switch( dpf )
		{
			case COLUMN_WISE:
				mem = mo.getNumRows() * 8; 
				break;
			case ROW_WISE:
				mem = mo.getNumColumns() * 8;
				break;
		}
		
		if( mem < Hops.getMemBudget(true) )
			return LopProperties.ExecType.CP;
		else
			return LopProperties.ExecType.CP_FILE;
	}
	
	///////
	//REWRITE set result partitioning
	///

	/**
	 * 
	 * @param n
	 * @throws DMLRuntimeException
	 */
	private boolean rewriteSetResultPartitioning(OptNode n, double M) 
		throws DMLRuntimeException
	{
		//preparations
		long id = n.getID();
		Object[] o = OptTreeConverter.getAbstractPlanMapping().getMappedProg(id);
		ParForProgramBlock pfpb = (ParForProgramBlock) o[1];
		
		//search for candidates
		Collection<OptNode> cand = n.getNodeList(ExecType.MR);
		
		//determine if applicable
		boolean apply =    M < _rm         //ops fit in remote memory budget
			            && cand.size() > 0 //at least one MR
		                && isResultPartitionableAll(cand,pfpb.getResultVariables(),pfpb.getVariables(), pfpb.getIterablePredicateVars()[0]); // check candidates
			
		//recompile LIX
		if( apply )
		{
			try
			{
				for(OptNode lix : cand)
					recompileLIX( lix );
			}
			catch(Exception ex)
			{
				throw new DMLRuntimeException("Unable to recompile LIX.", ex);
			}
		}
		
		LOG.debug("RULEBASED OPT: rewrite 'set result partitioning' - result="+apply );
	
		return apply;
	}
	
	private boolean isResultPartitionableAll( Collection<OptNode> nlist, ArrayList<String> resultVars, LocalVariableMap vars, String iterVarname ) 
		throws DMLRuntimeException
	{
		boolean ret = true;
		for( OptNode n : nlist )
		{
			ret &= isResultPartitionable(n, resultVars, vars, iterVarname);
			if(!ret) //early abort
				break;
		}
		
		return ret;
	}
	
	private boolean isResultPartitionable( OptNode n, ArrayList<String> resultVars, LocalVariableMap vars, String iterVarname ) 
		throws DMLRuntimeException
	{
		boolean ret = true;
		
		//check left indexing operator
		String opStr = n.getParam(ParamType.OPSTRING);
		if( opStr==null || !opStr.equals(LeftIndexingOp.OPSTRING) )
			ret = false;

		Hops h = null;
		Hops base = null;
		
		if( ret ) {
			h = OptTreeConverter.getAbstractPlanMapping().getMappedHop(n.getID());
			base = h.getInput().get(0);
			
			//check result variable
			if( !resultVars.contains(base.get_name()) )
				ret = false;
		}

		//check access pattern, memory budget
		if( ret ) {
			int dpf = 0;
			Hops inpRowL = h.getInput().get(2);
			Hops inpRowU = h.getInput().get(3);
			Hops inpColL = h.getInput().get(4);
			Hops inpColU = h.getInput().get(5);
			if( (inpRowL.get_name().equals(iterVarname) && inpRowU.get_name().equals(iterVarname)) )
				dpf = 1; //rowwise
			if( (inpColL.get_name().equals(iterVarname) && inpColU.get_name().equals(iterVarname)) )
				dpf = (dpf==0) ? 2 : 3; //colwise or cellwise
			
			if( dpf == 0 )
				ret = false;
			else
			{
				//check memory budget
				MatrixObject mo = (MatrixObject)vars.get(base.get_name());
				if( mo.getNnz() != 0 ) //0 or -1 valid because result var known during opt
					ret = false;
				
				double memTask1 = -1;
				int taskN = -1;
				switch(dpf) { //check tasksize = 1
					case 1:
						memTask1 = base.get_dim2()*8;
						taskN = (int) (_rm / memTask1); 
						break;
					case 2:
						memTask1 = base.get_dim1()*Math.min(SparseRow.initialCapacity, base.get_dim2())*8;
						taskN = (int) (_rm / (base.get_dim1()*8));
						break;
					case 3:
						memTask1 = Math.min(SparseRow.initialCapacity, base.get_dim2())*8;
						taskN = (int) (_rm / memTask1);
						break;	
				}

				if( memTask1>_rm )
					ret = false;
				else
					n.addParam(ParamType.TASK_SIZE, String.valueOf(taskN));
			}
		}
				
		//if(ret)
		//	System.out.println("isResultPartitioning: "+base.get_name()+" - "+ret);
		
		return ret;
	}
	
	private void recompileLIX( OptNode n ) 
		throws DMLRuntimeException, HopsException, LopsException, DMLUnsupportedOperationException, IOException
	{
		Hops h = OptTreeConverter.getAbstractPlanMapping().getMappedHop(n.getID());
		
		//set forced exec type
		h.setForcedExecType(LopProperties.ExecType.CP);
		n.setExecType(ExecType.CP);
		
		//recompile parent pb
		long pid = OptTreeConverter.getAbstractPlanMapping().getMappedParentID(n.getID());
		Object[] o = OptTreeConverter.getAbstractPlanMapping().getMappedProg(pid);
		StatementBlock sb = (StatementBlock) o[0];
		ProgramBlock pb = (ProgramBlock) o[1];
		
		//construct new instructions
		ArrayList<Instruction> newInst = Recompiler.recompileHopsDag(sb.get_hops(), pb.getVariables(), 0);
		pb.setInstructions( newInst );   
		
		//set new mem estimate (last, otherwise overwritten from recompile)
		h.setMemEstimate(_rm-1);
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
	private void rewriteSetExecutionStategy(OptNode n, double M, boolean flagLIX)
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
			else if( _lk < _N && _lk < _rk && (_N >= PROB_SIZE_THRESHOLD_REMOTE || _Nmax >= 10 * PROB_SIZE_THRESHOLD_REMOTE ) )
			{
				n.setExecType( ExecType.MR ); //remote parfor
			}
			//MR if necessary for LIX rewrite (LIX true iff cp only and rm valid)
			else if( flagLIX )
			{
				n.setExecType( ExecType.MR );  //remote parfor
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
		
		LOG.debug("RULEBASED OPT: rewrite 'set execution strategy' - result="+mode );
	}
	
	///////
	//REWRITE enable data colocation
	///

	/**
	 * 
	 * @param n
	 * @throws DMLRuntimeException 
	 */
	private void rewriteDataColocation( OptNode n, LocalVariableMap vars ) 
		throws DMLRuntimeException
	{
		// data colocation is beneficial if we have dp=REMOTE_MR, etype=REMOTE_MR
		// and there is at least one direct col-/row-wise access with the index variable
		// on the partitioned matrix
		boolean apply = false;
		String varname = null;
		ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
        							.getAbstractPlanMapping().getMappedProg(n.getID())[1];
		
		if(    n.getParam(ParamType.DATA_PARTITIONER).equals(PDataPartitioner.REMOTE_MR.toString())
			&& n.getExecType()==ExecType.MR )
		{
			//find all candidates matrices (at least one partitioned access via iterVar)
			HashSet<String> cand = new HashSet<String>();
			rFindDataColocationCandidates(n, cand, pfpb.getIterablePredicateVars()[0]);
			
			//select largest matrix for colocation (based on nnz to account for sparsity)
			long nnzMax = Long.MIN_VALUE;
			for( String c : cand ) {
				MatrixObject tmp = (MatrixObject)vars.get(c);
				long nnzTmp = tmp.getNnz();
				if( nnzTmp > nnzMax ) {
					nnzMax = nnzTmp;
					varname = c;
					apply = true;
				}
			}		
		}
		
		//modify the runtime plan (apply true if at least one candidate)
		if( apply )
			pfpb.enableColocatedPartitionedMatrix( varname );
		
		LOG.debug("RULEBASED OPT: rewrite 'enable data colocation' - result="+apply+((apply)?" ("+varname+")":"") );
	}
	
	/**
	 * 
	 * @param n
	 * @param cand
	 * @param iterVarname
	 * @return
	 * @throws DMLRuntimeException
	 */
	private void rFindDataColocationCandidates( OptNode n, HashSet<String> cand, String iterVarname ) 
		throws DMLRuntimeException
	{
		if( !n.isLeaf() )
		{
			for( OptNode cn : n.getChilds() )
				rFindDataColocationCandidates( cn, cand, iterVarname );
		}
		else if(    n.getNodeType()== NodeType.HOP
			     && n.getParam(ParamType.OPSTRING).equals(IndexingOp.OPSTRING)
			     && n.getParam(ParamType.DATA_PARTITION_FORMAT) != null )
		{
			PDataPartitionFormat dpf = PDataPartitionFormat.valueOf(n.getParam(ParamType.DATA_PARTITION_FORMAT));
			Hops h = OptTreeConverter.getAbstractPlanMapping().getMappedHop(n.getID());
			String inMatrix = h.getInput().get(0).get_name();
			String indexAccess = null;
			switch( dpf )
			{
				case ROW_WISE: //input 1 and 2 eq
					if( h.getInput().get(1) instanceof DataOp )
						indexAccess = h.getInput().get(1).get_name();
					break;
				case COLUMN_WISE: //input 3 and 4 eq
					if( h.getInput().get(3) instanceof DataOp )
						indexAccess = h.getInput().get(3).get_name();
					break;
			}
			
			if( indexAccess != null && indexAccess.equals(iterVarname) )
				cand.add( inMatrix );
		}
	}
	
	
	///////
	//REWRITE set partition replication factor
	///

	/**
	 * Increasing the partition replication factor is beneficial if partitions are
	 * read multiple times (e.g., in nested loops) because partitioning (done once)
	 * gets slightly slower but there is a higher probability for local access
	 * 
	 * NOTE: this rewrite requires 'set data partitioner' to be executed in order to
	 * leverage the partitioning information in the plan tree. 
	 *  
	 * @param n
	 * @throws DMLRuntimeException 
	 */
	private void rewriteSetPartitionReplicationFactor( OptNode n, LocalVariableMap vars ) 
		throws DMLRuntimeException
	{
		boolean apply = false;
		int replication = ParForProgramBlock.WRITE_REPLICATION_FACTOR;
		
		ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
        							.getAbstractPlanMapping().getMappedProg(n.getID())[1];
		
		if(    n.getExecType()==ExecType.MR
			&& n.getParam(ParamType.DATA_PARTITIONER).equals(PDataPartitioner.REMOTE_MR.toString())
		    && n.hasNestedParallelism(false) 
		    && n.hasNestedPartitionReads(false) )		
		{
			apply = true;
			replication = Math.max( ParForProgramBlock.WRITE_REPLICATION_FACTOR,
					                Math.min(_rnk, MAX_REPLICATION_FACTOR_PARTITIONING) );
		}
		
		//modify the runtime plan 
		if( apply )
			pfpb.setPartitionReplicationFactor( replication );
		
		LOG.debug("RULEBASED OPT: rewrite 'set partition replication factor' - result="+apply+((apply)?" ("+replication+")":"") );
	}
	
	
	///////
	//REWRITE enable nested parallelism
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
	private boolean rewriteNestedParallelism(OptNode n, double M, boolean flagLIX ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		boolean nested = false;
	
		if( APPLY_REWRITE_NESTED_PARALLELISM
			&& !flagLIX                      // if not applied left indexing rewrite	
			&& _N >= _rnk 					 // at least exploit all nodes
			&& !n.hasNestedParallelism(false)// only for 1D problems, otherwise potentially bad load balance
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

		LOG.debug("RULEBASED OPT: rewrite 'enable nested parallelism' - result="+nested );
		
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
	private void rewriteSetDegreeOfParallelism(OptNode n, double M, boolean flagNested) 
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
			if( flagNested )
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
		
		LOG.debug("RULEBASED OPT: rewrite 'set degree of parallelism' - result=(see EXPLAIN)" );
	}
	
	/**
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
	private void rewriteSetTaskPartitioner(OptNode pn, boolean flagNested, boolean flagLIX) 
	{
		//assertions (warnings of corrupt optimizer decisions)
		if( pn.getNodeType() != NodeType.PARFOR )
			LOG.warn("RULEBASED OPT: Task partitioner can only be set for a ParFor node.");
		if( flagNested && flagLIX )
			LOG.warn("RULEBASED OPT: Task partitioner decision has conflicting input from rewrites 'nested parallelism' and 'result partitioning'.");
		
		
		//set task partitioner
		if( flagNested )
		{
			setTaskPartitioner( pn, PTaskPartitioner.STATIC );
			setTaskPartitioner( pn.getChilds().get(0), PTaskPartitioner.FACTORING );
		}
		else if( flagLIX )
		{
			setTaskPartitioner( pn, PTaskPartitioner.FACTORING_CMAX );
		}
		else
		{
			setTaskPartitioner( pn, PTaskPartitioner.FACTORING );
		}
	}
	
	/**
	 * 
	 * @param n
	 * @param partitioner
	 * @param flagLIX
	 */
	private void setTaskPartitioner( OptNode n, PTaskPartitioner partitioner )
	{
		long id = n.getID();
		
		// modify rtprog
		ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
                                     .getAbstractPlanMapping().getMappedProg(id)[1];
		pfpb.setTaskPartitioner(partitioner);
		
		// modify plan
		n.addParam(ParamType.TASK_PARTITIONER, partitioner.toString());
		
		//handle specific case of LIX recompile
		boolean flagLIX = (partitioner == PTaskPartitioner.FACTORING_CMAX);
		if( flagLIX ) 
		{
			int maxc = n.getMaxC( _N );
			pfpb.setTaskSize( maxc ); //used as constraint 
			pfpb.disableJVMReuse();
			n.addParam(ParamType.TASK_SIZE, String.valueOf(maxc));
		}
		
		LOG.debug("RULEBASED OPT: rewrite 'set task partitioner' - result="+partitioner+((flagLIX) ? ","+n.getParam(ParamType.TASK_SIZE) : "") );	
	}
	
	
	///////
	//REWRITE set result merge
	///
	
	/**
	 * 
	 * @param n
	 * @throws DMLRuntimeException 
	 */
	private void rewriteSetResultMerge( OptNode n, LocalVariableMap vars ) 
		throws DMLRuntimeException
	{
		ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
								    .getAbstractPlanMapping().getMappedProg(n.getID())[1];
		
		//investigate details of current parfor node
		boolean flagMRParFOR = (n.getExecType() == ExecType.MR);
		boolean flagMRLeftIndexing = hasResultMRLeftIndexing( n, pfpb.getResultVariables(), vars, true );
		boolean flagCellFormatWoCompare = determineFlagCellFormatWoCompare(pfpb.getResultVariables(), vars); 
		boolean flagOnlyInMemResults = hasOnlyInMemoryResults(n, pfpb.getResultVariables(), vars );
		
		//actual decision on result merge
		PResultMerge ret = null;
		if( flagOnlyInMemResults )
			ret = PResultMerge.LOCAL_MEM;
		else if(    ( flagMRParFOR || flagMRLeftIndexing) 
			    && !(flagCellFormatWoCompare && ResultMergeLocalFile.ALLOW_COPY_CELLFILES ) )
			ret = PResultMerge.REMOTE_MR;
		else
			ret = PResultMerge.LOCAL_AUTOMATIC;
		//NOTE: 'at least one' instead of 'all' condition of flagMRLeftIndexing because the 
		//      benefit for large matrices outweigths potentially unnecessary MR jobs for smaller matrices)
		
		// modify rtprog	
		pfpb.setResultMerge(ret);
			
		// modify plan
		n.addParam(ParamType.RESULT_MERGE, ret.toString());			

		//recursively apply rewrite for parfor nodes
		if( n.getChilds() != null )
			rInvokeSetResultMerge(n.getChilds(), vars);
		
		LOG.debug("RULEBASED OPT: rewrite 'set result merge' - result="+ret );
	}
	
	/**
	 * 
	 * @param resultVars
	 * @param vars
	 * @return
	 */
	private boolean determineFlagCellFormatWoCompare( ArrayList<String> resultVars, LocalVariableMap vars  )
	{
		boolean ret = true;
		
		for( String rVar : resultVars )
		{
			Data dat = vars.get(rVar);
			if( dat == null || !(dat instanceof MatrixObject) )
			{
				ret = false; 
				break;
			}
			else
			{
				MatrixObject mo = (MatrixObject)dat;
				MatrixFormatMetaData meta = (MatrixFormatMetaData) mo.getMetaData();
				OutputInfo oi = meta.getOutputInfo();
				long nnz = meta.getMatrixCharacteristics().getNonZeros();
				
				if( oi == OutputInfo.BinaryBlockOutputInfo || nnz != 0 )
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
	 * @param n
	 * @param resultVars
	 * @return
	 * @throws DMLRuntimeException 
	 */
	public boolean hasResultMRLeftIndexing( OptNode n, ArrayList<String> resultVars, LocalVariableMap vars, boolean checkSize ) 
		throws DMLRuntimeException
	{
		boolean ret = false;
		
		if( n.isLeaf() )
		{
			String opName = n.getParam(ParamType.OPSTRING);
			//check opstring and exec type
			if( opName.equals(LeftIndexingOp.OPSTRING) && n.getExecType()==ExecType.MR )
			{
				LeftIndexingOp hop = (LeftIndexingOp) OptTreeConverter.getAbstractPlanMapping().getMappedHop(n.getID());
				//check agains set of varname
				String varName = hop.getInput().get(0).get_name();
				if( resultVars.contains(varName) )
				{
					ret = true;
					if( checkSize && vars.keySet().contains(varName) )
					{
						//dims of result vars must be known at this point in time
						MatrixObject mo = (MatrixObject) vars.get( hop.getInput().get(0).get_name() );
						long rows = mo.getNumRows();
						long cols = mo.getNumColumns();
						ret = !isInMemoryResultMerge(rows, cols);
					}
				}
			}
		}
		else
		{
			for( OptNode c : n.getChilds() )
				ret |= hasResultMRLeftIndexing(c, resultVars, vars, checkSize);
		}
		
		return ret;
	}
	
	public boolean hasOnlyInMemoryResults( OptNode n, ArrayList<String> resultVars, LocalVariableMap vars ) 
		throws DMLRuntimeException
	{
		boolean ret = true;
		
		if( n.isLeaf() )
		{
			String opName = n.getParam(ParamType.OPSTRING);
			//check opstring and exec type
			if( opName.equals(LeftIndexingOp.OPSTRING) )
			{
				LeftIndexingOp hop = (LeftIndexingOp) OptTreeConverter.getAbstractPlanMapping().getMappedHop(n.getID());
				//check agains set of varname
				String varName = hop.getInput().get(0).get_name();
				if( resultVars.contains(varName) && vars.keySet().contains(varName) )
				{
					//dims of result vars must be known at this point in time
					MatrixObject mo = (MatrixObject) vars.get( hop.getInput().get(0).get_name() );
					long rows = mo.getNumRows();
					long cols = mo.getNumColumns();
					ret &= isInMemoryResultMerge(rows, cols);
				}
			}
		}
		else
		{
			for( OptNode c : n.getChilds() )
				ret &= hasOnlyInMemoryResults(c, resultVars, vars);
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @param nodes
	 * @param vars
	 * @throws DMLRuntimeException 
	 */
	private void rInvokeSetResultMerge( Collection<OptNode> nodes, LocalVariableMap vars) 
		throws DMLRuntimeException
	{
		for( OptNode n : nodes )
			if( n.getNodeType() == NodeType.PARFOR )
				rewriteSetResultMerge(n, vars);
			else if( n.getChilds()!=null )  
				rInvokeSetResultMerge(n.getChilds(), vars);
	}
	
	public static boolean isInMemoryResultMerge( long rows, long cols )
	{
		return ( rows>=0 && cols>=0 && rows*cols < Math.pow(Hops.CPThreshold, 2) );
	}

	
	///////
	//REWRITE set recompile memory budget
	///

	/**
	 * 
	 * @param n
	 * @param M
	 */
	private void rewriteSetRecompileMemoryBudget( OptNode n )
	{
		double newLocalMem = _lm; 
		
		//check et because recompilation only happens at the master node
		if( n.getExecType() == ExecType.CP )
		{
			//compute local recompile memory budget
			int par = n.getTotalK();
			newLocalMem = _lm / par;
			
			//modify runtime plan
			ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
            							.getAbstractPlanMapping().getMappedProg(n.getID())[1];
			pfpb.setRecompileMemoryBudget( newLocalMem );
		}
		
		LOG.debug("RULEBASED OPT: rewrite 'set recompile memory budget' - result="+toMB(newLocalMem) );
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
		
		LOG.debug("RULEBASED OPT: rewrite 'remove unnecessary parfor' - result="+count );
	}
	
	/**
	 * 
	 * @param n
	 * @return
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 */
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
		return OptimizerUtils.toMB(inB) + "MB";
	}
}
