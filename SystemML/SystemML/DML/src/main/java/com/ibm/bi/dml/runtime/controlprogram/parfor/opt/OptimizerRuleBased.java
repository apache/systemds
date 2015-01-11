/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor.opt;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map.Entry;
import java.util.Set;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.conf.ConfigurationManager;
import com.ibm.bi.dml.hops.DataOp;
import com.ibm.bi.dml.hops.FunctionOp;
import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.hops.Hop.ReOrgOp;
import com.ibm.bi.dml.hops.HopsException;
import com.ibm.bi.dml.hops.IndexingOp;
import com.ibm.bi.dml.hops.LeftIndexingOp;
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.hops.ReorgOp;
import com.ibm.bi.dml.lops.LopProperties;
import com.ibm.bi.dml.lops.LopsException;
import com.ibm.bi.dml.lops.compile.Recompiler;
import com.ibm.bi.dml.parser.DMLProgram;
import com.ibm.bi.dml.parser.Expression.DataType;
import com.ibm.bi.dml.parser.FunctionStatementBlock;
import com.ibm.bi.dml.parser.LanguageException;
import com.ibm.bi.dml.parser.ParForStatementBlock;
import com.ibm.bi.dml.parser.StatementBlock;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.DMLUnsupportedOperationException;
import com.ibm.bi.dml.runtime.controlprogram.ExecutionContext;
import com.ibm.bi.dml.runtime.controlprogram.ForProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.FunctionProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.LocalVariableMap;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.Program;
import com.ibm.bi.dml.runtime.controlprogram.ProgramBlock;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PDataPartitioner;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PExecMode;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.POptMode;
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
import com.ibm.bi.dml.runtime.instructions.CPInstructions.FunctionCallCPInstruction;
import com.ibm.bi.dml.runtime.matrix.MatrixFormatMetaData;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.data.SparseRow;
import com.ibm.bi.dml.yarn.ropt.YarnClusterAnalyzer;

/**
 * Rule-Based ParFor Optimizer (time: O(n)):
 * 
 * Applied rule-based rewrites
 * - 1) rewrite set data partitioner (incl. recompile RIX)
 * - 2) rewrite result partitioning (incl. recompile LIX)
 * - 3) rewrite set execution strategy
 * - 4) rewrite set operations exec type (incl. recompile)
 * - 5) rewrite use data colocation		 
 * - 6) rewrite set partition replication factor
 * - 7) rewrite set export replication factor 
 * - 8) rewrite use nested parallelism 
 * - 9) rewrite set degree of parallelism
 * - 10) rewrite set task partitioner
 * - 11) rewrite set fused data partitioning and execution
 * - 12) rewrite transpose vector operations (for sparse)
 * - 13) rewrite set in-place result indexing
 * - 14) rewrite disable caching (prevent sparse serialization)
 * - 15) rewrite enable runtime piggybacking
 * - 16) rewrite set result merge 		 		 
 * - 17) rewrite set recompile memory budget
 * - 18) rewrite remove recursive parfor	
 * - 19) rewrite remove unnecessary parfor		
 * 	 
 * TODO fuse also result merge into fused data partitioning and execute
 *      (for writing the result directly from execute we need to partition
 *      columns/rows according to blocksize -> rewrite (only applicable if 
 *      numCols/blocksize>numreducers)+custom MR partitioner)
 * 
 * 
 * TODO take remote memory into account in data/result partitioning rewrites (smaller/larger)
 * TODO memory estimates with shared reads
 * TODO memory estimates of result merge into plan tree 
 * TODO blockwise partitioning
 *  
 */
public class OptimizerRuleBased extends Optimizer
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public static final double PROB_SIZE_THRESHOLD_REMOTE = 100; //wrt # top-level iterations (min)
	public static final double PROB_SIZE_THRESHOLD_PARTITIONING = 2; //wrt # top-level iterations (min)
	public static final double PROB_SIZE_THRESHOLD_MB = 128*1024*1024; //wrt overall memory consumption (min)
	public static final int MAX_REPLICATION_FACTOR_PARTITIONING = 5;     
	public static final int MAX_REPLICATION_FACTOR_EXPORT = 5;    
	public static final boolean ALLOW_REMOTE_NESTED_PARALLELISM = false;
	public static final boolean APPLY_REWRITE_NESTED_PARALLELISM = false;
	public static final String FUNCTION_UNFOLD_NAMEPREFIX = "__unfold_";
	
	public static final double PAR_K_FACTOR        = OptimizationWrapper.PAR_FACTOR_INFRASTRUCTURE; 
	public static final double PAR_K_MR_FACTOR     = 1.0 * OptimizationWrapper.PAR_FACTOR_INFRASTRUCTURE; 
	
	//problem and infrastructure properties
	protected long _N    = -1; //problemsize
	protected long _Nmax = -1; //max problemsize (including subproblems)
	protected int _lk   = -1; //local par
	protected int _lkmaxCP = -1; //local max par (if only CP inst)
	protected int _lkmaxMR = -1; //local max par (if also MR inst)
	protected int _rnk  = -1; //remote num nodes
	protected int _rk   = -1; //remote par (mappers)
	protected int _rk2  = -1; //remote par (reducers)
	protected int _rkmax = -1; //remote max par (mappers)
	protected int _rkmax2 = -1; //remote max par (reducers)
	protected double _lm = -1; //local memory constraint
	protected double _rm = -1; //remote memory constraint (mappers)
	protected double _rm2 = -1; //remote memory constraint (reducers)
	
	
	protected CostEstimator _cost = null;

	
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

	@Override
	public POptMode getOptMode() 
	{
		return POptMode.RULEBASED;
	}
	
	/**
	 * Main optimization procedure.
	 * 
	 * Transformation-based heuristic (rule-based) optimization
	 * (no use of sb, direct change of pb).
	 */
	@Override
	public boolean optimize(ParForStatementBlock sb, ParForProgramBlock pb, OptTree plan, CostEstimator est, ExecutionContext ec) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		LOG.debug("--- "+getOptMode()+" OPTIMIZER -------");

		OptNode pn = plan.getRoot();
		double M0 = -1, M1 = -1, M2 = -1; //memory consumption
		
		//early abort for empty parfor body 
		if( pn.isLeaf() )
			return true;
		
		//ANALYZE infrastructure properties
		analyzeProblemAndInfrastructure( pn );
		
		_cost = est;
		
		//debug and warnings output
		LOG.debug(getOptMode()+" OPT: Optimize with local_max_mem="+toMB(_lm)+" and remote_max_mem="+toMB(_rm)+")." );
		if( _rnk<=0 || _rk<=0 )
			LOG.warn(getOptMode()+" OPT: Optimize for inactive cluster (num_nodes="+_rnk+", num_map_slots="+_rk+")." );
		
		//ESTIMATE memory consumption 
		pn.setSerialParFor(); //for basic mem consumption 
		M0 = _cost.getEstimate(TestMeasure.MEMORY_USAGE, pn);
		LOG.debug(getOptMode()+" OPT: estimated mem (serial exec) M="+toMB(M0) );
		
		//OPTIMIZE PARFOR PLAN
		
		// rewrite 1: data partitioning (incl. log. recompile RIX)
		HashMap<String, PDataPartitionFormat> partitionedMatrices = new HashMap<String,PDataPartitionFormat>();
		rewriteSetDataPartitioner( pn, ec.getVariables(), partitionedMatrices );
		M1 = _cost.getEstimate(TestMeasure.MEMORY_USAGE, pn); //reestimate
		
		// rewrite 2: rewrite result partitioning (incl. log/phy recompile LIX) 
		boolean flagLIX = rewriteSetResultPartitioning( pn, M1, ec.getVariables() );
		M1 = _cost.getEstimate(TestMeasure.MEMORY_USAGE, pn); //reestimate 
		M2 = _cost.getEstimate(TestMeasure.MEMORY_USAGE, pn, LopProperties.ExecType.CP);
		LOG.debug(getOptMode()+" OPT: estimated new mem (serial exec) M="+toMB(M1) );
		LOG.debug(getOptMode()+" OPT: estimated new mem (serial exec, all CP) M="+toMB(M2) );
		
		// rewrite 3: execution strategy
		boolean flagRecompMR = rewriteSetExecutionStategy( pn, M0, M1, M2, flagLIX );
		
		//exec-type-specific rewrites
		if( pn.getExecType() == ExecType.MR )
		{
			if( flagRecompMR ){
				//rewrite 4: set operations exec type
				rewriteSetOperationsExecType( pn, flagRecompMR );
				M1 = _cost.getEstimate(TestMeasure.MEMORY_USAGE, pn); //reestimate 		
			}
			
			// rewrite 5: data colocation
			rewriteDataColocation( pn, ec.getVariables() );
			
			// rewrite 6: rewrite set partition replication factor
			rewriteSetPartitionReplicationFactor( pn, partitionedMatrices, ec.getVariables() );
			
			// rewrite 7: rewrite set partition replication factor
			rewriteSetExportReplicationFactor( pn, ec.getVariables() );
			
			// rewrite 8: nested parallelism (incl exec types)	
			boolean flagNested = rewriteNestedParallelism( pn, M1, flagLIX );
			
			// rewrite 9: determine parallelism
			rewriteSetDegreeOfParallelism( pn, M1, flagNested );
			
			// rewrite 10: task partitioning 
			rewriteSetTaskPartitioner( pn, flagNested, flagLIX );
			
			// rewrite 11: fused data partitioning and execution
			rewriteSetFusedDataPartitioningExecution(pn, M1, flagLIX, partitionedMatrices, ec.getVariables());
		
			// rewrite 12: transpose sparse vector operations
			rewriteSetTranposeSparseVectorOperations(pn, partitionedMatrices, ec.getVariables());
			
			//rewrite 13: set in-place result indexing
			HashSet<String> inplaceResultVars = new HashSet<String>();
			rewriteSetInPlaceResultIndexing(pn, M1, ec.getVariables(), inplaceResultVars);
			
			//rewrite 14: disable caching
			rewriteDisableCPCaching(pn, inplaceResultVars, ec.getVariables());
		}
		else //if( pn.getExecType() == ExecType.CP )
		{
			// rewrite 9: determine parallelism
			rewriteSetDegreeOfParallelism( pn, M1, false );
			
			// rewrite 10: task partitioning
			rewriteSetTaskPartitioner( pn, false, false ); //flagLIX always false 
			
			// rewrite 15: runtime piggybacking
			rewriteEnableRuntimePiggybacking( pn, ec.getVariables(), partitionedMatrices );
		}	

		// rewrite 16: set result merge
		rewriteSetResultMerge( pn, ec.getVariables(), true );
		
		// rewrite 17: set local recompile memory budget
		rewriteSetRecompileMemoryBudget( pn );
		
		///////
		//Final rewrites for cleanup / minor improvements
		
		// rewrite 18: parfor (in recursive functions) to for
		rewriteRemoveRecursiveParFor( pn, ec.getVariables() );
		
		// rewrite 19: parfor (par=1) to for 
		rewriteRemoveUnnecessaryParFor( pn );
		
		//info optimization result
		_numTotalPlans = -1; //_numEvaluatedPlans maintained in rewrites;
		return true;
	}

	/**
	 * 
	 * @param pn
	 */
	protected void analyzeProblemAndInfrastructure( OptNode pn )
	{
		_N       = Long.parseLong(pn.getParam(ParamType.NUM_ITERATIONS)); 
		_Nmax    = pn.getMaxProblemSize(); 
		_lk      = InfrastructureAnalyzer.getLocalParallelism();
		_lkmaxCP = (int) Math.ceil( PAR_K_FACTOR * _lk ); 
		_lkmaxMR = (int) Math.ceil( PAR_K_MR_FACTOR * _lk );
		_rnk     = InfrastructureAnalyzer.getRemoteParallelNodes();  
		_rk      = InfrastructureAnalyzer.getRemoteParallelMapTasks();
		_rk2     = InfrastructureAnalyzer.getRemoteParallelReduceTasks();
		_rkmax   = (int) Math.ceil( PAR_K_FACTOR * _rk ); 
		_rkmax2  = (int) Math.ceil( PAR_K_FACTOR * _rk2 ); 
		_lm      = OptimizerUtils.getLocalMemBudget();
		_rm      = OptimizerUtils.getRemoteMemBudgetMap(false); 	
		_rm2     = OptimizerUtils.getRemoteMemBudgetReduce(); 	
		
		//correction of max parallelism if yarn enabled because yarn
		//does not have the notion of map/reduce slots and hence returns 
		//small constants of map=10*nodes, reduce=2*nodes
		//(not doing this correction would loose available degree of parallelism)
		if( InfrastructureAnalyzer.isYarnEnabled() ) {
			long tmprk = YarnClusterAnalyzer.getNumCores();
			_rk = (int) Math.max( _rk, tmprk );
			_rk2 = (int) Math.max( _rk2, tmprk );
		}
	}
	
	///////
	//REWRITE set data partitioner
	///
	
	/**
	 * 
	 * @param n
	 * @param partitionedMatrices  
	 * @throws DMLRuntimeException 
	 */
	protected boolean rewriteSetDataPartitioner(OptNode n, LocalVariableMap vars, HashMap<String, PDataPartitionFormat> partitionedMatrices ) 
		throws DMLRuntimeException
	{
		if( n.getNodeType() != NodeType.PARFOR )
			LOG.warn(getOptMode()+" OPT: Data partitioner can only be set for a ParFor node.");
		
		boolean blockwise = false;
		
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
				//System.out.println("Partitioning Format: "+dpf);
				if( dpf != PDataPartitionFormat.NONE 
					&& dpf != PDataPartitionFormat.BLOCK_WISE_M_N ) //FIXME
				{
					cand2.put( c, dpf );
				}
					
			}
			apply = rFindDataPartitioningCandidates(n, cand2, vars);
			if( apply )
				partitionedMatrices.putAll(cand2);
		}
		PDataPartitioner pdp = (apply)? PDataPartitioner.REMOTE_MR : PDataPartitioner.NONE;		
		//NOTE: since partitioning is only applied in case of MR index access, we assume a large
		//      matrix and hence always apply REMOTE_MR (the benefit for large matrices outweigths
		//      potentially unnecessary MR jobs for smaller matrices)
		
		// modify rtprog 
		pfpb.setDataPartitioner( pdp );
		// modify plan
		n.addParam(ParamType.DATA_PARTITIONER, pdp.toString());
	
		_numEvaluatedPlans++;
		LOG.debug(getOptMode()+" OPT: rewrite 'set data partitioner' - result="+pdp.toString()+
				  " ("+ProgramConverter.serializeStringCollection(partitionedMatrices.keySet())+")" );
		
		return blockwise;
	}
	
	/**
	 * 
	 * @param n
	 * @param cand
	 * @return
	 * @throws DMLRuntimeException 
	 */
	protected boolean rFindDataPartitioningCandidates( OptNode n, HashMap<String, PDataPartitionFormat> cand, LocalVariableMap vars ) 
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
			     && n.getParam(ParamType.OPSTRING).equals(IndexingOp.OPSTRING) )
		{
			Hop h = OptTreeConverter.getAbstractPlanMapping().getMappedHop(n.getID());
			String inMatrix = h.getInput().get(0).get_name();
			if( cand.containsKey(inMatrix) ) //Required Condition: partitioning applicable
			{
				PDataPartitionFormat dpf = cand.get(inMatrix);
				double mnew = getNewRIXMemoryEstimate( n, inMatrix, dpf, vars );
				//NOTE: for the moment, we do not partition according to the remote mem, because we can execute 
				//it even without partitioning in CP. However, advanced optimizers should reason about this 					   
				//double mold = h.getMemEstimate();
				if(	   n.getExecType() == ExecType.MR ) //Opt Condition: MR, 
				   // || (mold > _rm && mnew <= _rm)   ) //Opt Condition: non-MR special cases (for remote exec)
				{
					//NOTE: subsequent rewrites will still use the MR mem estimate
					//(guarded by subsequent operations that have at least the memory req of one partition)
					if( mnew < _lm ) //apply rewrite if partitions fit into memory
						n.setExecType(ExecType.CP);
					else
						n.setExecType(ExecType.CP); //CP_FILE, but hop still in MR 
					n.addParam(ParamType.DATA_PARTITION_FORMAT, dpf.toString());
					h.setMemEstimate( mnew ); //CP vs CP_FILE in ProgramRecompiler bases on mem_estimate
					ret = true;
				}
			}
		}
		
		return ret;
	}
	
	/**
	 * TODO consolidate mem estimation with Indexing Hop
	 * 
	 * NOTE: Using the dimensions without sparsity is a conservative worst-case consideration.
	 * 
	 * @param n
	 * @param varName
	 * @param dpf
	 * @return
	 * @throws DMLRuntimeException 
	 */
	protected double getNewRIXMemoryEstimate( OptNode n, String varName, PDataPartitionFormat dpf, LocalVariableMap vars ) 
		throws DMLRuntimeException
	{
		double mem = -1;
		
		//not all intermediates need to be known on optimize
		Data dat = vars.get( varName );
		if( dat != null )
		{
			MatrixObject mo = (MatrixObject) dat;
			
			//those are worst-case (dense) estimates
			switch( dpf )
			{
				case COLUMN_WISE:
					mem = mo.getNumRows() * 8; 
					break;
				case ROW_WISE:
					mem = mo.getNumColumns() * 8;
					break;
				case BLOCK_WISE_M_N:
					mem = Integer.MAX_VALUE; //TODO
					break;
			}	
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
	protected static LopProperties.ExecType getRIXExecType( MatrixObject mo, PDataPartitionFormat dpf ) 
		throws DMLRuntimeException
	{
		return getRIXExecType(mo, dpf, false);
	}
	
	/**
	 * 
	 * @param mo
	 * @param dpf
	 * @return
	 * @throws DMLRuntimeException
	 */
	protected static LopProperties.ExecType getRIXExecType( MatrixObject mo, PDataPartitionFormat dpf, boolean withSparsity ) 
		throws DMLRuntimeException
	{
		double mem = -1;		
		
		long rlen = mo.getNumRows();
		long clen = mo.getNumColumns();
		long brlen = mo.getNumRowsPerBlock();
		long bclen = mo.getNumColumnsPerBlock();
		long nnz = mo.getNnz();
		double lsparsity = ((double)nnz)/rlen/clen;		
		double sparsity = withSparsity ? lsparsity : 1.0;
		
		switch( dpf )
		{
			case COLUMN_WISE:
				mem = OptimizerUtils.estimateSizeExactSparsity(mo.getNumRows(), 1, sparsity); 
				break;
			case COLUMN_BLOCK_WISE:
				mem = OptimizerUtils.estimateSizeExactSparsity(mo.getNumRows(), bclen, sparsity); 
				break;
			case ROW_WISE:
				mem = OptimizerUtils.estimateSizeExactSparsity(1, mo.getNumColumns(), sparsity);
				break;
			case ROW_BLOCK_WISE:
				mem = OptimizerUtils.estimateSizeExactSparsity(brlen, mo.getNumColumns(), sparsity);
				break;
		}
		
		if( mem < OptimizerUtils.getLocalMemBudget() )
			return LopProperties.ExecType.CP;
		else
			return LopProperties.ExecType.CP_FILE;
	}
	
	/**
	 * 
	 * @param mo
	 * @param dpf
	 * @return
	 * @throws DMLRuntimeException
	 */
	public static PDataPartitionFormat decideBlockWisePartitioning( MatrixObject mo, PDataPartitionFormat dpf ) 
		throws DMLRuntimeException
	{
		long rlen = mo.getNumRows();
		long clen = mo.getNumColumns();
		long brlen = mo.getNumRowsPerBlock();
		long bclen = mo.getNumColumnsPerBlock();
		long k = InfrastructureAnalyzer.getRemoteParallelMapTasks();
		
		PDataPartitionFormat ret = dpf;
		if( getRIXExecType(mo, dpf)==LopProperties.ExecType.CP )
		if( ret == PDataPartitionFormat.ROW_WISE )
		{
			if( rlen/brlen > 4*k && //note: average sparsity, read must deal with it
				getRIXExecType(mo, PDataPartitionFormat.ROW_BLOCK_WISE, false)==LopProperties.ExecType.CP )
			{
				ret = PDataPartitionFormat.ROW_BLOCK_WISE;				
			}
		}
		else if( ret == PDataPartitionFormat.COLUMN_WISE )
		{
			if( clen/bclen > 4*k && //note: average sparsity, read must deal with it
				getRIXExecType(mo, PDataPartitionFormat.COLUMN_BLOCK_WISE, false)==LopProperties.ExecType.CP )
			{
				ret = PDataPartitionFormat.COLUMN_BLOCK_WISE;				
			}
		}
				
		return ret;	
	}
	
	/**
	 * 
	 * @return
	 * @throws DMLRuntimeException 
	 */
	public static boolean allowsBinaryCellPartitions( MatrixObject mo, PDataPartitionFormat dpf ) 
		throws DMLRuntimeException
	{
		return (getRIXExecType(mo, PDataPartitionFormat.COLUMN_BLOCK_WISE, false)==LopProperties.ExecType.CP );
	}
	
	///////
	//REWRITE set result partitioning
	///

	/**
	 * 
	 * @param n
	 * @throws DMLRuntimeException
	 */
	protected boolean rewriteSetResultPartitioning(OptNode n, double M, LocalVariableMap vars) 
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
		                && isResultPartitionableAll(cand,pfpb.getResultVariables(),vars, pfpb.getIterablePredicateVars()[0]); // check candidates
			
		//recompile LIX
		if( apply )
		{
			try
			{
				for(OptNode lix : cand)
					recompileLIX( lix, vars );
			}
			catch(Exception ex)
			{
				throw new DMLRuntimeException("Unable to recompile LIX.", ex);
			}
		}
		
		_numEvaluatedPlans++;
		LOG.debug(getOptMode()+" OPT: rewrite 'set result partitioning' - result="+apply );
	
		return apply;
	}
	
	/**
	 * 
	 * @param nlist
	 * @param resultVars
	 * @param vars
	 * @param iterVarname
	 * @return
	 * @throws DMLRuntimeException
	 */
	protected boolean isResultPartitionableAll( Collection<OptNode> nlist, ArrayList<String> resultVars, LocalVariableMap vars, String iterVarname ) 
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
	
	/**
	 * 
	 * @param n
	 * @param resultVars
	 * @param vars
	 * @param iterVarname
	 * @return
	 * @throws DMLRuntimeException
	 */
	protected boolean isResultPartitionable( OptNode n, ArrayList<String> resultVars, LocalVariableMap vars, String iterVarname ) 
		throws DMLRuntimeException
	{
		boolean ret = true;
		
		//check left indexing operator
		String opStr = n.getParam(ParamType.OPSTRING);
		if( opStr==null || !opStr.equals(LeftIndexingOp.OPSTRING) )
			ret = false;

		Hop h = null;
		Hop base = null;
		
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
			Hop inpRowL = h.getInput().get(2);
			Hop inpRowU = h.getInput().get(3);
			Hop inpColL = h.getInput().get(4);
			Hop inpColU = h.getInput().get(5);
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
	
	/**
	 * 
	 * @param n
	 * @throws DMLRuntimeException
	 * @throws HopsException
	 * @throws LopsException
	 * @throws DMLUnsupportedOperationException
	 * @throws IOException
	 */
	protected void recompileLIX( OptNode n, LocalVariableMap vars ) 
		throws DMLRuntimeException, HopsException, LopsException, DMLUnsupportedOperationException, IOException
	{
		Hop h = OptTreeConverter.getAbstractPlanMapping().getMappedHop(n.getID());
		
		//set forced exec type
		h.setForcedExecType(LopProperties.ExecType.CP);
		n.setExecType(ExecType.CP);
		
		//recompile parent pb
		long pid = OptTreeConverter.getAbstractPlanMapping().getMappedParentID(n.getID());
		OptNode nParent = OptTreeConverter.getAbstractPlanMapping().getOptNode(pid);
		Object[] o = OptTreeConverter.getAbstractPlanMapping().getMappedProg(pid);
		StatementBlock sb = (StatementBlock) o[0];
		ProgramBlock pb = (ProgramBlock) o[1];
		
		//keep modified estimated of partitioned rix (in same dag as lix)
		HashMap<Hop, Double> estRix = getPartitionedRIXEstimates(nParent);
		
		//construct new instructions
		ArrayList<Instruction> newInst = Recompiler.recompileHopsDag(sb, sb.get_hops(), vars, false, 0);
		pb.setInstructions( newInst );   
		
		//reset all rix estimated (modified by recompile)
		resetPartitionRIXEstimates( estRix );
		
		//set new mem estimate (last, otherwise overwritten from recompile)
		h.setMemEstimate(_rm-1);
	}
	
	/**
	 * 
	 * @param parent
	 * @return
	 */
	protected HashMap<Hop, Double> getPartitionedRIXEstimates(OptNode parent)
	{
		HashMap<Hop, Double> estimates = new HashMap<Hop, Double>();
		for( OptNode n : parent.getChilds() )
			if( n.getParam(ParamType.DATA_PARTITION_FORMAT) != null )
			{
				Hop h = OptTreeConverter.getAbstractPlanMapping().getMappedHop(n.getID());
				estimates.put( h, h.getMemEstimate() );
			}
		return estimates;
	}
	
	/**
	 * 
	 * @param parent
	 * @param estimates
	 */
	protected void resetPartitionRIXEstimates( HashMap<Hop, Double> estimates )
	{
		for( Entry<Hop, Double> e : estimates.entrySet() )
		{
			Hop h = e.getKey();
			double val = e.getValue();
			h.setMemEstimate(val);
		}
	}
	
	
	///////
	//REWRITE set execution strategy
	///
	
	/**
	 * 
	 * @param n
	 * @param M
	 * @throws DMLRuntimeException 
	 */
	protected boolean rewriteSetExecutionStategy(OptNode n, double M0, double M, double M2, boolean flagLIX) 
		throws DMLRuntimeException
	{
		boolean isCPOnly = n.isCPOnly();
		boolean isCPOnlyPossible = isCPOnly || isCPOnlyPossible(n, _rm);
		
		//deciding on the execution strategy
		if(    (isCPOnly && M <= _rm )   //Required: all instruction can be be executed in CP
			|| (isCPOnlyPossible && M2 <= _rm) )  //Required: cp inst fit into remote JVM mem 
		{
			//at this point all required conditions for REMOTE_MR given, now its an opt decision
			int cpk = (int) Math.min( _lk, Math.floor( _lm / M ) ); //estimated local exploited par  
			
			//MR if local par cannot be exploited due to mem constraints (this implies that we work on large data)
			if( cpk < _lk && cpk < _N && cpk < _rk )
			{
				n.setExecType( ExecType.MR ); //remote parfor
			}
			//MR if problem is large enough and remote parallelism is larger than local   
			else if( _lk < _N && _lk < _rk && isLargeProblem(n, M0) )
			{
				n.setExecType( ExecType.MR ); //remote parfor
			}
			//MR if MR operations in local, but CP only in remote (less overall MR jobs)
			else if( (!isCPOnly) && isCPOnlyPossible )
			{
				n.setExecType( ExecType.MR ); //remote parfor
			}
			//MR if necessary for LIX rewrite (LIX true iff cp only and rm valid)
			else if( flagLIX ) 
			{
				n.setExecType( ExecType.MR );  //remote parfor
			}
			//MR if remote data partitioning, because data will be distributed on all nodes 
			else if( n.getParam(ParamType.DATA_PARTITIONER)!=null && n.getParam(ParamType.DATA_PARTITIONER).equals(PDataPartitioner.REMOTE_MR.toString())
					 && !InfrastructureAnalyzer.isLocalMode())
			{
				n.setExecType( ExecType.MR );  //remote parfor
			}
			//otherwise CP
			else 
			{
				n.setExecType( ExecType.CP ); //local parfor	
			}			
		}
		else //mr instructions in body, or rm too small
		{
			n.setExecType( ExecType.CP ); //local parfor
		}
		
		//actual programblock modification
		long id = n.getID();
		ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
		                             .getAbstractPlanMapping().getMappedProg(id)[1];
		PExecMode mode = (n.getExecType()==ExecType.CP)? PExecMode.LOCAL : PExecMode.REMOTE_MR;
		pfpb.setExecMode( mode );	
		
		//decide if recompilation according to remote mem budget necessary
		boolean requiresRecompile = (mode == PExecMode.REMOTE_MR && !isCPOnly );
		
		_numEvaluatedPlans++;
		LOG.debug(getOptMode()+" OPT: rewrite 'set execution strategy' - result="+mode+" (recompile="+requiresRecompile+")" );
		
		return requiresRecompile;
	}
	
	/**
	 * 
	 * @param pn
	 * @return
	 */
	protected boolean isLargeProblem(OptNode pn, double M0)
	{
		return ((_N >= PROB_SIZE_THRESHOLD_REMOTE || _Nmax >= 10 * PROB_SIZE_THRESHOLD_REMOTE )
				&& M0 > PROB_SIZE_THRESHOLD_MB ); //original operations at least larger than 128MB
	}
	
	/**
	 * 
	 * @param n
	 * @param memBudget
	 * @return
	 * @throws DMLRuntimeException
	 */
	protected boolean isCPOnlyPossible( OptNode n, double memBudget ) 
		throws DMLRuntimeException
	{
		ExecType et = n.getExecType();
		boolean ret = ( et == ExecType.CP);		
		
		if( n.isLeaf() && et == ExecType.MR )
		{
			Hop h = OptTreeConverter.getAbstractPlanMapping().getMappedHop( n.getID() );
			if( h.getForcedExecType()!=LopProperties.ExecType.MR ) //e.g., -exec=hadoop
			{
				double mem = _cost.getLeafNodeEstimate(TestMeasure.MEMORY_USAGE, n, LopProperties.ExecType.CP);
				if( mem <= memBudget )
					ret = true;
			}
		}
		
		if( !n.isLeaf() )
			for( OptNode c : n.getChilds() )
			{
				if( !ret ) break; //early abort if already false
				ret &= isCPOnlyPossible(c, memBudget);
			}
		return ret;
	}
	
	
	///////
	//REWRITE set operations exec type
	///
	
	/**
	 * 
	 * @param pn
	 * @param recompile
	 * @throws DMLRuntimeException
	 */
	protected void rewriteSetOperationsExecType(OptNode pn, boolean recompile) 
		throws DMLRuntimeException
	{
		//set exec type in internal opt tree
		int count = setOperationExecType(pn, ExecType.CP);
		
		//recompile program (actual programblock modification)
		if( recompile && count<=0 )
			LOG.warn("OPT: Forced set operations exec type 'CP', but no operation requires recompile.");
		ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
                                  .getAbstractPlanMapping().getMappedProg(pn.getID())[1];
		HashSet<String> fnStack = new HashSet<String>();
		Recompiler.recompileProgramBlockHierarchy2Forced(pfpb.getChildBlocks(), 0, fnStack, LopProperties.ExecType.CP);
		
		//debug output
		LOG.debug(getOptMode()+" OPT: rewrite 'set operation exec type CP' - result="+count);
	}
	
	/**
	 * 
	 * @param n
	 * @param et
	 * @return
	 */
	protected int setOperationExecType( OptNode n, ExecType et )
	{
		int count = 0;
		
		//set operation exec type to CP, count num recompiles
		if( n.getExecType()!=ExecType.CP && n.getNodeType()==NodeType.HOP ) {
			n.setExecType( ExecType.CP );
			count = 1;
		}
		
		//recursively set exec type of childs
		if( !n.isLeaf() )
			for( OptNode c : n.getChilds() )
				count += setOperationExecType(c, et);
		
		return count;
	}
	
	///////
	//REWRITE enable data colocation
	///

	/**
	 * NOTE: if MAX_REPLICATION_FACTOR_PARTITIONING is set larger than 10, co-location may
	 * throw warnings per split since this exceeds "max block locations"
	 * 
	 * @param n
	 * @throws DMLRuntimeException 
	 */
	protected void rewriteDataColocation( OptNode n, LocalVariableMap vars ) 
		throws DMLRuntimeException
	{
		// data colocation is beneficial if we have dp=REMOTE_MR, etype=REMOTE_MR
		// and there is at least one direct col-/row-wise access with the index variable
		// on the partitioned matrix
		boolean apply = false;
		String varname = null;
		String partitioner = n.getParam(ParamType.DATA_PARTITIONER);
		ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
        							.getAbstractPlanMapping().getMappedProg(n.getID())[1];
		
		if(   partitioner!=null && partitioner.equals(PDataPartitioner.REMOTE_MR.toString())
			&& n.getExecType()==ExecType.MR )
		{
			//find all candidates matrices (at least one partitioned access via iterVar)
			HashSet<String> cand = new HashSet<String>();
			rFindDataColocationCandidates(n, cand, pfpb.getIterablePredicateVars()[0]);
			
			//select largest matrix for colocation (based on nnz to account for sparsity)
			long nnzMax = Long.MIN_VALUE;
			for( String c : cand ) {
				MatrixObject tmp = (MatrixObject)vars.get(c);
				if( tmp != null ){
					long nnzTmp = tmp.getNnz();
					if( nnzTmp > nnzMax ) {
						nnzMax = nnzTmp;
						varname = c;
						apply = true;
					}
				}
			}		
		}
		
		//modify the runtime plan (apply true if at least one candidate)
		if( apply )
			pfpb.enableColocatedPartitionedMatrix( varname );
		
		_numEvaluatedPlans++;
		LOG.debug(getOptMode()+" OPT: rewrite 'enable data colocation' - result="+apply+((apply)?" ("+varname+")":"") );
	}
	
	/**
	 * 
	 * @param n
	 * @param cand
	 * @param iterVarname
	 * @return
	 * @throws DMLRuntimeException
	 */
	protected void rFindDataColocationCandidates( OptNode n, HashSet<String> cand, String iterVarname ) 
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
			Hop h = OptTreeConverter.getAbstractPlanMapping().getMappedHop(n.getID());
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
	protected void rewriteSetPartitionReplicationFactor( OptNode n, HashMap<String, PDataPartitionFormat> partitionedMatrices, LocalVariableMap vars ) 
		throws DMLRuntimeException
	{
		boolean apply = false;
		double sizeReplicated = 0;
		int replication = ParForProgramBlock.WRITE_REPLICATION_FACTOR;
		
		ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
        							.getAbstractPlanMapping().getMappedProg(n.getID())[1];
		
		if(    n.getExecType()==ExecType.MR
			&& n.getParam(ParamType.DATA_PARTITIONER).equals(PDataPartitioner.REMOTE_MR.toString())
		    && n.hasNestedParallelism(false) 
		    && n.hasNestedPartitionReads(false) )		
		{
			apply = true;
			
			//account for problem and cluster constraints
			replication = (int)Math.min( _N, _rnk );
			
			//account for internal max constraint (note hadoop will warn if max > 10)
			replication = (int)Math.min( replication, MAX_REPLICATION_FACTOR_EXPORT );
			
			//account for remaining hdfs capacity
			try {
				FileSystem fs = FileSystem.get(ConfigurationManager.getCachedJobConf());
				long hdfsCapacityRemain = fs.getStatus().getRemaining();
				long sizeInputs = 0; //sum of all input sizes (w/o replication)
				for( String var : partitionedMatrices.keySet() )
				{
					MatrixObject mo = (MatrixObject)vars.get(var);
					Path fname = new Path(mo.getFileName());
					if( fs.exists( fname ) ) //non-existing (e.g., CP) -> small file
						sizeInputs += fs.getContentSummary(fname).getLength();		
				}
				replication = (int) Math.min(replication, Math.floor(0.9*hdfsCapacityRemain/sizeInputs));
				
				//ensure at least replication 1
				replication = Math.max( replication, ParForProgramBlock.WRITE_REPLICATION_FACTOR);
				sizeReplicated = replication * sizeInputs;
			}
			catch(Exception ex)
			{
				throw new DMLRuntimeException("Failed to analyze remaining hdfs capacity.", ex);
			}
		}
		
		//modify the runtime plan 
		if( apply )
			pfpb.setPartitionReplicationFactor( replication );
		
		_numEvaluatedPlans++;
		LOG.debug(getOptMode()+" OPT: rewrite 'set partition replication factor' - result="+apply+
				                 ((apply)?" ("+replication+", "+toMB(sizeReplicated)+")":"") );
	}

	///////
	//REWRITE set export replication factor
	///

	/**
	 * Increasing the export replication factor is beneficial for remote execution
	 * because each task will read the full input data set. This only applies to
	 * matrices that are created as in-memory objects before parfor execution. 
	 * 
	 * NOTE: this rewrite requires 'set execution strategy' to be executed. 
	 *  
	 * @param n
	 * @param partitionedMatrices 
	 * @throws DMLRuntimeException 
	 */
	protected void rewriteSetExportReplicationFactor( OptNode n, LocalVariableMap vars ) 
		throws DMLRuntimeException
	{
		boolean apply = false;
		int replication = -1;
		
		ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
        							.getAbstractPlanMapping().getMappedProg(n.getID())[1];
		
		//decide on the replication factor 
		if( n.getExecType()==ExecType.MR )		
		{
			apply = true;
			
			//account for problem and cluster constraints
			replication = (int)Math.min( _N, _rnk );
			
			//account for internal max constraint (note hadoop will warn if max > 10)
			replication = (int)Math.min( replication, MAX_REPLICATION_FACTOR_EXPORT );
		}
		
		//modify the runtime plan 
		if( apply )
			pfpb.setExportReplicationFactor( replication );
		
		_numEvaluatedPlans++;
		LOG.debug(getOptMode()+" OPT: rewrite 'set export replication factor' - result="+apply+((apply)?" ("+replication+")":"") );
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
	protected boolean rewriteNestedParallelism(OptNode n, double M, boolean flagLIX ) 
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

		_numEvaluatedPlans++;
		LOG.debug(getOptMode()+" OPT: rewrite 'enable nested parallelism' - result="+nested );
		
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
	protected void rewriteSetDegreeOfParallelism(OptNode n, double M, boolean flagNested) 
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
			int tmpK = (int)((_N<kMax)? _N : kMax);
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
				int tmpK = (int)((_N<_rk)? _N : _rk);
				pfpb.setDegreeOfParallelism(tmpK);
				n.setK(tmpK);	
				
				kMax = _rkmax / tmpK; //per node (CP only inside)
			}
			
			if( ALLOW_REMOTE_NESTED_PARALLELISM ) {
				//ensure remote memory constraint
				kMax = Math.min( kMax, (int)Math.floor( _rm / M ) ); //guaranteed >= 1 (see exec strategy)
				if( kMax < 1 )
					kMax = 1;
					
				//distribute remaining parallelism
				rAssignRemainingParallelism( n, kMax ); 
			}
		}		
		
		_numEvaluatedPlans++;
		LOG.debug(getOptMode()+" OPT: rewrite 'set degree of parallelism' - result=(see EXPLAIN)" );
	}
	
	/**
	 * 
	 * @param n
	 * @param par
	 */
	protected void rAssignRemainingParallelism(OptNode n, int par) 
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
	protected void rewriteSetTaskPartitioner(OptNode pn, boolean flagNested, boolean flagLIX) 
	{
		//assertions (warnings of corrupt optimizer decisions)
		if( pn.getNodeType() != NodeType.PARFOR )
			LOG.warn(getOptMode()+" OPT: Task partitioner can only be set for a ParFor node.");
		if( flagNested && flagLIX )
			LOG.warn(getOptMode()+" OPT: Task partitioner decision has conflicting input from rewrites 'nested parallelism' and 'result partitioning'.");
		
		
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
		else if( _N/4 >= pn.getK() ) //to prevent imbalance due to ceiling
		{
			setTaskPartitioner( pn, PTaskPartitioner.FACTORING );
		}
		else
		{
			setTaskPartitioner( pn, PTaskPartitioner.NAIVE );
		}
	}
	
	/**
	 * 
	 * @param n
	 * @param partitioner
	 * @param flagLIX
	 */
	protected void setTaskPartitioner( OptNode n, PTaskPartitioner partitioner )
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
			long maxc = n.getMaxC( _N );
			pfpb.setTaskSize( maxc ); //used as constraint 
			pfpb.disableJVMReuse();
			n.addParam(ParamType.TASK_SIZE, String.valueOf(maxc));
		}
		
		_numEvaluatedPlans++;
		LOG.debug(getOptMode()+" OPT: rewrite 'set task partitioner' - result="+partitioner+((flagLIX) ? ","+n.getParam(ParamType.TASK_SIZE) : "") );	
	}
	
	///////
	//REWRITE set fused data partitioning / execution
	///
	
	/**
	 * This dedicated execution mode can only be applied if all of the 
	 * following conditions are true:
	 * - Only cp instructions in the parfor body
	 * - Only one partitioned input 
	 * - number of iterations is equal to number of partitions (nrow/ncol)
	 * - partitioned matrix access via plain iteration variables (no composed expressions)
	 *   (this ensures that each partition is exactly read once)
	 * - no left indexing (since by default static task partitioning)
	 * 
	 * Furthermore, it should be only chosen if we already decided for remote partitioning
	 * and otherwise would create a large number of partition files.
	 * 
	 * NOTE: We already respect the reducer memory budget for plan correctness. However,
	 * we miss optimization potential if the reducer budget is larger than the mapper budget
	 * (if we were not able to select REMOTE_MR as execution strategy wrt mapper budget)
	 * TODO modify 'set exec strategy' and related rewrites for conditional data partitioning.
	 * 
	 * 
	 * @param M 
	 * @param partitionedMatrices, ExecutionContext ec 
	 * 
	 * @param n
	 * @param partitioner
	 * @throws DMLRuntimeException 
	 */
	protected void rewriteSetFusedDataPartitioningExecution(OptNode pn, double M, boolean flagLIX, HashMap<String, PDataPartitionFormat> partitionedMatrices, LocalVariableMap vars) 
		throws DMLRuntimeException 
	{
		//assertions (warnings of corrupt optimizer decisions)
		if( pn.getNodeType() != NodeType.PARFOR )
			LOG.warn(getOptMode()+" OPT: Fused data partitioning and execution is only applicable for a ParFor node.");
		
		boolean apply = false;
		String partitioner = pn.getParam(ParamType.DATA_PARTITIONER);
		
		//precondition: rewrite only invoked if exec type MR 
		// (this also implies that the body is CP only)
		
		// try to merge MR data partitioning and MR exec 
		if(  pn.getExecType()==ExecType.MR   //MR EXEC and CP body
			&& M < _rm2 //fits into remote memory of reducers	
			&& partitioner!=null && partitioner.equals(PDataPartitioner.REMOTE_MR.toString()) //MR partitioning
			&& partitionedMatrices.size()==1 ) //only one partitioned matrix
		{
			ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
	                  .getAbstractPlanMapping().getMappedProg(pn.getID())[1];
			
			//partitioned matrix
			String moVarname = partitionedMatrices.keySet().iterator().next();
			PDataPartitionFormat moDpf = partitionedMatrices.get(moVarname);
			MatrixObject mo = (MatrixObject)vars.get(moVarname);
			
			//check if access via iteration variable and sizes match
			String iterVarname = pfpb.getIterablePredicateVars()[0];
			
			if( rIsAccessByIterationVariable(pn, moVarname, iterVarname) &&
			   ((moDpf==PDataPartitionFormat.ROW_WISE && mo.getNumRows()==_N ) ||
				(moDpf==PDataPartitionFormat.COLUMN_WISE && mo.getNumColumns()==_N)) )
			{
				int k = (int)Math.min(_N,_rk2);
				
				pn.addParam(ParamType.DATA_PARTITIONER, "REMOTE_MR(fused)");
				pn.setK( k );
				
				pfpb.setExecMode(PExecMode.REMOTE_MR_DP); //set fused exec type	
				pfpb.setDataPartitioner(PDataPartitioner.NONE);
				pfpb.enableColocatedPartitionedMatrix( moVarname ); 
				pfpb.setDegreeOfParallelism(k);
				
				apply = true;
			}
		}
		
		LOG.debug(getOptMode()+" OPT: rewrite 'set fused data partitioning and execution' - result="+apply );
	}
	
	/**
	 * 
	 * @param n
	 * @param iterVarname
	 * @return
	 * @throws DMLRuntimeException
	 */
	protected boolean rIsAccessByIterationVariable( OptNode n, String varName, String iterVarname ) 
		throws DMLRuntimeException
	{
		boolean ret = true;
		
		if( !n.isLeaf() )
		{
			for( OptNode cn : n.getChilds() )
				rIsAccessByIterationVariable( cn, varName, iterVarname );
		}
		else if(    n.getNodeType()== NodeType.HOP
			     && n.getParam(ParamType.OPSTRING).equals(IndexingOp.OPSTRING)
			     && n.getParam(ParamType.DATA_PARTITION_FORMAT) != null )
		{
			PDataPartitionFormat dpf = PDataPartitionFormat.valueOf(n.getParam(ParamType.DATA_PARTITION_FORMAT));
			Hop h = OptTreeConverter.getAbstractPlanMapping().getMappedHop(n.getID());
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
			
			ret &= (   (inMatrix!=null && inMatrix.equals(varName)) 
				    && (indexAccess!=null && indexAccess.equals(iterVarname)));
		}
		
		return ret;
	}
	
	///////
	//REWRITE transpose sparse vector operations
	///
	
	protected void rewriteSetTranposeSparseVectorOperations(OptNode pn, HashMap<String, PDataPartitionFormat> partitionedMatrices, LocalVariableMap vars) 
		throws DMLRuntimeException 
	{
		//assertions (warnings of corrupt optimizer decisions)
		if( pn.getNodeType() != NodeType.PARFOR )
			LOG.warn(getOptMode()+" OPT: Transpose sparse vector operations is only applicable for a ParFor node.");
		
		boolean apply = false;
		
		ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
                .getAbstractPlanMapping().getMappedProg(pn.getID())[1];
		
		if(    pfpb.getExecMode() == PExecMode.REMOTE_MR_DP 
			&& partitionedMatrices.size()==1 ) //general applicable
		{
			String moVarname = partitionedMatrices.keySet().iterator().next();
			PDataPartitionFormat moDpf = partitionedMatrices.get(moVarname);
			Data dat = vars.get(moVarname);
			
			if(    dat !=null && dat instanceof MatrixObject 
				&& moDpf == PDataPartitionFormat.COLUMN_WISE	
				&& ((MatrixObject)dat).getSparsity()<= MatrixBlock.SPARSITY_TURN_POINT  //check for sparse matrix
				&& rIsTransposeSafePartition(pn, moVarname) ) //tranpose-safe
			{
				pfpb.setTransposeSparseColumnVector( true );
				
				apply = true;
			}
		}
		
		LOG.debug(getOptMode()+" OPT: rewrite 'set transpose sparse vector operations' - result="+apply );			
	}
	
	/**
	 * 
	 * @param n
	 * @param iterVarname
	 * @return
	 * @throws DMLRuntimeException
	 */
	protected boolean rIsTransposeSafePartition( OptNode n, String varName ) 
		throws DMLRuntimeException
	{
		boolean ret = true;
		
		if( !n.isLeaf() )
		{
			for( OptNode cn : n.getChilds() )
				rIsTransposeSafePartition( cn, varName );
		}
		else if(    n.getNodeType()== NodeType.HOP
			     && n.getParam(ParamType.OPSTRING).equals(IndexingOp.OPSTRING)
			     && n.getParam(ParamType.DATA_PARTITION_FORMAT) != null )
		{
			Hop h = OptTreeConverter.getAbstractPlanMapping().getMappedHop(n.getID());
			
			String inMatrix = h.getInput().get(0).get_name();
			if( inMatrix.equals(varName) )
			{
				//check that all parents are transpose-safe operations
				//(even a transient write would not be safe due to indirection into other DAGs)			
				ArrayList<Hop> parent = h.getParent();
				for( Hop p : parent )
					ret &= p.isTransposeSafe();
			}
		}
		
		return ret;
	}
	
	
	///////
	//REWRITE set in-place result indexing
	///
	
	/**
	 * 
	 * @param pn
	 * @param M
	 * @param vars
	 * @param inPlaceResultVars
	 * @throws DMLRuntimeException
	 */
	protected void rewriteSetInPlaceResultIndexing(OptNode pn, double M, LocalVariableMap vars, HashSet<String> inPlaceResultVars) 
		throws DMLRuntimeException 
	{
		//assertions (warnings of corrupt optimizer decisions)
		if( pn.getNodeType() != NodeType.PARFOR )
			LOG.warn(getOptMode()+" OPT: Transpose sparse vector operations is only applicable for a ParFor node.");
		
		boolean apply = false;

		ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
              .getAbstractPlanMapping().getMappedProg(pn.getID())[1];
		
		//note currently we decide for all result vars jointly, i.e.,
		//only if all fit pinned in remaining budget, we apply this rewrite.
		
		ArrayList<String> retVars = pfpb.getResultVariables();
		
		//compute total sum of pinned result varible memory
		double sum = computeTotalSizeResultVariables(retVars, vars);
		
		//NOTE: currently this rule is too conservative (the result variable is assumed to be dense and
		//most importantly counted twice if this is part of the maximum operation)
		double totalMem = Math.min((M+sum), rComputeSumMemoryIntermediates(pn, new HashSet<String>()));
		
		if(   (pfpb.getExecMode() == PExecMode.REMOTE_MR_DP || pfpb.getExecMode() == PExecMode.REMOTE_MR) 
			&& totalMem < _rm //max operator and sum of result vars fit
			&& rHasOnlyInPlaceSafeLeftIndexing(pn, retVars) )
		{
			apply = true;
			
			//add result vars to result and set state
			//will be serialized and transfered via symbol table 
			for( String var : retVars ){
				Data dat = vars.get(var);
				if( dat instanceof MatrixObject )
					((MatrixObject)dat).enableUpdateInPlace(true);
			}
			inPlaceResultVars.addAll(retVars);
		}
		
		LOG.debug(getOptMode()+" OPT: rewrite 'set in-place result indexing' - result="+
		          apply+" ("+ProgramConverter.serializeStringCollection(inPlaceResultVars)+", M="+toMB(M+sum)+")" );	
	}
	
	/**
	 * 
	 * @param n
	 * @param retVars
	 * @return
	 * @throws DMLRuntimeException
	 */
	protected boolean rHasOnlyInPlaceSafeLeftIndexing( OptNode n, ArrayList<String> retVars ) 
		throws DMLRuntimeException
	{
		boolean ret = true;
		
		if( !n.isLeaf() )
		{
			for( OptNode cn : n.getChilds() )
				ret &= rHasOnlyInPlaceSafeLeftIndexing( cn, retVars );
		}
		else if(    n.getNodeType()== NodeType.HOP
			     && n.getParam(ParamType.OPSTRING).equals(LeftIndexingOp.OPSTRING) )
		{
			Hop h = OptTreeConverter.getAbstractPlanMapping().getMappedHop(n.getID());
			if( retVars.contains( h.getInput().get(0).get_name() ) )
			{
				ret &= (h.getParent().size()==1 
						&& h.getParent().get(0).get_name().equals(h.getInput().get(0).get_name()));
			}
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @param retVars
	 * @param vars
	 * @return
	 */
	private double computeTotalSizeResultVariables(ArrayList<String> retVars, LocalVariableMap vars)
	{
		double sum = 1;
		for( String var : retVars ){
			Data dat = vars.get(var);
			if( dat instanceof MatrixObject )
			{
				MatrixObject mo = (MatrixObject)dat;
				sum += OptimizerUtils.estimateSizeExactSparsity(mo.getNumRows(), mo.getNumColumns(), 1.0);	
			} 
		}
		
		return sum;
	}
	
	///////
	//REWRITE disable CP caching  
	///
	
	/**
	 * 
	 * @param pn
	 * @param inplaceResultVars
	 * @param vars
	 * @throws DMLRuntimeException
	 */
	protected void rewriteDisableCPCaching(OptNode pn, HashSet<String> inplaceResultVars, LocalVariableMap vars) 
		throws DMLRuntimeException 
	{
		//assertions (warnings of corrupt optimizer decisions)
		if( pn.getNodeType() != NodeType.PARFOR )
			LOG.warn(getOptMode()+" OPT: Disable caching is only applicable for a ParFor node.");
		
		
		ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
              .getAbstractPlanMapping().getMappedProg(pn.getID())[1];
		
		double M_sumInterm = rComputeSumMemoryIntermediates(pn, inplaceResultVars);
		boolean apply = false;
		
		if( (pfpb.getExecMode() == PExecMode.REMOTE_MR_DP || pfpb.getExecMode() == PExecMode.REMOTE_MR)
			&& M_sumInterm < _rm ) //all intermediates and operations fit into memory budget
		{
			pfpb.setCPCaching(false); //default is true			
			apply = true;
		}
		
		LOG.debug(getOptMode()+" OPT: rewrite 'disable CP caching' - result="+apply+" (M="+toMB(M_sumInterm)+")" );			
	}
	
	/**
	 * 
	 * @param n
	 * @param inplaceResultVars 
	 * @return
	 * @throws DMLRuntimeException
	 */
	protected double rComputeSumMemoryIntermediates( OptNode n, HashSet<String> inplaceResultVars ) 
		throws DMLRuntimeException
	{
		double sum = 0;
		
		if( !n.isLeaf() )
		{
			for( OptNode cn : n.getChilds() )
				sum += rComputeSumMemoryIntermediates( cn, inplaceResultVars );
		}
		else if(    n.getNodeType()== NodeType.HOP )
		{
			Hop h = OptTreeConverter.getAbstractPlanMapping().getMappedHop(n.getID());
			
			if(    n.getParam(ParamType.OPSTRING).equals(IndexingOp.OPSTRING)
				&& n.getParam(ParamType.DATA_PARTITION_FORMAT) != null )
			{
				//set during partitioning rewrite
				sum += h.getMemEstimate();
			}
			else
			{
				//base intermediate (worst-case w/ materialized intermediates)
				sum +=   h.getOutputMemEstimate()
					   + h.getIntermediateMemEstimate(); 

				//inputs not represented in the planopttree (worst-case no CSE)
				if( h.getInput() != null )
					for( Hop cn : h.getInput() )
						if( cn instanceof DataOp && ((DataOp)cn).isRead()  //read data
							&& !inplaceResultVars.contains(cn.get_name())) //except in-place result vars
						{
							sum += cn.getMemEstimate();	
						}
			}
		}
		
		return sum;
	}
	
	///////
	//REWRITE enable runtime piggybacking
	///
	
	/**
	 * 
	 * @param n
	 * @param partitionedMatrices.keySet() 
	 * @param vars 
	 * @throws DMLRuntimeException
	 */
	protected void rewriteEnableRuntimePiggybacking( OptNode n, LocalVariableMap vars, HashMap<String, PDataPartitionFormat> partitionedMatrices ) 
		throws DMLRuntimeException
	{
		ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
								    .getAbstractPlanMapping().getMappedProg(n.getID())[1];

		HashSet<String> sharedVars = new HashSet<String>();
		boolean apply = false; 
		
		//enable runtime piggybacking if MR jobs on shared read-only data set
		if( OptimizerUtils.ALLOW_RUNTIME_PIGGYBACKING )
		{
			//apply runtime piggybacking if hop in mr and shared input variable 
			//(any input variabled which is not partitioned and is read only and applies)
			apply = rHasSharedMRInput(n, vars.keySet(), partitionedMatrices.keySet(), sharedVars)
					&& n.getTotalK() > 1; //apply only if degree of parallelism > 1
		}
		
		if( apply )
			pfpb.setRuntimePiggybacking(apply);
		
		_numEvaluatedPlans++;
		LOG.debug(getOptMode()+" OPT: rewrite 'enable runtime piggybacking' - result="+apply+
				" ("+ProgramConverter.serializeStringCollection(sharedVars)+")" );
	}
	
	/**
	 * 
	 * @param n
	 * @param inputVars
	 * @param partitionedVars
	 * @return
	 * @throws DMLRuntimeException
	 */
	protected boolean rHasSharedMRInput( OptNode n, Set<String> inputVars, Set<String> partitionedVars, HashSet<String> sharedVars ) 
		throws DMLRuntimeException
	{
		boolean ret = false;
		
		if( !n.isLeaf() )
		{
			for( OptNode cn : n.getChilds() )
				ret |= rHasSharedMRInput( cn, inputVars, partitionedVars, sharedVars );
		}
		else if( n.getNodeType()== NodeType.HOP && n.getExecType()==ExecType.MR )
		{
			Hop h = OptTreeConverter.getAbstractPlanMapping().getMappedHop(n.getID());
			for( Hop ch : h.getInput() )
			{
				//note: we replaxed the contraint of non-partitioned inputs for additional 
				//latecy hiding and scan sharing of partitions which are read multiple times
				
				if(    ch instanceof DataOp && ch.get_dataType() == DataType.MATRIX
					&& inputVars.contains(ch.get_name()) )
					//&& !partitionedVars.contains(ch.get_name()))
				{
					ret = true;
					sharedVars.add(ch.get_name());
				}
				else if(    ch instanceof ReorgOp && ((ReorgOp)ch).getOp()==ReOrgOp.TRANSPOSE 
					&& ch.getInput().get(0) instanceof DataOp && ch.getInput().get(0).get_dataType() == DataType.MATRIX
					&& inputVars.contains(ch.getInput().get(0).get_name()) )
					//&& !partitionedVars.contains(ch.getInput().get(0).get_name()))
				{
					ret = true;
					sharedVars.add(ch.getInput().get(0).get_name());
				}
			}
		}

		return ret;
	}

	
	///////
	//REWRITE set result merge
	///
	
	/**
	 *
	 * 
	 * @param n
	 * @throws DMLRuntimeException 
	 */
	protected void rewriteSetResultMerge( OptNode n, LocalVariableMap vars, boolean inLocal ) 
		throws DMLRuntimeException
	{
		ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
								    .getAbstractPlanMapping().getMappedProg(n.getID())[1];
		
		PResultMerge ret = null;
		
		//investigate details of current parfor node
		boolean flagMRParFOR = (n.getExecType() == ExecType.MR);
		boolean flagLargeResult = hasLargeTotalResults( n, pfpb.getResultVariables(), vars, true );
		boolean flagMRLeftIndexing = hasResultMRLeftIndexing( n, pfpb.getResultVariables(), vars, true );
		boolean flagCellFormatWoCompare = determineFlagCellFormatWoCompare(pfpb.getResultVariables(), vars); 
		boolean flagOnlyInMemResults = hasOnlyInMemoryResults(n, pfpb.getResultVariables(), vars, true );
		
		//optimimality decision on result merge
		//MR, if remote exec, and w/compare (prevent huge transfer/merge costs)
		if( flagMRParFOR && flagLargeResult )
		{
			ret = PResultMerge.REMOTE_MR;
		}
		//CP, if all results in mem	
		else if( flagOnlyInMemResults )
		{
			ret = PResultMerge.LOCAL_MEM;
		}
		//MR, if result partitioning and copy not possible
		//NOTE: 'at least one' instead of 'all' condition of flagMRLeftIndexing because the 
		//      benefit for large matrices outweigths potentially unnecessary MR jobs for smaller matrices)
		else if(    ( flagMRParFOR || flagMRLeftIndexing) 
			    && !(flagCellFormatWoCompare && ResultMergeLocalFile.ALLOW_COPY_CELLFILES ) )
		{
			ret = PResultMerge.REMOTE_MR;
		}
		//CP, otherwise (decide later if in mem or file-based)
		else
		{
			ret = PResultMerge.LOCAL_AUTOMATIC;
		}
		
		// modify rtprog	
		pfpb.setResultMerge(ret);
			
		// modify plan
		n.addParam(ParamType.RESULT_MERGE, ret.toString());			

		//recursively apply rewrite for parfor nodes
		if( n.getChilds() != null )
			rInvokeSetResultMerge(n.getChilds(), vars, inLocal && !flagMRParFOR);
		
		_numEvaluatedPlans++;
		LOG.debug(getOptMode()+" OPT: rewrite 'set result merge' - result="+ret );
	}
	
	/**
	 * 
	 * @param resultVars
	 * @param vars
	 * @return
	 */
	protected boolean determineFlagCellFormatWoCompare( ArrayList<String> resultVars, LocalVariableMap vars  )
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
	protected boolean hasResultMRLeftIndexing( OptNode n, ArrayList<String> resultVars, LocalVariableMap vars, boolean checkSize ) 
		throws DMLRuntimeException
	{
		boolean ret = false;
		
		if( n.isLeaf() )
		{
			String opName = n.getParam(ParamType.OPSTRING);
			//check opstring and exec type
			if( opName !=null && opName.equals(LeftIndexingOp.OPSTRING) && n.getExecType()==ExecType.MR )
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
						ret = !isInMemoryResultMerge(rows, cols, OptimizerUtils.getRemoteMemBudgetMap(false));
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

	/**
	 * Heuristically compute total result sizes, if larger than local mem budget assumed to be large.
	 * 
	 * @param n
	 * @param resultVars
	 * @param vars
	 * @param checkSize
	 * @return
	 * @throws DMLRuntimeException
	 */
	protected boolean hasLargeTotalResults( OptNode pn, ArrayList<String> resultVars, LocalVariableMap vars, boolean checkSize ) 
		throws DMLRuntimeException
	{
		double totalSize = 0;
		
		//get num tasks according to task partitioning 
		PTaskPartitioner tp = PTaskPartitioner.valueOf(pn.getParam(ParamType.TASK_PARTITIONER));
		int k = pn.getK();
		long W = estimateNumTasks(tp, _N, k); 
		
		for( String var : resultVars )
		{
			//Potential unknowns: for local result var of child parfor (but we're only interested in top level)
			//Potential scalars: for disabled dependency analysis and unbounded scoping			
			Data dat = vars.get( var );
			if( dat != null && dat instanceof MatrixObject ) 
			{
				MatrixObject mo = (MatrixObject) vars.get( var );
				
				long rows = mo.getNumRows();
				long cols = mo.getNumColumns();	
				long nnz = mo.getNnz();
				
				if( nnz > 0 ) //w/ compare
				{
					totalSize += W * OptimizerUtils.estimateSizeExactSparsity(rows, cols, 1.0);
				}
				else //in total at most as dimensions (due to disjoint results)
				{
					totalSize += OptimizerUtils.estimateSizeExactSparsity(rows, cols, 1.0);
				}
			}
		}
		
		return ( totalSize >= _lm ); //heuristic:  large if >= local mem budget 
	}
	
	/**
	 * 
	 * @param tp
	 * @param N
	 * @param k
	 * @return
	 */
	protected long estimateNumTasks( PTaskPartitioner tp, long N, int k )
	{
		long W = -1;
		
		switch( tp )
		{
			case NAIVE:
			case FIXED:            W = N; break; 
			case STATIC:           W = N / k; break;
			case FACTORING:
			case FACTORING_CMIN:
			case FACTORING_CMAX:   W = k * (long)(Math.log(((double)N)/k)/Math.log(2.0)); break;
			default:               W = N; break; //N as worst case estimate
		}
		
		return W;
	}
	
	/**
	 * 
	 * @param n
	 * @param resultVars
	 * @param vars
	 * @return
	 * @throws DMLRuntimeException
	 */
	protected boolean hasOnlyInMemoryResults( OptNode n, ArrayList<String> resultVars, LocalVariableMap vars, boolean inLocal ) 
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
					double memBudget = inLocal ? OptimizerUtils.getLocalMemBudget() : 
						                         OptimizerUtils.getRemoteMemBudgetMap();
					ret &= isInMemoryResultMerge(rows, cols, memBudget);
				}
			}
		}
		else
		{
			for( OptNode c : n.getChilds() )
				ret &= hasOnlyInMemoryResults(c, resultVars, vars, inLocal);
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @param nodes
	 * @param vars
	 * @throws DMLRuntimeException 
	 */
	protected void rInvokeSetResultMerge( Collection<OptNode> nodes, LocalVariableMap vars, boolean inLocal) 
		throws DMLRuntimeException
	{
		for( OptNode n : nodes )
			if( n.getNodeType() == NodeType.PARFOR )
			{
				rewriteSetResultMerge(n, vars, inLocal);
				if( n.getExecType()==ExecType.MR )
					inLocal = false;
			}
			else if( n.getChilds()!=null )  
				rInvokeSetResultMerge(n.getChilds(), vars, inLocal);
	}
	
	/**
	 * 
	 * @param rows
	 * @param cols
	 * @return
	 */
	public static boolean isInMemoryResultMerge( long rows, long cols, double memBudget )
	{
		if( !ParForProgramBlock.USE_PARALLEL_RESULT_MERGE )
		{
			//1/4 mem budget because: 2xout (incl sparse-dense change), 1xin, 1xcompare  
			return ( rows>=0 && cols>=0 && MatrixBlock.estimateSizeInMemory(rows, cols, 1.0) < memBudget/4 );
		}
		else
			return ( rows>=0 && cols>=0 && rows*cols < Math.pow(Hop.CPThreshold, 2) );
	}

	
	///////
	//REWRITE set recompile memory budget
	///

	/**
	 * 
	 * @param n
	 * @param M
	 */
	protected void rewriteSetRecompileMemoryBudget( OptNode n )
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
		
		_numEvaluatedPlans++;
		LOG.debug(getOptMode()+" OPT: rewrite 'set recompile memory budget' - result="+toMB(newLocalMem) );
	}	
	
	
	///////
	//REWRITE remove recursive parfor
	///
	
	/**
	 * 
	 * @param n
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	protected void rewriteRemoveRecursiveParFor(OptNode n, LocalVariableMap vars) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		int count = 0; //num removed parfor
		
		//find recursive parfor
		HashSet<ParForProgramBlock> recPBs = new HashSet<ParForProgramBlock>();
		rFindRecursiveParFor( n, recPBs, false );

		if( recPBs.size() > 0 )
		{
			//unfold if necessary
			try 
			{
				ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
		        							.getAbstractPlanMapping().getMappedProg(n.getID())[1];
				if( recPBs.contains(pfpb) ) 
					rFindAndUnfoldRecursiveFunction(n, pfpb, recPBs, vars);
			}
			catch(Exception ex)
			{
				throw new DMLRuntimeException(ex);
			}
			
			//remove recursive parfor (parfor to for)
			count = removeRecursiveParFor(n, recPBs);
		}
		
		_numEvaluatedPlans++;
		LOG.debug(getOptMode()+" OPT: rewrite 'remove recursive parfor' - result="+recPBs.size()+"/"+count );
	}
	
	/**
	 * 
	 * @param n
	 * @param cand
	 * @param recContext
	 * @return
	 */
	protected void rFindRecursiveParFor( OptNode n, HashSet<ParForProgramBlock> cand, boolean recContext )
	{
		//recursive invocation
		if( !n.isLeaf() )
			for( OptNode c : n.getChilds() )
			{
				if( c.getNodeType() == NodeType.FUNCCALL && c.isRecursive() )
					rFindRecursiveParFor(c, cand, true);
				else
					rFindRecursiveParFor(c, cand, recContext);
			}
		
		//add candidate program blocks
		if( recContext && n.getNodeType()==NodeType.PARFOR )
		{
			ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
									    .getAbstractPlanMapping().getMappedProg(n.getID())[1];
			cand.add(pfpb);
		}
	}
	
	/**
	 * 
	 * @param n
	 * @param parfor
	 * @param recPBs
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 * @throws HopsException
	 * @throws LanguageException
	 */
	protected void rFindAndUnfoldRecursiveFunction( OptNode n, ParForProgramBlock parfor, HashSet<ParForProgramBlock> recPBs, LocalVariableMap vars )
		throws DMLRuntimeException, DMLUnsupportedOperationException, HopsException, LanguageException
	{
		//unfold if found
		if( n.getNodeType() == NodeType.FUNCCALL && n.isRecursive())
		{
			boolean exists = rContainsNode(n, parfor);
			if( exists )
			{
				String fnameKey = n.getParam(ParamType.OPSTRING);
				String[] names = fnameKey.split(Program.KEY_DELIM);
				String fnamespace = names[0];
				String fname = names[1];
				String fnameNew = FUNCTION_UNFOLD_NAMEPREFIX + fname;
				
				//unfold function
				FunctionOp fop = (FunctionOp) OptTreeConverter.getAbstractPlanMapping().getMappedHop(n.getID());
				Program prog = parfor.getProgram();
				DMLProgram dmlprog = parfor.getStatementBlock().getDMLProg();
				FunctionProgramBlock fpb = prog.getFunctionProgramBlock(fnamespace, fname);	
				FunctionStatementBlock fsb = dmlprog.getFunctionStatementBlock(fnamespace, fname);
				FunctionProgramBlock copyfpb = ProgramConverter.createDeepCopyFunctionProgramBlock(fpb, new HashSet<String>());
				prog.addFunctionProgramBlock(fnamespace, fnameNew, copyfpb);
				dmlprog.addFunctionStatementBlock(fnamespace, fnameNew, (FunctionStatementBlock)copyfpb.getStatementBlock());
				
				//replace function names in old subtree (link to new function)
				rReplaceFunctionNames(n, fname, fnameNew);
				
				//recreate sub opttree
				String fnameNewKey = fnamespace + Program.KEY_DELIM + fnameNew;
				OptNode nNew = new OptNode(NodeType.FUNCCALL);
				OptTreeConverter.getAbstractPlanMapping().putHopMapping(fop, nNew);
				nNew.setExecType(ExecType.CP);
				nNew.addParam(ParamType.OPSTRING, fnameNewKey);
				long parentID = OptTreeConverter.getAbstractPlanMapping().getMappedParentID(n.getID());
				OptTreeConverter.getAbstractPlanMapping().getOptNode(parentID).exchangeChild(n, nNew);
				HashSet<String> memo = new HashSet<String>();
				memo.add(fnameKey); //required if functionop not shared (because not replaced yet)
				memo.add(fnameNewKey); //requied if functionop shared (indirectly replaced)
				for( int i=0; i<copyfpb.getChildBlocks().size() /*&& i<len*/; i++ )
				{
					ProgramBlock lpb = copyfpb.getChildBlocks().get(i);
					StatementBlock lsb = lpb.getStatementBlock();
					nNew.addChild( OptTreeConverter.rCreateAbstractOptNode(lsb,lpb,vars,false, memo) );
				}
				
				//compute delta for recPB set (use for removing parfor)
				recPBs.removeAll( rGetAllParForPBs(n, new HashSet<ParForProgramBlock>()) );
				recPBs.addAll( rGetAllParForPBs(nNew, new HashSet<ParForProgramBlock>()) );
				
				//replace function names in new subtree (recursive link to new function)
				rReplaceFunctionNames(nNew, fname, fnameNew);
				
			}
			//else, we can return anyway because we will not find that parfor
			
			return;
		}
		
		//recursive invocation (only for non-recursive functions)
		if( !n.isLeaf() )
			for( OptNode c : n.getChilds() )
				rFindAndUnfoldRecursiveFunction(c, parfor, recPBs, vars);
	}
	
	/**
	 * 
	 * @param n
	 * @param parfor
	 * @return
	 */
	protected boolean rContainsNode( OptNode n, ParForProgramBlock parfor )
	{
		boolean ret = false;
		
		if( n.getNodeType() == NodeType.PARFOR )
		{
			ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
		    						.getAbstractPlanMapping().getMappedProg(n.getID())[1];	
			ret = (parfor == pfpb);
		}
		
		if( !ret && !n.isLeaf() )
			for( OptNode c : n.getChilds() ) {
				ret |= rContainsNode(c, parfor);
				if( ret ) break; //early abort
			}
		
		return ret;
	}
	
	/**
	 * 
	 * @param n
	 * @param pbs
	 * @return
	 */
	protected HashSet<ParForProgramBlock> rGetAllParForPBs( OptNode n, HashSet<ParForProgramBlock> pbs )
	{
		//collect parfor
		if( n.getNodeType()==NodeType.PARFOR )
		{
			ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
									.getAbstractPlanMapping().getMappedProg(n.getID())[1];
			pbs.add(pfpb);
		}
		
		//recursive invocation
		if( !n.isLeaf() )
			for( OptNode c : n.getChilds() )
				rGetAllParForPBs(c, pbs);
		
		return pbs;
	}
	
	/**
	 * 
	 * @param n
	 * @param oldName
	 * @param newName
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 * @throws HopsException 
	 */
	protected void rReplaceFunctionNames( OptNode n, String oldName, String newName ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException, HopsException
	{
		if( n.getNodeType() == NodeType.FUNCCALL)
		{
			FunctionOp fop = (FunctionOp) OptTreeConverter.getAbstractPlanMapping().getMappedHop(n.getID());	
			
			String[] names = n.getParam(ParamType.OPSTRING).split(Program.KEY_DELIM);
			String fnamespace = names[0];
			String fname = names[1];
			
			if( fname.equals(oldName) || fname.equals(newName) ) //newName if shared hop
			{
				//set opttree function name
				n.addParam(ParamType.OPSTRING, DMLProgram.constructFunctionKey(fnamespace,newName));
				
				//set instruction function name
				long parentID = OptTreeConverter.getAbstractPlanMapping().getMappedParentID(n.getID());	
				ProgramBlock pb = (ProgramBlock) OptTreeConverter.getAbstractPlanMapping().getMappedProg(parentID)[1];
				ArrayList<Instruction> instArr = pb.getInstructions();				
				for( int i=0; i<instArr.size(); i++ )
				{
					Instruction inst = instArr.get(i);
					if( inst instanceof FunctionCallCPInstruction ) 
					{
						FunctionCallCPInstruction fci = (FunctionCallCPInstruction) inst;
						if( oldName.equals(fci.getFunctionName()) )
							instArr.set(i, FunctionCallCPInstruction.parseInstruction(fci.toString().replaceAll(oldName, newName)));
					}
				}
				
				//set hop name (for recompile)
				if( fop.getFunctionName().equals(oldName) )
					fop.setFunctionName(newName);
			}
		}
	
		//recursive invocation
		if( !n.isLeaf() )
			for( OptNode c : n.getChilds() )
				rReplaceFunctionNames(c, oldName, newName);
	}
	
	/**
	 * 
	 * @param n
	 * @param recPBs
	 * @return
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 */
	protected int removeRecursiveParFor( OptNode n, HashSet<ParForProgramBlock> recPBs ) 
		throws DMLUnsupportedOperationException, DMLRuntimeException
	{
		int count = 0;
		
		if( !n.isLeaf() )
		{
			for( OptNode sub : n.getChilds() )
			{
				if( sub.getNodeType() == NodeType.PARFOR )
				{
					long id = sub.getID();
					Object[] progobj = OptTreeConverter.getAbstractPlanMapping().getMappedProg(id);
					ParForStatementBlock pfsb = (ParForStatementBlock)progobj[0];
					ParForProgramBlock pfpb = (ParForProgramBlock)progobj[1];
					
					if( recPBs.contains(pfpb) )
					{
						//create for pb as replacement
						Program prog = pfpb.getProgram();
						ForProgramBlock fpb = ProgramConverter.createShallowCopyForProgramBlock(pfpb, prog);

						//replace parfor with for, and update objectmapping
						OptTreeConverter.replaceProgramBlock(n, sub, pfpb, fpb, false);
						//update link to statement block
						fpb.setStatementBlock(pfsb);
							
						//update node
						sub.setNodeType(NodeType.FOR);
						sub.setK(1);
						
						count++;
					}
				}
				
				count += removeRecursiveParFor(sub, recPBs);
			}
		}
		
		return count;
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
	protected void rewriteRemoveUnnecessaryParFor(OptNode n) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		int count = removeUnnecessaryParFor( n );
		
		_numEvaluatedPlans++;
		LOG.debug(getOptMode()+" OPT: rewrite 'remove unnecessary parfor' - result="+count );
	}
	
	/**
	 * 
	 * @param n
	 * @return
	 * @throws DMLUnsupportedOperationException
	 * @throws DMLRuntimeException
	 */
	protected int removeUnnecessaryParFor( OptNode n ) 
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
					Object[] progobj = OptTreeConverter.getAbstractPlanMapping().getMappedProg(id);
					ParForStatementBlock pfsb = (ParForStatementBlock)progobj[0];
					ParForProgramBlock pfpb = (ParForProgramBlock)progobj[1];
					
					//create for pb as replacement
					Program prog = pfpb.getProgram();
					ForProgramBlock fpb = ProgramConverter.createShallowCopyForProgramBlock(pfpb, prog);
					
					//replace parfor with for, and update objectmapping
					OptTreeConverter.replaceProgramBlock(n, sub, pfpb, fpb, false);
					//update link to statement block
					fpb.setStatementBlock(pfsb);
					
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
