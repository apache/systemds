/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysml.runtime.controlprogram.parfor.opt;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.conf.DMLConfig;
import org.apache.sysml.hops.AggBinaryOp;
import org.apache.sysml.hops.DataOp;
import org.apache.sysml.hops.FunctionOp;
import org.apache.sysml.hops.Hop;
import org.apache.sysml.hops.AggBinaryOp.MMultMethod;
import org.apache.sysml.hops.Hop.DataOpTypes;
import org.apache.sysml.hops.Hop.MultiThreadedHop;
import org.apache.sysml.hops.Hop.ParamBuiltinOp;
import org.apache.sysml.hops.Hop.ReOrgOp;
import org.apache.sysml.hops.HopsException;
import org.apache.sysml.hops.IndexingOp;
import org.apache.sysml.hops.LeftIndexingOp;
import org.apache.sysml.hops.LiteralOp;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.hops.ParameterizedBuiltinOp;
import org.apache.sysml.hops.ReorgOp;
import org.apache.sysml.hops.rewrite.HopRewriteUtils;
import org.apache.sysml.hops.rewrite.ProgramRewriteStatus;
import org.apache.sysml.hops.rewrite.ProgramRewriter;
import org.apache.sysml.hops.rewrite.RewriteInjectSparkLoopCheckpointing;
import org.apache.sysml.hops.recompile.Recompiler;
import org.apache.sysml.lops.LopProperties;
import org.apache.sysml.lops.LopsException;
import org.apache.sysml.parser.DMLProgram;
import org.apache.sysml.parser.Expression;
import org.apache.sysml.parser.Expression.DataType;
import org.apache.sysml.parser.FunctionStatementBlock;
import org.apache.sysml.parser.LanguageException;
import org.apache.sysml.parser.ParForStatement;
import org.apache.sysml.parser.ParForStatementBlock;
import org.apache.sysml.parser.StatementBlock;
import org.apache.sysml.parser.VariableSet;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.DMLUnsupportedOperationException;
import org.apache.sysml.runtime.controlprogram.ForProgramBlock;
import org.apache.sysml.runtime.controlprogram.FunctionProgramBlock;
import org.apache.sysml.runtime.controlprogram.IfProgramBlock;
import org.apache.sysml.runtime.controlprogram.LocalVariableMap;
import org.apache.sysml.runtime.controlprogram.ParForProgramBlock;
import org.apache.sysml.runtime.controlprogram.Program;
import org.apache.sysml.runtime.controlprogram.ProgramBlock;
import org.apache.sysml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import org.apache.sysml.runtime.controlprogram.ParForProgramBlock.PDataPartitioner;
import org.apache.sysml.runtime.controlprogram.ParForProgramBlock.PExecMode;
import org.apache.sysml.runtime.controlprogram.ParForProgramBlock.POptMode;
import org.apache.sysml.runtime.controlprogram.ParForProgramBlock.PResultMerge;
import org.apache.sysml.runtime.controlprogram.ParForProgramBlock.PTaskPartitioner;
import org.apache.sysml.runtime.controlprogram.WhileProgramBlock;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.context.SparkExecutionContext;
import org.apache.sysml.runtime.controlprogram.parfor.ProgramConverter;
import org.apache.sysml.runtime.controlprogram.parfor.ResultMergeLocalFile;
import org.apache.sysml.runtime.controlprogram.parfor.opt.OptNode.ExecType;
import org.apache.sysml.runtime.controlprogram.parfor.opt.OptNode.NodeType;
import org.apache.sysml.runtime.controlprogram.parfor.opt.OptNode.ParamType;
import org.apache.sysml.runtime.controlprogram.parfor.opt.PerfTestTool.TestMeasure;
import org.apache.sysml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysml.runtime.instructions.Instruction;
import org.apache.sysml.runtime.instructions.cp.Data;
import org.apache.sysml.runtime.instructions.cp.FunctionCallCPInstruction;
import org.apache.sysml.runtime.instructions.spark.data.RDDObject;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.MatrixFormatMetaData;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.matrix.data.SparseRow;
import org.apache.sysml.yarn.ropt.YarnClusterAnalyzer;

/**
 * Rule-Based ParFor Optimizer (time: O(n)):
 * 
 * Applied rule-based rewrites
 * - 1) rewrite set data partitioner (incl. recompile RIX)
 * - 2) rewrite remove unnecessary compare matrix
 * - 3) rewrite result partitioning (incl. recompile LIX)
 * - 4) rewrite set execution strategy
 * - 5) rewrite set operations exec type (incl. recompile)
 * - 6) rewrite use data colocation		 
 * - 7) rewrite set partition replication factor
 * - 8) rewrite set export replication factor 
 * - 9) rewrite use nested parallelism 
 * - 10) rewrite set degree of parallelism
 * - 11) rewrite set task partitioner
 * - 12) rewrite set fused data partitioning and execution
 * - 13) rewrite transpose vector operations (for sparse)
 * - 14) rewrite set in-place result indexing
 * - 15) rewrite disable caching (prevent sparse serialization)
 * - 16) rewrite enable runtime piggybacking
 * - 17) rewrite inject spark loop checkpointing 
 * - 18) rewrite inject spark repartition (for zipmm)
 * - 19) rewrite set spark eager rdd caching 
 * - 20) rewrite set result merge 		 		 
 * - 21) rewrite set recompile memory budget
 * - 22) rewrite remove recursive parfor	
 * - 23) rewrite remove unnecessary parfor		
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
	
	public static final double PROB_SIZE_THRESHOLD_REMOTE = 100; //wrt # top-level iterations (min)
	public static final double PROB_SIZE_THRESHOLD_PARTITIONING = 2; //wrt # top-level iterations (min)
	public static final double PROB_SIZE_THRESHOLD_MB = 256*1024*1024; //wrt overall memory consumption (min)
	public static final int MAX_REPLICATION_FACTOR_PARTITIONING = 5;     
	public static final int MAX_REPLICATION_FACTOR_EXPORT = 7;    
	public static final boolean ALLOW_REMOTE_NESTED_PARALLELISM = false;
	public static final boolean APPLY_REWRITE_NESTED_PARALLELISM = false;
	public static final String FUNCTION_UNFOLD_NAMEPREFIX = "__unfold_";
	
	public static final boolean APPLY_REWRITE_UPDATE_INPLACE_INTERMEDIATE = true;
	
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
	
	protected static ThreadLocal<ArrayList<String>> listUIPRes = new ThreadLocal<ArrayList<String>>() {
		@Override protected ArrayList<String> initialValue() { return new ArrayList<String>(); }
	};

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
		LOG.debug(getOptMode()+" OPT: Optimize w/ max_mem="+toMB(_lm)+"/"+toMB(_rm)+"/"+toMB(_rm2)+", max_k="+_lk+"/"+_rk+"/"+_rk2+")." );
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
		
		// rewrite 2: remove unnecessary compare matrix (before result partitioning)
		rewriteRemoveUnnecessaryCompareMatrix(pn, ec);
		
		// rewrite 3: rewrite result partitioning (incl. log/phy recompile LIX) 
		boolean flagLIX = rewriteSetResultPartitioning( pn, M1, ec.getVariables() );
		M1 = _cost.getEstimate(TestMeasure.MEMORY_USAGE, pn); //reestimate 
		M2 = _cost.getEstimate(TestMeasure.MEMORY_USAGE, pn, LopProperties.ExecType.CP);
		LOG.debug(getOptMode()+" OPT: estimated new mem (serial exec) M="+toMB(M1) );
		LOG.debug(getOptMode()+" OPT: estimated new mem (serial exec, all CP) M="+toMB(M2) );
		
		// rewrite 4: execution strategy
		boolean flagRecompMR = rewriteSetExecutionStategy( pn, M0, M1, M2, flagLIX );
		
		//exec-type-specific rewrites
		if( pn.getExecType() == ExecType.MR || pn.getExecType()==ExecType.SPARK )
		{
			if( flagRecompMR ){
				//rewrite 5: set operations exec type
				rewriteSetOperationsExecType( pn, flagRecompMR );
				M1 = _cost.getEstimate(TestMeasure.MEMORY_USAGE, pn); //reestimate 		
			}
			
			// rewrite 6: data colocation
			rewriteDataColocation( pn, ec.getVariables() );
			
			// rewrite 7: rewrite set partition replication factor
			rewriteSetPartitionReplicationFactor( pn, partitionedMatrices, ec.getVariables() );
			
			// rewrite 8: rewrite set partition replication factor
			rewriteSetExportReplicationFactor( pn, ec.getVariables() );
			
			// rewrite 9: nested parallelism (incl exec types)	
			boolean flagNested = rewriteNestedParallelism( pn, M1, flagLIX );
			
			// rewrite 10: determine parallelism
			rewriteSetDegreeOfParallelism( pn, M1, flagNested );
			
			// rewrite 11: task partitioning 
			rewriteSetTaskPartitioner( pn, flagNested, flagLIX );
			
			// rewrite 12: fused data partitioning and execution
			rewriteSetFusedDataPartitioningExecution(pn, M1, flagLIX, partitionedMatrices, ec.getVariables());
		
			// rewrite 13: transpose sparse vector operations
			rewriteSetTranposeSparseVectorOperations(pn, partitionedMatrices, ec.getVariables());
			
			// rewrite 14: set in-place result indexing
			HashSet<String> inplaceResultVars = new HashSet<String>();
			rewriteSetInPlaceResultIndexing(pn, M1, ec.getVariables(), inplaceResultVars);
			
			// rewrite 15: disable caching
			rewriteDisableCPCaching(pn, inplaceResultVars, ec.getVariables());
		}
		else //if( pn.getExecType() == ExecType.CP )
		{
			// rewrite 10: determine parallelism
			rewriteSetDegreeOfParallelism( pn, M1, false );
			
			// rewrite 11: task partitioning
			rewriteSetTaskPartitioner( pn, false, false ); //flagLIX always false 
			
			// rewrite 14: set in-place result indexing
			HashSet<String> inplaceResultVars = new HashSet<String>();
			rewriteSetInPlaceResultIndexing(pn, M1, ec.getVariables(), inplaceResultVars);
			
			if( !OptimizerUtils.isSparkExecutionMode() ) {
				// rewrite 16: runtime piggybacking
				rewriteEnableRuntimePiggybacking( pn, ec.getVariables(), partitionedMatrices );
			}
			else {
				//rewrite 17: checkpoint injection for parfor loop body
				rewriteInjectSparkLoopCheckpointing( pn );
				
				//rewrite 18: repartition read-only inputs for zipmm 
				rewriteInjectSparkRepartition( pn, ec.getVariables() );
				
				//rewrite 19: eager caching for checkpoint rdds
				rewriteSetSparkEagerRDDCaching( pn, ec.getVariables() );
			}
		}	
	
		// rewrite 20: set result merge
		rewriteSetResultMerge( pn, ec.getVariables(), true );
		
		// rewrite 21: set local recompile memory budget
		rewriteSetRecompileMemoryBudget( pn );
		
		///////
		//Final rewrites for cleanup / minor improvements
		
		// rewrite 22: parfor (in recursive functions) to for
		rewriteRemoveRecursiveParFor( pn, ec.getVariables() );
		
		// rewrite 23: parfor (par=1) to for 
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
			_rk2 = (int) Math.max( _rk2, tmprk/2 );
		}
		
		//correction of max parallelism and memory if spark runtime enabled because
		//spark limits the available parallelism by its own executor configuration
		if( OptimizerUtils.isSparkExecutionMode() ) {
			_rk = (int) SparkExecutionContext.getDefaultParallelism(true);
			_rk2 = _rk; //equal map/reduce unless we find counter-examples 
			_rkmax   = (int) Math.ceil( PAR_K_FACTOR * _rk ); 
			_rkmax2  = (int) Math.ceil( PAR_K_FACTOR * _rk2 ); 
			int cores = SparkExecutionContext.getDefaultParallelism(true)
					/ SparkExecutionContext.getNumExecutors();
			int ccores = (int) Math.min(cores, _N);
			_rm = SparkExecutionContext.getBroadcastMemoryBudget() / ccores;
			_rm2 = SparkExecutionContext.getBroadcastMemoryBudget() / ccores;
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
		if(    OptimizerUtils.isHybridExecutionMode()  //only if we are allowed to recompile
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
		
		PDataPartitioner REMOTE = OptimizerUtils.isSparkExecutionMode() ? 
				PDataPartitioner.REMOTE_SPARK : PDataPartitioner.REMOTE_MR;
		PDataPartitioner pdp = (apply)? REMOTE : PDataPartitioner.NONE;		
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
			String inMatrix = h.getInput().get(0).getName();
			if( cand.containsKey(inMatrix) ) //Required Condition: partitioning applicable
			{
				PDataPartitionFormat dpf = cand.get(inMatrix);
				double mnew = getNewRIXMemoryEstimate( n, inMatrix, dpf, vars );
				//NOTE: for the moment, we do not partition according to the remote mem, because we can execute 
				//it even without partitioning in CP. However, advanced optimizers should reason about this 					   
				//double mold = h.getMemEstimate();
				if(	   n.getExecType() == ExecType.MR ||  n.getExecType()==ExecType.SPARK ) //Opt Condition: MR/Spark
				   // || (mold > _rm && mnew <= _rm)   ) //Opt Condition: non-MR special cases (for remote exec)
				{
					//NOTE: subsequent rewrites will still use the MR mem estimate
					//(guarded by subsequent operations that have at least the memory req of one partition)
					//if( mnew < _lm ) //apply rewrite if partitions fit into memory
					//	n.setExecType(ExecType.CP);
					//else
					//	n.setExecType(ExecType.CP); //CP_FILE, but hop still in MR 
					n.setExecType(ExecType.CP);
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
					mem = OptimizerUtils.estimateSize(mo.getNumRows(), 1); 
					break;
				case ROW_WISE:
					mem = OptimizerUtils.estimateSize(1, mo.getNumColumns());
					break;
				case BLOCK_WISE_M_N:
					mem = Integer.MAX_VALUE; //TODO
					break;
					
				default:
					//do nothing
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
				
			default:
				//do nothing	
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
			            && !cand.isEmpty() //at least one MR
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
			if( !resultVars.contains(base.getName()) )
				ret = false;
		}

		//check access pattern, memory budget
		if( ret ) {
			int dpf = 0;
			Hop inpRowL = h.getInput().get(2);
			Hop inpRowU = h.getInput().get(3);
			Hop inpColL = h.getInput().get(4);
			Hop inpColU = h.getInput().get(5);
			if( (inpRowL.getName().equals(iterVarname) && inpRowU.getName().equals(iterVarname)) )
				dpf = 1; //rowwise
			if( (inpColL.getName().equals(iterVarname) && inpColU.getName().equals(iterVarname)) )
				dpf = (dpf==0) ? 2 : 3; //colwise or cellwise
			
			if( dpf == 0 )
				ret = false;
			else
			{
				//check memory budget
				MatrixObject mo = (MatrixObject)vars.get(base.getName());
				if( mo.getNnz() != 0 ) //-1 valid because result var known during opt
					ret = false;
		
				//Note: for memory estimation the common case is sparse since remote_mr and individual tasks;
				//and in the dense case, we would not benefit from result partitioning
				boolean sparse = MatrixBlock.evalSparseFormatInMemory(base.getDim1(), base.getDim2(),base.getDim1());
				
				if( sparse ) 
				{
					//custom memory estimatation in order to account for structural properties
					//e.g., for rowwise we know that we only pay one sparserow overhead per task
					double memSparseBlock = estimateSizeSparseRowBlock(base.getDim1());
					double memSparseRow1 = estimateSizeSparseRow(base.getDim2(), base.getDim2());
					double memSparseRowMin = estimateSizeSparseRowMin(base.getDim2());
					
					double memTask1 = -1;
					int taskN = -1;
					switch(dpf) { 
						case 1: //rowwise
							//sparse block and one sparse row per task
							memTask1 = memSparseBlock + memSparseRow1;
							taskN = (int) ((_rm-memSparseBlock) / memSparseRow1); 
							break;
						case 2: //colwise
							//sparse block, sparse row per row but shared over tasks
							memTask1 = memSparseBlock + memSparseRowMin * base.getDim1();
							taskN = estimateNumTasksSparseCol(_rm-memSparseBlock, base.getDim1());
							break;
						case 3: //cellwise
							//sparse block and one minimal sparse row per task
							memTask1 = memSparseBlock + memSparseRowMin;
							taskN = (int) ((_rm-memSparseBlock) / memSparseRowMin); 
							break;	
					}

					if( memTask1>_rm || memTask1<0 )
						ret = false;
					else
						n.addParam(ParamType.TASK_SIZE, String.valueOf(taskN));
				}
				else 
				{ 
					//dense (no result partitioning possible)
					ret = false;
				}
			}
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @param rows
	 * @return
	 */
	private double estimateSizeSparseRowBlock( long rows ) {
		//see MatrixBlock.estimateSizeSparseInMemory
		return 44 + rows * 8;
	}
	
	/**
	 * 
	 * @param cols
	 * @param nnz
	 * @return
	 */
	private double estimateSizeSparseRow( long cols, long nnz ) {
		//see MatrixBlock.estimateSizeSparseInMemory
		long cnnz = Math.max(SparseRow.initialCapacity, Math.max(cols, nnz));
		return ( 116 + 12 * cnnz ); //sparse row
	}
	
	/**
	 * 
	 * @param cols
	 * @return
	 */
	private double estimateSizeSparseRowMin( long cols ) {
		//see MatrixBlock.estimateSizeSparseInMemory
		long cnnz = Math.min(SparseRow.initialCapacity, cols);
		return ( 116 + 12 * cnnz ); //sparse row
	}
	
	/**
	 * 
	 * @param budget
	 * @param rows
	 * @return
	 */
	private int estimateNumTasksSparseCol( double budget, long rows ) {
		//see MatrixBlock.estimateSizeSparseInMemory
		double lbudget = budget - rows * 116;
		return (int) Math.floor( lbudget / 12 );
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
		ArrayList<Instruction> newInst = Recompiler.recompileHopsDag(sb, sb.get_hops(), vars, null, false, 0);
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


		String datapartitioner = n.getParam(ParamType.DATA_PARTITIONER);
		ExecType REMOTE = OptimizerUtils.isSparkExecutionMode() ? ExecType.SPARK : ExecType.MR;
		PDataPartitioner REMOTE_DP = OptimizerUtils.isSparkExecutionMode() ? PDataPartitioner.REMOTE_SPARK : PDataPartitioner.REMOTE_MR;

		//deciding on the execution strategy
		if(    (isCPOnly && M <= _rm )   //Required: all instruction can be be executed in CP
			|| (isCPOnlyPossible && M2 <= _rm) )  //Required: cp inst fit into remote JVM mem 
		{
			//at this point all required conditions for REMOTE_MR given, now its an opt decision
			int cpk = (int) Math.min( _lk, Math.floor( _lm / M ) ); //estimated local exploited par  
			
			//MR if local par cannot be exploited due to mem constraints (this implies that we work on large data)
			//(the factor of 2 is to account for hyper-threading and in order prevent too eager remote parfor)
			if( 2*cpk < _lk && 2*cpk < _N && 2*cpk < _rk )
			{
				n.setExecType( REMOTE ); //remote parfor
			}
			//MR if problem is large enough and remote parallelism is larger than local   
			else if( _lk < _N && _lk < _rk && isLargeProblem(n, M0) )
			{
				n.setExecType( REMOTE ); //remote parfor
			}
			//MR if MR operations in local, but CP only in remote (less overall MR jobs)
			else if( (!isCPOnly) && isCPOnlyPossible )
			{
				n.setExecType( REMOTE ); //remote parfor
			}
			//MR if necessary for LIX rewrite (LIX true iff cp only and rm valid)
			else if( flagLIX ) 
			{
				n.setExecType( REMOTE );  //remote parfor
			}
			//MR if remote data partitioning, because data will be distributed on all nodes 
			else if( datapartitioner!=null && datapartitioner.equals(REMOTE_DP.toString())
					 && !InfrastructureAnalyzer.isLocalMode())
			{
				n.setExecType( REMOTE );  //remote parfor
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
		
		PExecMode mode = n.getExecType().toParForExecMode();
		pfpb.setExecMode( mode );	
		
		//decide if recompilation according to remote mem budget necessary
		boolean requiresRecompile = ((mode == PExecMode.REMOTE_MR || mode == PExecMode.REMOTE_SPARK) && !isCPOnly );
		
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
				&& M0 > PROB_SIZE_THRESHOLD_MB ); //original operations at least larger than 256MB
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
		
		if( n.isLeaf() && (et == ExecType.MR || et == ExecType.SPARK) )
		{
			Hop h = OptTreeConverter.getAbstractPlanMapping().getMappedHop( n.getID() );
			if(    h.getForcedExecType()!=LopProperties.ExecType.MR  //e.g., -exec=hadoop
				&& h.getForcedExecType()!=LopProperties.ExecType.SPARK) 
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
			String inMatrix = h.getInput().get(0).getName();
			String indexAccess = null;
			switch( dpf )
			{
				case ROW_WISE: //input 1 and 2 eq
					if( h.getInput().get(1) instanceof DataOp )
						indexAccess = h.getInput().get(1).getName();
					break;
				case COLUMN_WISE: //input 3 and 4 eq
					if( h.getInput().get(3) instanceof DataOp )
						indexAccess = h.getInput().get(3).getName();
					break;
				default:
					//do nothing
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
		if( n.getExecType()==ExecType.MR || n.getExecType()==ExecType.SPARK )		
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
	 * @throws DMLRuntimeException 
	 */
	protected void rewriteSetDegreeOfParallelism(OptNode n, double M, boolean flagNested) 
		throws DMLRuntimeException 
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
			
			//ensure local memory constraint (for spark more conservative in order to 
			//prevent unnecessary guarded collect)
			double mem = OptimizerUtils.isSparkExecutionMode() ? _lm/2 : _lm;
			kMax = Math.min( kMax, (int)Math.floor( mem / M ) );
			kMax = Math.max( kMax, 1);
			
			//constrain max parfor parallelism by problem size
			int parforK = (int)((_N<kMax)? _N : kMax);
			
			//set parfor degree of parallelism
			pfpb.setDegreeOfParallelism(parforK);
			n.setK(parforK);	
			
			//distribute remaining parallelism 
			int remainParforK = (int)Math.ceil(((double)(kMax-parforK+1))/parforK);
			int remainOpsK = Math.max(_lkmaxCP / parforK, 1);
			rAssignRemainingParallelism( n, remainParforK, remainOpsK ); 
		}
		else // ExecType.MR/ExecType.SPARK
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
			
			//ensure remote memory constraint
			kMax = Math.min( kMax, (int)Math.floor( _rm / M ) ); //guaranteed >= 1 (see exec strategy)
			if( kMax < 1 )
				kMax = 1;
			
			//disable nested parallelism, if required
			if( !ALLOW_REMOTE_NESTED_PARALLELISM )
				kMax = 1;
					
			//distribute remaining parallelism and recompile parallel instructions
			rAssignRemainingParallelism( n, kMax, 1 ); 
		}		
		
		_numEvaluatedPlans++;
		LOG.debug(getOptMode()+" OPT: rewrite 'set degree of parallelism' - result=(see EXPLAIN)" );
	}
	
	/**
	 * 
	 * @param n
	 * @param par
	 * @throws DMLRuntimeException 
	 */
	protected void rAssignRemainingParallelism(OptNode n, int parforK, int opsK) 
		throws DMLRuntimeException
	{		
		ArrayList<OptNode> childs = n.getChilds();
		if( childs != null ) 
		{
			boolean recompileSB = false;
			for( OptNode c : childs )
			{
				//NOTE: we cannot shortcut with c.setSerialParFor() on par=1 because
				//this would miss to recompile multi-threaded hop operations
				
				if( c.getNodeType() == NodeType.PARFOR )
				{
					//constrain max parfor parallelism by problem size
					int tmpN = Integer.parseInt(c.getParam(ParamType.NUM_ITERATIONS));
					int tmpK = (tmpN<parforK)? tmpN : parforK;
					
					//set parfor degree of parallelism
					long id = c.getID();
					c.setK(tmpK);
					ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
                                                  .getAbstractPlanMapping().getMappedProg(id)[1];
					pfpb.setDegreeOfParallelism(tmpK);
					
					//distribute remaining parallelism 
					int remainParforK = (int)Math.ceil(((double)(parforK-tmpK+1))/tmpK);
					int remainOpsK = Math.max(opsK / tmpK, 1);
					rAssignRemainingParallelism(c, remainParforK, remainOpsK);
				}
				else if( c.getNodeType() == NodeType.HOP )
				{
					//set degree of parallelism for multi-threaded leaf nodes
					Hop h = OptTreeConverter.getAbstractPlanMapping().getMappedHop(c.getID());
					if(    OptimizerUtils.PARALLEL_CP_MATRIX_MULTIPLY 
						&& h instanceof MultiThreadedHop //abop, datagenop, qop, paramop
						&& !( h instanceof ParameterizedBuiltinOp //only paramop-grpagg
							 && ((ParameterizedBuiltinOp)h).getOp()!=ParamBuiltinOp.GROUPEDAGG) )
					{
						MultiThreadedHop mhop = (MultiThreadedHop) h;
						mhop.setMaxNumThreads(opsK); //set max constraint in hop
						c.setK(opsK); //set optnode k (for explain)
						//need to recompile SB, if changed constraint
						recompileSB = true;	
					}
				}
				else
					rAssignRemainingParallelism(c, parforK, opsK);
			}
			
			//recompile statement block if required
			if( recompileSB ) {
				try {
					//guaranteed to be a last-level block (see hop change)
					ProgramBlock pb = (ProgramBlock) OptTreeConverter.getAbstractPlanMapping().getMappedProg(n.getID())[1];
					Recompiler.recompileProgramBlockInstructions(pb);
				}
				catch(Exception ex){
					throw new DMLRuntimeException(ex);
				}
			}
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
		
		boolean jvmreuse = ConfigurationManager.getConfig().getBooleanValue(DMLConfig.JVM_REUSE); 
		
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
		else if( pn.getExecType()==ExecType.MR && !jvmreuse && pn.hasOnlySimpleChilds() )
		{
			//for simple body programs without loops, branches, or function calls, we don't
			//expect much load imbalance and hence use static partitioning in order to
			//(1) reduce task latency, (2) prevent repeated read (w/o jvm reuse), and (3)
			//preaggregate results (less write / less read by result merge)
			setTaskPartitioner( pn, PTaskPartitioner.STATIC );
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
		PDataPartitioner REMOTE_DP = OptimizerUtils.isSparkExecutionMode() ? PDataPartitioner.REMOTE_SPARK : PDataPartitioner.REMOTE_MR;
		PExecMode REMOTE_DPE = OptimizerUtils.isSparkExecutionMode() ? PExecMode.REMOTE_SPARK_DP : PExecMode.REMOTE_MR_DP;
		
		//precondition: rewrite only invoked if exec type MR 
		// (this also implies that the body is CP only)
		
		// try to merge MR data partitioning and MR exec 
		if( (pn.getExecType()==ExecType.MR || pn.getExecType()==ExecType.SPARK) //MR/SP EXEC and CP body
			&& M < _rm2 //fits into remote memory of reducers	
			&& partitioner!=null && partitioner.equals(REMOTE_DP.toString()) //MR/SP partitioning
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
				
				pn.addParam(ParamType.DATA_PARTITIONER, REMOTE_DPE.toString()+"(fused)");
				pn.setK( k );
				
				pfpb.setExecMode(REMOTE_DPE); //set fused exec type	
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
			String inMatrix = h.getInput().get(0).getName();
			String indexAccess = null;
			switch( dpf )
			{
				case ROW_WISE: //input 1 and 2 eq
					if( h.getInput().get(1) instanceof DataOp )
						indexAccess = h.getInput().get(1).getName();
					break;
				case COLUMN_WISE: //input 3 and 4 eq
					if( h.getInput().get(3) instanceof DataOp )
						indexAccess = h.getInput().get(3).getName();
					break;
					
				default:
					//do nothing
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
			
			String inMatrix = h.getInput().get(0).getName();
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
			LOG.warn(getOptMode()+" OPT: Set in-place result update is only applicable for a ParFor node.");
		
		boolean apply = false;

		ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
              .getAbstractPlanMapping().getMappedProg(pn.getID())[1];
		
		//note currently we decide for all result vars jointly, i.e.,
		//only if all fit pinned in remaining budget, we apply this rewrite.
		
		ArrayList<String> retVars = pfpb.getResultVariables();
		
		//compute total sum of pinned result variable memory
		double sum = computeTotalSizeResultVariables(retVars, vars, pfpb.getDegreeOfParallelism());
		
		//NOTE: currently this rule is too conservative (the result variable is assumed to be dense and
		//most importantly counted twice if this is part of the maximum operation)
		HashMap <String, ArrayList <UIPCandidateHop>> uipCandHopHM = new HashMap <String, ArrayList<UIPCandidateHop>>();   
		double totalMem = Math.max((M+sum), rComputeSumMemoryIntermediates(pn, new HashSet<String>(), uipCandHopHM));
		
		//optimization decision
		if( rHasOnlyInPlaceSafeLeftIndexing(pn, retVars) ) //basic correctness constraint
		{
			//result update in-place for MR/Spark (w/ remote memory constraint)
			if( (  pfpb.getExecMode() == PExecMode.REMOTE_MR_DP || pfpb.getExecMode() == PExecMode.REMOTE_MR
				|| pfpb.getExecMode() == PExecMode.REMOTE_SPARK_DP || pfpb.getExecMode() == PExecMode.REMOTE_SPARK) 
				&& totalMem < _rm )
			{ 
				apply = true;
			}
			//result update in-place for CP (w/ local memory constraint)
			else if(   pfpb.getExecMode() == PExecMode.LOCAL 
					&& totalMem * pfpb.getDegreeOfParallelism()  < _lm
					&& pn.isCPOnly() ) //no forced mr/spark execution  
			{ 
				apply = true;
			}
		}
		
		if(APPLY_REWRITE_UPDATE_INPLACE_INTERMEDIATE && LOG.isDebugEnabled())
			listUIPRes.remove();

		//modify result variable meta data, if rewrite applied
		if( apply ) 
		{
			//add result vars to result and set state
			//will be serialized and transfered via symbol table 
			for( String var : retVars ){
				Data dat = vars.get(var);
				if( dat instanceof MatrixObject )
					((MatrixObject)dat).enableUpdateInPlace(true);
			}
			inPlaceResultVars.addAll(retVars);

			if(APPLY_REWRITE_UPDATE_INPLACE_INTERMEDIATE)
			{
				isUpdateInPlaceApplicable(pn, uipCandHopHM);
	
				boolean bAnyUIPApplicable = false;
				for(Entry<String, ArrayList <UIPCandidateHop>> entry: uipCandHopHM.entrySet())
				{
					ArrayList <UIPCandidateHop> uipCandHopList = entry.getValue();
					
					if (uipCandHopList != null) {
						for (UIPCandidateHop uipCandHop: uipCandHopList)
							if(uipCandHop.isLoopApplicable() && uipCandHop.isUpdateInPlace())
							{
								uipCandHop.getHop().setUpdateInPlace(true);
								bAnyUIPApplicable = true;
								
								if(LOG.isDebugEnabled())
									listUIPRes.get().add(uipCandHop.getHop().getName());
							}
					}
				}
				if(bAnyUIPApplicable)
					try {
						//Recompile this block if there is any update in place applicable.
						Recompiler.recompileProgramBlockInstructions(pfpb);		//TODO: Recompile @ very high level ok?
					}
					catch(Exception ex){
						throw new DMLRuntimeException(ex);
					}
			}
		}

		if(APPLY_REWRITE_UPDATE_INPLACE_INTERMEDIATE && LOG.isTraceEnabled())
		{
			LOG.trace("UpdateInPlace = " + apply + " for lines between " + pn.getBeginLine() + " and " + pn.getEndLine());
			for(Entry<String, ArrayList <UIPCandidateHop>> entry: uipCandHopHM.entrySet())
			{
				ArrayList <UIPCandidateHop> uipCandHopList = entry.getValue();
				
				if (uipCandHopList != null) {
					for (UIPCandidateHop uipCandHop: uipCandHopList)
					{
						LOG.trace("Matrix Object: Name: " + uipCandHop.getHop().getName() + "<" + uipCandHop.getHop().getBeginLine() + "," + uipCandHop.getHop().getEndLine()+ ">, InLoop:"
								+ uipCandHop.isLoopApplicable() + ", UIPApplicable:" + uipCandHop.isUpdateInPlace() + ", HopUIPApplicable:" + uipCandHop.getHop().getUpdateInPlace());	
					}
				}
			}
		}
							
		LOG.debug(getOptMode()+" OPT: rewrite 'set in-place result indexing' - result="+
		          apply+" ("+ProgramConverter.serializeStringCollection(inPlaceResultVars)+", M="+toMB(totalMem)+")" );	
	}
	
	/* 
	 * Algorithm: isUpdateInPlaceApplicable()
	 *
	 * Purpose of this algorithm to identify intermediate hops containing matrix objects to be marked as "UpdateInPlace" from ParforProgramBlock. 
	 * First, list of candidates are identified. Then list is pruned based on conditions descibed below.
	 * 
	 * A.Identification of candidates:
	 *  1. Candidate's identity defined with name, beginline, endline, and hop.
	 * 	2. Operation of type LeftIndexingOp
	 *  3. Number of consumers for Hop's first input should be one.
	 *  4. Matrix Object on which leftindexing operation done has defined outside "Loop" A. 
	 *  5. LeftIndexingOp operation is within a "Loop" A.
	 *
	 * 	Notes: 1. Loop is of type while, for, or parfor with parallelism of one.
	 * 		   2. Some conidtions ignored at this point listed below
	 * 			2.1 Unsure of general instructions. It will be hard to identify and iterate.
	 * 			2.2 LeftIndexing outside "loop"
	 * 
	 * 
	 * B.Pruning the list:
	 *  Candidates are pruned based on any condition met from conditions from 1 to 3 below.
	 *  0. Identify of candidate is defined with name, begineline, endline, and hop.
	 * 	1. Based on the scope of candidate. If Variable (name) is defined in liveout of loop's statementblock.
	 *  2. Based on operation type and order  
	 *  	2.1 If hop's input variable name is same as candidate's name 
	 *  	2.2 Location of hop is before candidate.
	 *  	2.3 Hop's operator type is any of following 
	 *  		2.3.1 DataOp (with operation type TransientWrite or TransientRead)
	 *  		2.3.2 ReorgOp (with operation type Reshape or Transpose)
	 *  		2.3.3 FunctionOp
	 *  3. Location of consumer being affected.
	 *  	3.1 Consumer defined before leftindexing through operation process defined in 2.3 above.
	 *  	3.2 Consumer is being utilized after leftindexing on candidate.
	 *  
	 *  Notes:
	 *  	1. No interleave operations.
	 *  	2. Function with actual operation to be scanned for candiate exclusion list.
	 *  	3. Operattion that does not include updated data through updateinplace. 
	 * 	
	 * 
	 * @param pn:				OpNode of parfor loop
	 * @param uipCandHopHM:		Hashmap of UIPCandidateHop with name as a key.		
	 * @throws DMLRuntimeException
	 */
	private void isUpdateInPlaceApplicable(OptNode pn, HashMap <String, ArrayList <UIPCandidateHop>> uipCandHopHM)
			throws DMLRuntimeException 
	{
		rIsInLoop(pn, uipCandHopHM, false);
		
		// Prune candidate list based on non-existance of candidate in the loop
		Iterator<Map.Entry<String, ArrayList <UIPCandidateHop>>> uipCandHopHMIter = uipCandHopHM.entrySet().iterator();
		while(uipCandHopHMIter.hasNext())
		{
			Map.Entry<String, ArrayList <UIPCandidateHop>> uipCandHopHMentry = uipCandHopHMIter.next();
			ArrayList <UIPCandidateHop> uipCandHopList = uipCandHopHMentry.getValue();
			
			if (uipCandHopList != null) {
				for (Iterator<UIPCandidateHop> uipCandHopListIter = uipCandHopList.iterator(); uipCandHopListIter.hasNext();) 
				{
					UIPCandidateHop uipCandHop = uipCandHopListIter.next();
					if (!uipCandHop.isLoopApplicable())	//If Loop is not applicable then remove it from the list. 
					{
						uipCandHopListIter.remove();
						if(LOG.isTraceEnabled())
							LOG.trace("Matrix Object: Name: " + uipCandHop.getHop().getName() + "<" + uipCandHop.getHop().getBeginLine() + "," + uipCandHop.getHop().getEndLine()+ 
									">, removed from the candidate list as it does not have loop criteria applicable.");
					}
				}
				if(uipCandHopList.isEmpty())
					uipCandHopHMIter.remove();
			}
		}
		
		if(!uipCandHopHM.isEmpty())
		{
			// Get consumer list
			rResetVisitStatus(pn);
			rGetUIPConsumerList(pn, uipCandHopHM);
			
			
			// Prune candidate list if consumer is in function call.
			uipCandHopHMIter = uipCandHopHM.entrySet().iterator();
			while(uipCandHopHMIter.hasNext())
			{
				Map.Entry<String, ArrayList <UIPCandidateHop>> uipCandHopHMentry = uipCandHopHMIter.next();
				ArrayList <UIPCandidateHop> uipCandHopList = uipCandHopHMentry.getValue();
				
				if (uipCandHopList != null) {
					for (Iterator<UIPCandidateHop> uipCandHopListIter = uipCandHopList.iterator(); uipCandHopListIter.hasNext();) 
					{
						UIPCandidateHop uipCandHop = uipCandHopListIter.next();
						// if one of the consumer is FunctionOp then remove it.
						ArrayList<Hop> consHops = uipCandHop.getConsumerHops();
						if(consHops != null)
							for (Hop hop: consHops)
							{
								if(hop instanceof FunctionOp)
								{
									uipCandHopListIter.remove();
									if(LOG.isTraceEnabled())
										LOG.trace("Matrix Object: Name: " + uipCandHop.getHop().getName() + "<" + uipCandHop.getHop().getBeginLine() + "," + uipCandHop.getHop().getEndLine()+ 
												">, removed from the candidate list as one of the consumer is FunctionOp.");
									break;
								}
							}
					}
					if(uipCandHopList.isEmpty())
						uipCandHopHMIter.remove();
				}
			}

			//Validate the consumer list
			rResetVisitStatus(pn);
			rValidateUIPConsumerList(pn, uipCandHopHM);
		}
	}
	
	
	
	/* 	
	 * This will check if candidate LeftIndexingOp are in loop (while, for or parfor).
	 * 
	 * @param pn:				OpNode of parfor loop
	 * @param uipCandHopHM:		Hashmap of UIPCandidateHop with name as a key.		
	 * @throws DMLRuntimeException
	 */
	private void rIsInLoop(OptNode pn, HashMap <String, ArrayList<UIPCandidateHop>> uipCandHopHM, boolean bInLoop)
			throws DMLRuntimeException 
	{
		if(!pn.isLeaf())  
		{
			ProgramBlock pb = (ProgramBlock) OptTreeConverter.getAbstractPlanMapping().getMappedProg(pn.getID())[1];

			VariableSet varUpdated = pb.getStatementBlock().variablesUpdated();
			boolean bUIPCandHopUpdated = false;
			for(Entry<String, ArrayList <UIPCandidateHop>> entry: uipCandHopHM.entrySet())
			{
				String uipCandHopID = entry.getKey();
				
				if (varUpdated.containsVariable(uipCandHopID))
				{	
					bUIPCandHopUpdated = true;
					break;
				}
			}

			// As none of the UIP candidates updated in this DAG, no need for further processing within this DAG
			if(!bUIPCandHopUpdated)
				return;

			boolean bLoop = false;
			if(	bInLoop || pb instanceof WhileProgramBlock || 
				(pb instanceof ParForProgramBlock && ((ParForProgramBlock)pb).getDegreeOfParallelism() == 1) ||
				(pb instanceof ForProgramBlock && !(pb instanceof ParForProgramBlock)))
				bLoop = true;

			for (OptNode optNode: pn.getChilds())
			{
				rIsInLoop(optNode, uipCandHopHM, bLoop);
			}
		}
		else if(bInLoop)
		{
			Hop hop = (Hop) OptTreeConverter.getAbstractPlanMapping().getMappedHop(pn.getID());

			for(Entry<String, ArrayList <UIPCandidateHop>> entry: uipCandHopHM.entrySet())
			{
				ArrayList <UIPCandidateHop> uipCandHopList = entry.getValue();
				
				if (uipCandHopList != null) 
				{
					for (UIPCandidateHop uipCandHop: uipCandHopList)
					{
						//Update if candiate hop defined outside this loop, and leftindexing is within this loop.
						if (uipCandHop.getLocation() <= hop.getBeginLine() && uipCandHop.getHop().getBeginLine() <= hop.getEndLine())
							uipCandHop.setIsLoopApplicable(true);
					}
				}
			}
		}
		
	}
	
	
	
	/* 	
	 * This will get consumer list for candidate LeftIndexingOp.
	 * 
	 * @param pn:				OpNode of parfor loop
	 * @param uipCandHopHM:		Hashmap of UIPCandidateHop with name as a key.		
	 * @throws DMLRuntimeException
	 */
	private void rGetUIPConsumerList(OptNode pn, HashMap <String, ArrayList<UIPCandidateHop>> uipCandHopHM)
			throws DMLRuntimeException 
	{
		if(!pn.isLeaf())
		{
			if(pn.getNodeType() == OptNode.NodeType.FUNCCALL)
				return;

			ProgramBlock pb = (ProgramBlock) OptTreeConverter.getAbstractPlanMapping().getMappedProg(pn.getID())[1];
			
			VariableSet varRead = pb.getStatementBlock().variablesRead();
			boolean bUIPCandHopRead = false;
			for(Entry<String, ArrayList <UIPCandidateHop>> entry: uipCandHopHM.entrySet())
			{
				String uipCandHopID = entry.getKey();
				
				if (varRead.containsVariable(uipCandHopID))
				{	
					bUIPCandHopRead = true;
					break;
				}
			}
			
			// As none of the UIP candidates updated in this DAG, no need for further processing within this DAG
			if(!bUIPCandHopRead)
				return;
			
			for (OptNode optNode: pn.getChilds())
				rGetUIPConsumerList(optNode, uipCandHopHM);
		}
		else
		{
			OptTreePlanMappingAbstract map = OptTreeConverter.getAbstractPlanMapping();
			long ppid = map.getMappedParentID(map.getMappedParentID(pn.getID()));
			Object[] o = map.getMappedProg(ppid);
			ProgramBlock pb = (ProgramBlock) o[1];
			
			Hop hop = (Hop) OptTreeConverter.getAbstractPlanMapping().getMappedHop(pn.getID());
			rGetUIPConsumerList(hop, uipCandHopHM);

			if(pb instanceof IfProgramBlock || pb instanceof WhileProgramBlock || 
				(pb instanceof ForProgramBlock && !(pb instanceof ParForProgramBlock)))  //TODO
				rGetUIPConsumerList(pb, uipCandHopHM);
		} 
	}
	
	
	private void rGetUIPConsumerList(ProgramBlock pb, HashMap <String, ArrayList<UIPCandidateHop>> uipCandHopHM)
			throws DMLRuntimeException
	{
		ArrayList<ProgramBlock> childBlocks = null;
		if (pb instanceof WhileProgramBlock)
			childBlocks = ((WhileProgramBlock)pb).getChildBlocks();
		else if (pb instanceof ForProgramBlock)
			childBlocks = ((ForProgramBlock)pb).getChildBlocks();
		else if (pb instanceof IfProgramBlock) 
		{
			childBlocks = ((IfProgramBlock)pb).getChildBlocksIfBody();
			ArrayList<ProgramBlock> elseBlocks = ((IfProgramBlock)pb).getChildBlocksElseBody();
			if(childBlocks != null && elseBlocks != null)
				childBlocks.addAll(elseBlocks);
			else if (childBlocks == null)
				childBlocks = elseBlocks;
		}

		if(childBlocks != null)
			for (ProgramBlock childBlock: childBlocks)
			{
				rGetUIPConsumerList(childBlock, uipCandHopHM);
				try 
				{
					rGetUIPConsumerList(childBlock.getStatementBlock().get_hops(), uipCandHopHM);
				}
				catch (Exception e) {
					throw new DMLRuntimeException(e);
				}
			}
	}	
		
	private void rGetUIPConsumerList(ArrayList<Hop> hops, HashMap <String, ArrayList<UIPCandidateHop>> uipCandHopHM)
			throws DMLRuntimeException
	{
		if(hops != null)
			for (Hop hop: hops)
				rGetUIPConsumerList(hop, uipCandHopHM);
	}

		
	private void rGetUIPConsumerList(Hop hop, HashMap <String, ArrayList<UIPCandidateHop>> uipCandHopHM)
		throws DMLRuntimeException
	{
		if(hop.getVisited() != Hop.VisitStatus.DONE)
		{
			if ((!(!hop.getParent().isEmpty() && hop.getParent().get(0) instanceof LeftIndexingOp)) &&
				   ((hop instanceof DataOp && ((DataOp)hop).getDataOpType() == DataOpTypes.TRANSIENTREAD ) ||
					(hop instanceof ReorgOp && (((ReorgOp)hop).getOp() == ReOrgOp.RESHAPE || ((ReorgOp)hop).getOp() == ReOrgOp.TRANSPOSE)) ||
					(hop instanceof FunctionOp)))
			{	
				// If candidate's name is same as input hop.
				String uipCandiateID = hop.getName();
				ArrayList <UIPCandidateHop> uipCandHopList = uipCandHopHM.get(uipCandiateID);
				
				if (uipCandHopList != null) 
				{
					for (UIPCandidateHop uipCandHop: uipCandHopList)
					{
						// Add consumers for candidate hop.
						ArrayList<Hop> consumerHops = uipCandHop.getConsumerHops();
						if(uipCandHop.getConsumerHops() == null)
							consumerHops = new ArrayList<Hop>();
						consumerHops.add(getRootHop(hop));
						uipCandHop.setConsumerHops(consumerHops);
					}
				}
			}
			
			for(Hop hopIn: hop.getInput())
			{
				rGetUIPConsumerList(hopIn, uipCandHopHM);
			}
			
			hop.setVisited(Hop.VisitStatus.DONE);
		}
	}
	

	private Hop getRootHop(Hop hop)
	{
		return (!hop.getParent().isEmpty())?getRootHop(hop.getParent().get(0)):hop;
	}
	
	
	private void rResetVisitStatus(OptNode pn)
		throws DMLRuntimeException
	{
		
		if(!pn.isLeaf())
		{
			if(pn.getNodeType() == OptNode.NodeType.FUNCCALL)
			{
				Hop hopFunc = (Hop) OptTreeConverter.getAbstractPlanMapping().getMappedHop(pn.getID());
				hopFunc.resetVisitStatus();
				return;
			}
			ProgramBlock pb = (ProgramBlock) OptTreeConverter.getAbstractPlanMapping().getMappedProg(pn.getID())[1];

			ArrayList<ProgramBlock> childBlocks = null;
			if (pb instanceof WhileProgramBlock)
				childBlocks = ((WhileProgramBlock)pb).getChildBlocks();
			else if (pb instanceof ForProgramBlock)
				childBlocks = ((ForProgramBlock)pb).getChildBlocks();
			else if (pb instanceof IfProgramBlock) {
				childBlocks = ((IfProgramBlock)pb).getChildBlocksIfBody();
				ArrayList<ProgramBlock> elseBlocks = ((IfProgramBlock)pb).getChildBlocksElseBody();
				if(childBlocks != null && elseBlocks != null)
					childBlocks.addAll(elseBlocks);
				else if (childBlocks == null)
					childBlocks = elseBlocks;
			}
				
			if(childBlocks != null)
			{
				for (ProgramBlock childBlock: childBlocks)
				{
					try 
					{
						Hop.resetVisitStatus(childBlock.getStatementBlock().get_hops());
					}
					catch (Exception e)
					{
						throw new DMLRuntimeException(e);
					}
				}
			}
			
			for (OptNode optNode: pn.getChilds())
			{
				rResetVisitStatus(optNode);
			}
		}
		else
		{
			Hop hop = (Hop) OptTreeConverter.getAbstractPlanMapping().getMappedHop(pn.getID());
			if(hop != null)
			{
				hop.resetVisitStatus();
			}
		}
	}
	

	
	/* 	
	 * This will validate candidate's consumer list.
	 * 
	 * @param pn:				OpNode of parfor loop
	 * @param uipCandHopHM:		Hashmap of UIPCandidateHop with name as a key.		
	 * @throws DMLRuntimeException
	 */
	
	private void rValidateUIPConsumerList(OptNode pn, HashMap <String, ArrayList<UIPCandidateHop>> uipCandHopHM)
			throws DMLRuntimeException 
	{
		if(!pn.isLeaf())
		{
			if(pn.getNodeType() == OptNode.NodeType.FUNCCALL)
			{
				Hop hop = (Hop) OptTreeConverter.getAbstractPlanMapping().getMappedHop(pn.getID());
				rValidateUIPConsumerList(hop.getInput(), uipCandHopHM);
				return;
			}

			ProgramBlock pb = (ProgramBlock) OptTreeConverter.getAbstractPlanMapping().getMappedProg(pn.getID())[1];
			
			VariableSet varRead = pb.getStatementBlock().variablesRead();
			boolean bUIPCandHopRead = false;
			for(Entry<String, ArrayList <UIPCandidateHop>> entry: uipCandHopHM.entrySet())
			{
				ArrayList <UIPCandidateHop> uipCandHopList = entry.getValue();
				if (uipCandHopList != null) 
				{
					for (UIPCandidateHop uipCandHop: uipCandHopList)
					{
						ArrayList<Hop> consumerHops = uipCandHop.getConsumerHops();
						if(consumerHops != null)
						{
							// If any of consumer's input (or any parent in hierachy of input) matches candiate's name, then 
							// remove candidate from the list.
							for (Hop consumerHop: consumerHops)
							{
								if (varRead.containsVariable(consumerHop.getName()))
								{	
									bUIPCandHopRead = true;
									break;
								}
							}
						}
					}
				}
			}
			// As none of the UIP candidates updated in this DAG, no need for further processing within this DAG
			if(!bUIPCandHopRead)
				return;

			for (OptNode optNode: pn.getChilds())
					rValidateUIPConsumerList(optNode, uipCandHopHM);
		}
		else
		{
			OptTreePlanMappingAbstract map = OptTreeConverter.getAbstractPlanMapping();
			long ppid = map.getMappedParentID(map.getMappedParentID(pn.getID()));
			Object[] o = map.getMappedProg(ppid);
			ProgramBlock pb = (ProgramBlock) o[1];

			if(pb instanceof IfProgramBlock || pb instanceof WhileProgramBlock || 
				(pb instanceof ForProgramBlock && !(pb instanceof ParForProgramBlock)))	//TODO
				rValidateUIPConsumerList(pb, uipCandHopHM);

			long pid = map.getMappedParentID(pn.getID());
			o = map.getMappedProg(pid);
			pb = (ProgramBlock) o[1];
			Hop hop = map.getMappedHop(pn.getID());
			rValidateUIPConsumerList(hop, uipCandHopHM, pb.getStatementBlock().variablesRead());
		}
	}
	
	private void rValidateUIPConsumerList(ProgramBlock pb, HashMap <String, ArrayList<UIPCandidateHop>> uipCandHopHM)
			throws DMLRuntimeException
	{
		ArrayList<ProgramBlock> childBlocks = null;
		if (pb instanceof WhileProgramBlock)
			childBlocks = ((WhileProgramBlock)pb).getChildBlocks();
		else if (pb instanceof ForProgramBlock)
			childBlocks = ((ForProgramBlock)pb).getChildBlocks();
		else if (pb instanceof IfProgramBlock) 
		{
			childBlocks = ((IfProgramBlock)pb).getChildBlocksIfBody();
			ArrayList<ProgramBlock> elseBlocks = ((IfProgramBlock)pb).getChildBlocksElseBody();
			if(childBlocks != null && elseBlocks != null)
				childBlocks.addAll(elseBlocks);
			else if (childBlocks == null)
				childBlocks = elseBlocks;
		}

		if(childBlocks != null)
			for (ProgramBlock childBlock: childBlocks)
			{
				rValidateUIPConsumerList(childBlock, uipCandHopHM);
				try 
				{
					rValidateUIPConsumerList(childBlock.getStatementBlock(), uipCandHopHM);
				}
				catch (Exception e) {
					throw new DMLRuntimeException(e);
				}
			}
	}	
		

	private void rValidateUIPConsumerList(StatementBlock sb, HashMap <String, ArrayList<UIPCandidateHop>> uipCandHopHM)
			throws DMLRuntimeException 
	{
		VariableSet readVariables = sb.variablesRead();
		
		for(Entry<String, ArrayList <UIPCandidateHop>> entry: uipCandHopHM.entrySet())
		{
			ArrayList <UIPCandidateHop> uipCandHopList = entry.getValue();
			if (uipCandHopList != null) 
			{
				for (UIPCandidateHop uipCandHop: uipCandHopList)
				{
					ArrayList<Hop> consumerHops = uipCandHop.getConsumerHops();
					if(consumerHops != null)
					{
						// If consumer has read then remove candidate from the list (set flag to false).
						for (Hop consumerHop: consumerHops)
							if(readVariables.containsVariable(consumerHop.getName()))
							{
								uipCandHop.setUpdateInPlace(false);
								break;
							}
					}
				}
			}
		}
	}
	

	private void rValidateUIPConsumerList(ArrayList<Hop> hops, HashMap <String, ArrayList<UIPCandidateHop>> uipCandHopHM)
			throws DMLRuntimeException
	{
		if(hops != null)
			for (Hop hop: hops)
				rValidateUIPConsumerList(hop, uipCandHopHM);
	}


	private void rValidateUIPConsumerList(Hop hop, HashMap <String, ArrayList<UIPCandidateHop>> uipCandHopHM)
			throws DMLRuntimeException 
	{
		if(hop.getVisited() != Hop.VisitStatus.DONE)
		{
			for(Entry<String, ArrayList <UIPCandidateHop>> entry: uipCandHopHM.entrySet())
			{
				ArrayList <UIPCandidateHop> uipCandHopList = entry.getValue();
				if (uipCandHopList != null) 
				{
					for (UIPCandidateHop uipCandHop: uipCandHopList)
					{
						ArrayList<Hop> consumerHops = uipCandHop.getConsumerHops();
						if(consumerHops != null)
						{
							// If consumer has read then remove candidate from the list (set flag to false).
							for (Hop consumerHop: consumerHops)
								if(hop.getName().equals(consumerHop.getName()))
								{
									uipCandHop.setUpdateInPlace(false);
									break;
								}
						}
					}
				}
			}
			hop.setVisited(Hop.VisitStatus.DONE);
		}
	}

	private void rValidateUIPConsumerList(Hop hop, HashMap <String, ArrayList<UIPCandidateHop>> uipCandHopHM, VariableSet readVariables)
			throws DMLRuntimeException 
	{
		if(hop.getVisited() != Hop.VisitStatus.DONE)
		{
			for(Entry<String, ArrayList <UIPCandidateHop>> entry: uipCandHopHM.entrySet())
			{
				ArrayList <UIPCandidateHop> uipCandHopList = entry.getValue();
				if (uipCandHopList != null) 
				{
					for (UIPCandidateHop uipCandHop: uipCandHopList)
					{
						ArrayList<Hop> consumerHops = uipCandHop.getConsumerHops();
						if(consumerHops != null)
						{
							// If consumer has read then remove candidate from the list (set flag to false).
							for (Hop consumerHop: consumerHops)
								if(readVariables.containsVariable(consumerHop.getName()))
								{
									uipCandHop.setUpdateInPlace(false);
									break;
								}
						}
					}
				}
			}
			hop.setVisited(Hop.VisitStatus.DONE);
		}
	}
	
	
	public static List<String> getUIPList()
	{
		return listUIPRes.get();
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
			if( retVars.contains( h.getInput().get(0).getName() ) )
			{
				ret &= (h.getParent().size()==1 
						&& h.getParent().get(0).getName().equals(h.getInput().get(0).getName()));
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
	private double computeTotalSizeResultVariables(ArrayList<String> retVars, LocalVariableMap vars, int k)
	{
		double sum = 1;
		for( String var : retVars ){
			Data dat = vars.get(var);
			if( dat instanceof MatrixObject )
			{
				MatrixObject mo = (MatrixObject)dat;
				double nnz = mo.getNnz();

				if(nnz == 0.0) 
					sum += OptimizerUtils.estimateSizeExactSparsity(mo.getNumRows(), mo.getNumColumns(), 1.0);
				else {
					double sp = mo.getSparsity();
					sum += (k+1) * (OptimizerUtils.estimateSizeExactSparsity(mo.getNumRows(), mo.getNumColumns(),
							Math.min((1.0/k)+sp, 1.0)));	// Every worker will consume memory for (MatrixSize/k + nnz) data.
														// This is applicable only when there is non-zero nnz. 
				}
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
		
		double M_sumInterm = rComputeSumMemoryIntermediates(pn, inplaceResultVars, new HashMap <String, ArrayList <UIPCandidateHop>>());
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
	protected double rComputeSumMemoryIntermediates( OptNode n, HashSet<String> inplaceResultVars, 
													HashMap <String, ArrayList <UIPCandidateHop>> uipCandidateHM )	
		throws DMLRuntimeException
	{
		double sum = 0;
		
		if( !n.isLeaf() )
		{
			for( OptNode cn : n.getChilds() )
				sum += rComputeSumMemoryIntermediates( cn, inplaceResultVars, uipCandidateHM );
		}
		else if(    n.getNodeType()== NodeType.HOP )
		{
			Hop h = OptTreeConverter.getAbstractPlanMapping().getMappedHop(n.getID());
			if (h.getDataType() == Expression.DataType.MATRIX && h instanceof LeftIndexingOp &&
					h.getInput().get(0).getParent().size() == 1)
			{
				long pid =  OptTreeConverter.getAbstractPlanMapping().getMappedParentID(n.getID());
				ProgramBlock pb = (ProgramBlock) OptTreeConverter.getAbstractPlanMapping().getMappedProg(pid)[1];
				
				while(!(pb instanceof WhileProgramBlock || pb instanceof ForProgramBlock))
				{
					pid = OptTreeConverter.getAbstractPlanMapping().getMappedParentID(pid);
					pb = (ProgramBlock) OptTreeConverter.getAbstractPlanMapping().getMappedProg(pid)[1];
				}

				String uipCandiateID = new String(h.getName());
				ArrayList <UIPCandidateHop> uipCandiHopList = uipCandidateHM.get(uipCandiateID);
				if(uipCandiHopList == null)
					uipCandiHopList = new ArrayList<UIPCandidateHop>();
				uipCandiHopList.add(new UIPCandidateHop(h, pb));
				uipCandidateHM.put(uipCandiateID, uipCandiHopList);

				StatementBlock sb = (StatementBlock) OptTreeConverter.getAbstractPlanMapping().getMappedProg(OptTreeConverter.getAbstractPlanMapping().getMappedParentID(n.getID()))[0];
				if(LOG.isDebugEnabled())
					LOG.debug("Candidate Hop:" + h.getName() + "<" + h.getBeginLine() + "," + h.getEndLine() + ">,<" + 
						h.getBeginColumn() + "," + h.getEndColumn() + "> PB:" + "<" + pb.getBeginLine() + "," + pb.getEndLine() + ">,<" + 
						pb.getBeginColumn() + "," + pb.getEndColumn() + "> SB:" + "<" + sb.getBeginLine() + "," + sb.getEndLine() + ">,<" + 
						sb.getBeginColumn() + "," + sb.getEndColumn() + ">");
			}
			
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
							&& !inplaceResultVars.contains(cn.getName())) //except in-place result vars
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
				
				if(    ch instanceof DataOp && ch.getDataType() == DataType.MATRIX
					&& inputVars.contains(ch.getName()) )
					//&& !partitionedVars.contains(ch.getName()))
				{
					ret = true;
					sharedVars.add(ch.getName());
				}
				else if(    ch instanceof ReorgOp && ((ReorgOp)ch).getOp()==ReOrgOp.TRANSPOSE 
					&& ch.getInput().get(0) instanceof DataOp && ch.getInput().get(0).getDataType() == DataType.MATRIX
					&& inputVars.contains(ch.getInput().get(0).getName()) )
					//&& !partitionedVars.contains(ch.getInput().get(0).getName()))
				{
					ret = true;
					sharedVars.add(ch.getInput().get(0).getName());
				}
			}
		}

		return ret;
	}


	///////
	//REWRITE inject spark loop checkpointing
	///
	
	/**
	 * 
	 * @param n
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	protected void rewriteInjectSparkLoopCheckpointing(OptNode n) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		//get program blocks of root parfor
		Object[] progobj = OptTreeConverter.getAbstractPlanMapping().getMappedProg(n.getID());
		ParForStatementBlock pfsb = (ParForStatementBlock)progobj[0];
		ParForStatement fs = (ParForStatement) pfsb.getStatement(0);
		ParForProgramBlock pfpb = (ParForProgramBlock)progobj[1];
		
		boolean applied = false;
		
		try
		{
			//apply hop rewrite inject spark checkpoints (but without context awareness)
			RewriteInjectSparkLoopCheckpointing rewrite = new RewriteInjectSparkLoopCheckpointing(false);
			ProgramRewriter rewriter = new ProgramRewriter(rewrite);
			ProgramRewriteStatus state = new ProgramRewriteStatus();
			rewriter.rewriteStatementBlockHopDAGs( pfsb, state );
			fs.setBody(rewriter.rewriteStatementBlocks(fs.getBody(), state));
			
			//recompile if additional checkpoints introduced
			if( state.getInjectedCheckpoints() ) {
				pfpb.setChildBlocks(ProgramRecompiler.generatePartitialRuntimeProgram(pfpb.getProgram(), fs.getBody()));
				applied = true;
			}
		}
		catch(Exception ex) {
			throw new DMLRuntimeException(ex);
		}
			
		LOG.debug(getOptMode()+" OPT: rewrite 'inject spark loop checkpointing' - result="+applied );
	}
	
	///////
	//REWRITE inject spark repartition for zipmm
	///
	
	/**
	 * 
	 * @param n
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	protected void rewriteInjectSparkRepartition(OptNode n, LocalVariableMap vars) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		//get program blocks of root parfor
		Object[] progobj = OptTreeConverter.getAbstractPlanMapping().getMappedProg(n.getID());
		ParForStatementBlock pfsb = (ParForStatementBlock)progobj[0];
		ParForProgramBlock pfpb = (ParForProgramBlock)progobj[1];
		
		ArrayList<String> ret = new ArrayList<String>();
		
		if(    OptimizerUtils.isSparkExecutionMode() //spark exec mode
			&& n.getExecType() == ExecType.CP		 //local parfor 
			&& _N > 1                            )   //at least 2 iterations                             
		{
			//collect candidates from zipmm spark instructions
			HashSet<String> cand = new HashSet<String>();
			rCollectZipmmPartitioningCandidates(n, cand);
			
			//prune updated candidates
			HashSet<String> probe = new HashSet<String>(pfsb.getReadOnlyParentVars());				
			for( String var : cand )
				if( probe.contains( var ) )
					ret.add( var );
				
			//prune small candidates
			ArrayList<String> tmp = new ArrayList<String>(ret);
			ret.clear();
			for( String var : tmp )
				if( vars.get(var) instanceof MatrixObject )
				{
					MatrixObject mo = (MatrixObject) vars.get(var);
					double sp = OptimizerUtils.getSparsity(mo.getNumRows(), mo.getNumColumns(), mo.getNnz());
					double size = OptimizerUtils.estimateSizeExactSparsity(mo.getNumRows(), mo.getNumColumns(), sp);
					if( size > OptimizerUtils.getLocalMemBudget() )
						ret.add(var);
				}
			
			//apply rewrite to parfor pb
			if( !ret.isEmpty() ) {
				pfpb.setSparkRepartitionVariables(ret);
			}
		}
		
		_numEvaluatedPlans++;
		LOG.debug(getOptMode()+" OPT: rewrite 'inject spark input repartition' - result="+ret.size()+
				" ("+ProgramConverter.serializeStringCollection(ret)+")" );
	}
	
	/**
	 * 
	 * @param n
	 * @param cand
	 */
	private void rCollectZipmmPartitioningCandidates( OptNode n, HashSet<String> cand )
	{
		//collect zipmm inputs
		if( n.getNodeType()==NodeType.HOP ) 
		{
			Hop h = OptTreeConverter.getAbstractPlanMapping().getMappedHop(n.getID());
			if( h instanceof AggBinaryOp && (((AggBinaryOp)h).getMMultMethod()==MMultMethod.ZIPMM 
				||((AggBinaryOp)h).getMMultMethod()==MMultMethod.CPMM) )
			{
				//found zipmm or cpmm (unknowns) which might turn into zipmm
				//check for dataop or dataops under transpose on both sides
				for( Hop in : h.getInput() ) {
					if( in instanceof DataOp )
						cand.add( in.getName() );
					else if( in instanceof ReorgOp 
						&& ((ReorgOp)in).getOp()==ReOrgOp.TRANSPOSE
						&& in.getInput().get(0) instanceof DataOp )
						cand.add( in.getInput().get(0).getName() );
				}
			}
		}
		
		//recursively process childs
		if( !n.isLeaf() )
			for( OptNode c : n.getChilds() )
				rCollectZipmmPartitioningCandidates(c, cand);
	}
	
	///////
	//REWRITE set spark eager rdd caching
	///
	
	/**
	 * 
	 * @param n
	 * @throws DMLRuntimeException
	 * @throws DMLUnsupportedOperationException
	 */
	protected void rewriteSetSparkEagerRDDCaching(OptNode n, LocalVariableMap vars) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		//get program blocks of root parfor
		Object[] progobj = OptTreeConverter.getAbstractPlanMapping().getMappedProg(n.getID());
		ParForStatementBlock pfsb = (ParForStatementBlock)progobj[0];
		ParForProgramBlock pfpb = (ParForProgramBlock)progobj[1];
		
		ArrayList<String> ret = new ArrayList<String>();
		
		if(    OptimizerUtils.isSparkExecutionMode() //spark exec mode
			&& n.getExecType() == ExecType.CP		 //local parfor 
			&& _N > 1                            )   //at least 2 iterations                             
		{
			Set<String> cand = pfsb.variablesRead().getVariableNames();
			Collection<String> rpVars = pfpb.getSparkRepartitionVariables();
			for( String var : cand)
			{
				Data dat = vars.get(var);
				
				if( dat!=null && dat instanceof MatrixObject
					&& ((MatrixObject)dat).getRDDHandle()!=null )
				{
					MatrixObject mo = (MatrixObject)dat;
					MatrixCharacteristics mc = mo.getMatrixCharacteristics();
					RDDObject rdd = mo.getRDDHandle();
					if( (rpVars==null || !rpVars.contains(var)) //not a repartition var
						&& rdd.rHasCheckpointRDDChilds()        //is cached rdd 
						&& _lm / n.getK() <                     //is out-of-core dataset
						OptimizerUtils.estimateSizeExactSparsity(mc))
					{
						ret.add(var);
					}
				}
			}
			
			//apply rewrite to parfor pb
			if( !ret.isEmpty() ) {
				pfpb.setSparkEagerCacheVariables(ret);
			}
		}
		
		_numEvaluatedPlans++;
		LOG.debug(getOptMode()+" OPT: rewrite 'set spark eager rdd caching' - result="+ret.size()+
				" ("+ProgramConverter.serializeStringCollection(ret)+")" );
	}
	
	///////
	//REWRITE remove compare matrix (for result merge, needs to be invoked before setting result merge)
	///
	
	/**
	 *
	 * 
	 * @param n
	 * @throws DMLRuntimeException 
	 */
	protected void rewriteRemoveUnnecessaryCompareMatrix( OptNode n, ExecutionContext ec ) 
		throws DMLRuntimeException
	{
		ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
			    .getAbstractPlanMapping().getMappedProg(n.getID())[1];

		ArrayList<String> cleanedVars = new ArrayList<String>();
		ArrayList<String> resultVars = pfpb.getResultVariables();
		String itervar = pfpb.getIterablePredicateVars()[0]; 
		
		for( String rvar : resultVars ) {
			Data dat = ec.getVariable(rvar);
			if( dat instanceof MatrixObject && ((MatrixObject)dat).getNnz()!=0     //subject to result merge with compare
				&& n.hasOnlySimpleChilds()                                         //guaranteed no conditional indexing	
				&& rContainsResultFullReplace(n, rvar, itervar, (MatrixObject)dat) //guaranteed full matrix replace 
				//&& !pfsb.variablesRead().containsVariable(rvar)                  //never read variable in loop body
				&& !rIsReadInRightIndexing(n, rvar)                                //never read variable in loop body
				&& ((MatrixObject)dat).getNumRows()<=Integer.MAX_VALUE
				&& ((MatrixObject)dat).getNumColumns()<=Integer.MAX_VALUE )
			{
				//replace existing matrix object with empty matrix
				MatrixObject mo = (MatrixObject)dat;
				ec.cleanupMatrixObject(mo);
				ec.setMatrixOutput(rvar, new MatrixBlock((int)mo.getNumRows(), (int)mo.getNumColumns(),false));
				
				//keep track of cleaned result variables
				cleanedVars.add(rvar);
			}
		}

		_numEvaluatedPlans++;
		LOG.debug(getOptMode()+" OPT: rewrite 'remove unnecessary compare matrix' - result="+(!cleanedVars.isEmpty())+" ("+ProgramConverter.serializeStringCollection(cleanedVars)+")" );
	}
	

	/**
	 * 
	 * @param n
	 * @param resultVar
	 * @param iterVarname
	 * @param mo
	 * @return
	 * @throws DMLRuntimeException
	 */
	protected boolean rContainsResultFullReplace( OptNode n, String resultVar, String iterVarname, MatrixObject mo ) 
		throws DMLRuntimeException
	{
		boolean ret = false;
		
		//process hop node
		if( n.getNodeType()==NodeType.HOP )
			ret |= isResultFullReplace(n, resultVar, iterVarname, mo);
			
		//process childs recursively
		if( !n.isLeaf() ) {
			for( OptNode c : n.getChilds() ) 
				ret |= rContainsResultFullReplace(c, resultVar, iterVarname, mo);
		}
		
		return ret;
	}
	
	/**
	 * 
	 * @param n
	 * @param resultVar
	 * @param iterVarname
	 * @param mo
	 * @return
	 * @throws DMLRuntimeException
	 */
	protected boolean isResultFullReplace( OptNode n, String resultVar, String iterVarname, MatrixObject mo ) 
		throws DMLRuntimeException
	{
		//check left indexing operator
		String opStr = n.getParam(ParamType.OPSTRING);
		if( opStr==null || !opStr.equals(LeftIndexingOp.OPSTRING) )
			return false;

		Hop h = OptTreeConverter.getAbstractPlanMapping().getMappedHop(n.getID());
		Hop base = h.getInput().get(0);

		//check result variable
		if( !resultVar.equals(base.getName()) )
			return false;

		//check access pattern, memory budget
		Hop inpRowL = h.getInput().get(2);
		Hop inpRowU = h.getInput().get(3);
		Hop inpColL = h.getInput().get(4);
		Hop inpColU = h.getInput().get(5);
		//check for rowwise overwrite
		if(   (inpRowL.getName().equals(iterVarname) && inpRowU.getName().equals(iterVarname))
		   && inpColL instanceof LiteralOp && HopRewriteUtils.getDoubleValueSafe((LiteralOp)inpColL)==1
		   && inpColU instanceof LiteralOp && HopRewriteUtils.getDoubleValueSafe((LiteralOp)inpColU)==mo.getNumColumns() )
		{
			return true;
		}
		
		//check for colwise overwrite
		if(   (inpColL.getName().equals(iterVarname) && inpColU.getName().equals(iterVarname))
		   && inpRowL instanceof LiteralOp && HopRewriteUtils.getDoubleValueSafe((LiteralOp)inpRowL)==1
		   && inpRowU instanceof LiteralOp && HopRewriteUtils.getDoubleValueSafe((LiteralOp)inpRowU)==mo.getNumRows() )
		{
			return true;
		}
		
		return false;
	}
	
	/**
	 * 
	 * @param n
	 * @param var
	 * @return
	 */
	protected boolean rIsReadInRightIndexing(OptNode n, String var) 
	{
		//NOTE: This method checks if a given variables is used in right indexing
		//expressions. This is sufficient for "remove unnecessary compare matrix" because
		//we already checked for full replace, which is only valid if we dont access
		//the entire matrix in any other operation.
		boolean ret = false;
		
		if( n.getNodeType()==NodeType.HOP ) {
			Hop h = OptTreeConverter.getAbstractPlanMapping().getMappedHop(n.getID());
			if( h instanceof IndexingOp && h.getInput().get(0) instanceof DataOp
				&& h.getInput().get(0).getName().equals(var) )
			{
				ret |= true;
			}
		}
			
		//process childs recursively
		if( !n.isLeaf() )
			for( OptNode c : n.getChilds() )
				ret |= rIsReadInRightIndexing(c, var);
		
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
		
		PResultMerge REMOTE = OptimizerUtils.isSparkExecutionMode() ? 
				PResultMerge.REMOTE_SPARK : PResultMerge.REMOTE_MR;
		PResultMerge ret = null;
		
		//investigate details of current parfor node
		boolean flagRemoteParFOR = (n.getExecType() == ExecType.MR || n.getExecType() == ExecType.SPARK);
		boolean flagLargeResult = hasLargeTotalResults( n, pfpb.getResultVariables(), vars, true );
		boolean flagRemoteLeftIndexing = hasResultMRLeftIndexing( n, pfpb.getResultVariables(), vars, true );
		boolean flagCellFormatWoCompare = determineFlagCellFormatWoCompare(pfpb.getResultVariables(), vars); 
		boolean flagOnlyInMemResults = hasOnlyInMemoryResults(n, pfpb.getResultVariables(), vars, true );
		
		//optimimality decision on result merge
		//MR, if remote exec, and w/compare (prevent huge transfer/merge costs)
		if( flagRemoteParFOR && flagLargeResult )
		{
			ret = REMOTE;
		}
		//CP, if all results in mem	
		else if( flagOnlyInMemResults )
		{
			ret = PResultMerge.LOCAL_MEM;
		}
		//MR, if result partitioning and copy not possible
		//NOTE: 'at least one' instead of 'all' condition of flagMRLeftIndexing because the 
		//      benefit for large matrices outweigths potentially unnecessary MR jobs for smaller matrices)
		else if(    ( flagRemoteParFOR || flagRemoteLeftIndexing) 
			    && !(flagCellFormatWoCompare && ResultMergeLocalFile.ALLOW_COPY_CELLFILES ) )
		{
			ret = REMOTE;
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
			rInvokeSetResultMerge(n.getChilds(), vars, inLocal && !flagRemoteParFOR);
		
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
			if( opName !=null && opName.equals(LeftIndexingOp.OPSTRING) && 
				(n.getExecType() == ExecType.MR || n.getExecType() == ExecType.SPARK) )
			{
				LeftIndexingOp hop = (LeftIndexingOp) OptTreeConverter.getAbstractPlanMapping().getMappedHop(n.getID());
				//check agains set of varname
				String varName = hop.getInput().get(0).getName();
				if( resultVars.contains(varName) )
				{
					ret = true;
					if( checkSize && vars.keySet().contains(varName) )
					{
						//dims of result vars must be known at this point in time
						MatrixObject mo = (MatrixObject) vars.get( hop.getInput().get(0).getName() );
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
				String varName = hop.getInput().get(0).getName();
				if( resultVars.contains(varName) && vars.keySet().contains(varName) )
				{
					//dims of result vars must be known at this point in time
					MatrixObject mo = (MatrixObject) vars.get( hop.getInput().get(0).getName() );
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
				if( n.getExecType()==ExecType.MR || n.getExecType()==ExecType.SPARK )
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

		if( !recPBs.isEmpty() )
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
				FunctionProgramBlock copyfpb = ProgramConverter.createDeepCopyFunctionProgramBlock(fpb, new HashSet<String>(), new HashSet<String>());
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

	/*
	 * This class stores information for the candidate hop, such as hop itself, program block.
	 * When it gets evaluated if Matrix can be marked for "UpdateInPlace", additional properties such
	 * as location, flag to indicate if its in loop (for, parfor, while), flag to indicate if hop can be marked as "UpdateInPlace".
	 */
	class UIPCandidateHop {
		Hop hop;
		int iLocation = -1;
		ProgramBlock pb;
		Boolean bIsLoopApplicable = false, bUpdateInPlace = true;
		ArrayList<Hop> consumerHops = null;
		
		
		UIPCandidateHop(Hop hop, ProgramBlock pb)
		{
			this.hop = hop;
			this.pb = pb;
		}
		
		Hop getHop()
		{
			return hop;
		}
		
		ProgramBlock getProgramBlock()
		{
			return pb;
		}
		
		int getLocation()
		{
			return this.iLocation;
		}
		
		void setLocation(int iLocation)
		{
			this.iLocation = iLocation;
		}
		
		boolean isLoopApplicable()
		{
			return(bIsLoopApplicable);
		}

		void setIsLoopApplicable(boolean bInWhileLoop)
		{
			this.bIsLoopApplicable = bInWhileLoop;
		}

		boolean isUpdateInPlace()
		{
			return(bUpdateInPlace);
		}

		void setUpdateInPlace(boolean bUpdateInPlace)
		{
			this.bUpdateInPlace = bUpdateInPlace;
		}
		
		ArrayList<Hop> getConsumerHops()
		{
			return this.consumerHops;
		}
		
		void setConsumerHops(ArrayList<Hop> consumerHops)
		{
			this.consumerHops = consumerHops;
		}
		
	}
}
