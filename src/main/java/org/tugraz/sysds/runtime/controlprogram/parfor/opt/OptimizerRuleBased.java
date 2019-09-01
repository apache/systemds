/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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

package org.tugraz.sysds.runtime.controlprogram.parfor.opt;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.tugraz.sysds.api.DMLScript;
import org.tugraz.sysds.conf.ConfigurationManager;
import org.tugraz.sysds.hops.AggBinaryOp;
import org.tugraz.sysds.hops.AggBinaryOp.MMultMethod;
import org.tugraz.sysds.hops.BinaryOp;
import org.tugraz.sysds.hops.DataOp;
import org.tugraz.sysds.hops.FunctionOp;
import org.tugraz.sysds.hops.Hop;
import org.tugraz.sysds.hops.Hop.ParamBuiltinOp;
import org.tugraz.sysds.hops.Hop.ReOrgOp;
import org.tugraz.sysds.hops.IndexingOp;
import org.tugraz.sysds.hops.LeftIndexingOp;
import org.tugraz.sysds.hops.LiteralOp;
import org.tugraz.sysds.hops.MemoTable;
import org.tugraz.sysds.hops.MultiThreadedHop;
import org.tugraz.sysds.hops.OptimizerUtils;
import org.tugraz.sysds.hops.ParameterizedBuiltinOp;
import org.tugraz.sysds.hops.ReorgOp;
import org.tugraz.sysds.hops.UnaryOp;
import org.tugraz.sysds.hops.recompile.Recompiler;
import org.tugraz.sysds.hops.rewrite.HopRewriteUtils;
import org.tugraz.sysds.hops.rewrite.ProgramRewriteStatus;
import org.tugraz.sysds.hops.rewrite.ProgramRewriter;
import org.tugraz.sysds.hops.rewrite.RewriteInjectSparkLoopCheckpointing;
import org.tugraz.sysds.lops.LopProperties;
import org.tugraz.sysds.parser.DMLProgram;
import org.tugraz.sysds.parser.FunctionStatementBlock;
import org.tugraz.sysds.parser.ParForStatement;
import org.tugraz.sysds.parser.ParForStatementBlock;
import org.tugraz.sysds.parser.ParForStatementBlock.ResultVar;
import org.tugraz.sysds.parser.StatementBlock;
import org.tugraz.sysds.runtime.DMLRuntimeException;
import org.tugraz.sysds.runtime.controlprogram.BasicProgramBlock;
import org.tugraz.sysds.runtime.controlprogram.ForProgramBlock;
import org.tugraz.sysds.runtime.controlprogram.FunctionProgramBlock;
import org.tugraz.sysds.runtime.controlprogram.LocalVariableMap;
import org.tugraz.sysds.runtime.controlprogram.ParForProgramBlock;
import org.tugraz.sysds.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import org.tugraz.sysds.runtime.controlprogram.ParForProgramBlock.PDataPartitioner;
import org.tugraz.sysds.runtime.controlprogram.ParForProgramBlock.PExecMode;
import org.tugraz.sysds.runtime.controlprogram.ParForProgramBlock.POptMode;
import org.tugraz.sysds.runtime.controlprogram.ParForProgramBlock.PResultMerge;
import org.tugraz.sysds.runtime.controlprogram.ParForProgramBlock.PTaskPartitioner;
import org.tugraz.sysds.runtime.controlprogram.ParForProgramBlock.PartitionFormat;
import org.tugraz.sysds.runtime.controlprogram.Program;
import org.tugraz.sysds.runtime.controlprogram.ProgramBlock;
import org.tugraz.sysds.runtime.controlprogram.caching.MatrixObject;
import org.tugraz.sysds.runtime.controlprogram.caching.MatrixObject.UpdateType;
import org.tugraz.sysds.runtime.controlprogram.context.ExecutionContext;
import org.tugraz.sysds.runtime.controlprogram.context.SparkExecutionContext;
import org.tugraz.sysds.runtime.controlprogram.parfor.ResultMergeLocalFile;
import org.tugraz.sysds.runtime.controlprogram.parfor.opt.CostEstimator.ExcludeType;
import org.tugraz.sysds.runtime.controlprogram.parfor.opt.CostEstimator.TestMeasure;
import org.tugraz.sysds.runtime.controlprogram.parfor.opt.OptNode.ExecType;
import org.tugraz.sysds.runtime.controlprogram.parfor.opt.OptNode.NodeType;
import org.tugraz.sysds.runtime.controlprogram.parfor.opt.OptNode.ParamType;
import org.tugraz.sysds.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.tugraz.sysds.runtime.data.SparseRowVector;
import org.tugraz.sysds.runtime.instructions.Instruction;
import org.tugraz.sysds.runtime.instructions.cp.Data;
import org.tugraz.sysds.runtime.instructions.cp.FunctionCallCPInstruction;
import org.tugraz.sysds.runtime.instructions.gpu.context.GPUContextPool;
import org.tugraz.sysds.runtime.instructions.spark.data.RDDObject;
import org.tugraz.sysds.runtime.io.IOUtilFunctions;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.matrix.data.OutputInfo;
import org.tugraz.sysds.runtime.meta.DataCharacteristics;
import org.tugraz.sysds.runtime.meta.MetaDataFormat;
import org.tugraz.sysds.runtime.util.ProgramConverter;
import org.tugraz.sysds.utils.NativeHelper;
import org.tugraz.sysds.yarn.ropt.YarnClusterAnalyzer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map.Entry;
import java.util.Set;
import java.util.stream.Collectors;

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
 *      columns/rows according to blocksize -&gt; rewrite (only applicable if 
 *      numCols/blocksize&gt;numreducers)+custom MR partitioner)
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
	public CostModelType getCostModelType() {
		return CostModelType.STATIC_MEM_METRIC;
	}


	@Override
	public PlanInputType getPlanInputType() {
		return PlanInputType.ABSTRACT_PLAN;
	}

	@Override
	public POptMode getOptMode() {
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
	{
		LOG.debug("--- "+getOptMode()+" OPTIMIZER -------");

		OptNode pn = plan.getRoot();
		
		//early abort for empty parfor body 
		if( pn.isLeaf() )
			return true;
		
		//ANALYZE infrastructure properties
		analyzeProblemAndInfrastructure( pn );
		
		_cost = est;
		
		//debug and warnings output
		if( LOG.isDebugEnabled() ) {
			LOG.debug(getOptMode()+" OPT: Optimize w/ max_mem="+toMB(_lm)+"/"+toMB(_rm)+"/"+toMB(_rm2)+", max_k="+_lk+"/"+_rk+"/"+_rk2+")." );
			if( OptimizerUtils.isSparkExecutionMode() )
				LOG.debug(getOptMode()+" OPT: Optimize w/ "+SparkExecutionContext.getSparkClusterConfig().toString());
			if( _rnk <= 0 || _rk <= 0 )
				LOG.warn(getOptMode()+" OPT: Optimize for inactive cluster (num_nodes="+_rnk+", num_map_slots="+_rk+")." );
		}
		
		//ESTIMATE memory consumption 
		pn.setSerialParFor(); //for basic mem consumption 
		double M0a = _cost.getEstimate(TestMeasure.MEMORY_USAGE, pn);
		LOG.debug(getOptMode()+" OPT: estimated mem (serial exec) M="+toMB(M0a) );
		
		//OPTIMIZE PARFOR PLAN
		
		// rewrite 1: data partitioning (incl. log. recompile RIX and flag opt nodes)
		HashMap<String, PartitionFormat> partitionedMatrices = new HashMap<>();
		rewriteSetDataPartitioner( pn, ec.getVariables(), partitionedMatrices, OptimizerUtils.getLocalMemBudget(), false );
		double M0b = _cost.getEstimate(TestMeasure.MEMORY_USAGE, pn); //reestimate
		
		// rewrite 2: remove unnecessary compare matrix (before result partitioning)
		rewriteRemoveUnnecessaryCompareMatrix(pn, ec);
		
		// rewrite 3: rewrite result partitioning (incl. log/phy recompile LIX) 
		boolean flagLIX = rewriteSetResultPartitioning( pn, M0b, ec.getVariables() );
		double M1 = _cost.getEstimate(TestMeasure.MEMORY_USAGE, pn); //reestimate 
		LOG.debug(getOptMode()+" OPT: estimated new mem (serial exec) M="+toMB(M1) );
		
		//determine memory consumption for what-if: all-cp or partitioned 
		double M2 = pn.isCPOnly() ? M1 :
			_cost.getEstimate(TestMeasure.MEMORY_USAGE, pn, LopProperties.ExecType.CP);
		LOG.debug(getOptMode()+" OPT: estimated new mem (serial exec, all CP) M="+toMB(M2) );
		double M3 = _cost.getEstimate(TestMeasure.MEMORY_USAGE, pn, true);
		LOG.debug(getOptMode()+" OPT: estimated new mem (cond partitioning) M="+toMB(M3) );
		
		// rewrite 4: execution strategy
		boolean flagRecompMR = rewriteSetExecutionStategy( pn, M0a, M1, M2, M3, flagLIX );
		
		//exec-type-specific rewrites
		if( pn.getExecType() == getRemoteExecType() )
		{
			if( M1 > _rm && M3 <= _rm  ) {
				// rewrite 1: data partitioning (apply conditional partitioning)
				rewriteSetDataPartitioner( pn, ec.getVariables(), partitionedMatrices, M3, false );
				M1 = _cost.getEstimate(TestMeasure.MEMORY_USAGE, pn); //reestimate
			}
			
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
			
			// rewrite 10: determine parallelism
			rewriteSetDegreeOfParallelism( pn, _cost, ec.getVariables(), M1, false );
			
			// rewrite 11: task partitioning 
			rewriteSetTaskPartitioner( pn, false, flagLIX );
			
			// rewrite 12: fused data partitioning and execution
			rewriteSetFusedDataPartitioningExecution(pn, M1, flagLIX, partitionedMatrices, ec.getVariables());
			
			// rewrite 14: set in-place result indexing
			HashSet<ResultVar> inplaceResultVars = new HashSet<>();
			rewriteSetInPlaceResultIndexing(pn, _cost, ec.getVariables(), inplaceResultVars, ec);
		}
		else //if( pn.getExecType() == ExecType.CP )
		{
			// rewrite 10: determine parallelism
			rewriteSetDegreeOfParallelism( pn, _cost, ec.getVariables(), M1, false );
			
			// rewrite 11: task partitioning
			rewriteSetTaskPartitioner( pn, false, false ); //flagLIX always false 
			
			// rewrite 14: set in-place result indexing
			HashSet<ResultVar> inplaceResultVars = new HashSet<>();
			rewriteSetInPlaceResultIndexing(pn, _cost, ec.getVariables(), inplaceResultVars, ec);
			
			//rewrite 17: checkpoint injection for parfor loop body
			rewriteInjectSparkLoopCheckpointing( pn );
			
			//rewrite 18: repartition read-only inputs for zipmm 
			rewriteInjectSparkRepartition( pn, ec.getVariables() );
			
			//rewrite 19: eager caching for checkpoint rdds
			rewriteSetSparkEagerRDDCaching( pn, ec.getVariables() );
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

	protected void analyzeProblemAndInfrastructure( OptNode pn )
	{
		_N       = Long.parseLong(pn.getParam(ParamType.NUM_ITERATIONS));
		_Nmax    = pn.getMaxProblemSize(); 
		_lk      = InfrastructureAnalyzer.getLocalParallelism();
		_lkmaxCP = (int) Math.ceil( PAR_K_FACTOR * _lk ); 
		_lkmaxMR = (int) Math.ceil( PAR_K_MR_FACTOR * _lk );
		_lm      = OptimizerUtils.getLocalMemBudget();
		
		//spark-specific cluster characteristics
		if( OptimizerUtils.isSparkExecutionMode() ) {
			//we get all required cluster characteristics from spark's configuration
			//to avoid invoking yarns cluster status
			_rnk = SparkExecutionContext.getNumExecutors(); 
			_rk  = (int) SparkExecutionContext.getDefaultParallelism(true);
			_rk2 = _rk; //equal map/reduce unless we find counter-examples 
			int cores = SparkExecutionContext.getDefaultParallelism(true)
					/ SparkExecutionContext.getNumExecutors();
			int ccores = Math.max((int) Math.min(cores, _N), 1);
			_rm  = SparkExecutionContext.getBroadcastMemoryBudget() / ccores;
			_rm2 = SparkExecutionContext.getBroadcastMemoryBudget() / ccores;
		}
		//mr/yarn-specific cluster characteristics
		else {
			_rnk = InfrastructureAnalyzer.getRemoteParallelNodes();  
			_rk  = InfrastructureAnalyzer.getRemoteParallelMapTasks();
			_rk2 = InfrastructureAnalyzer.getRemoteParallelReduceTasks();
			_rm  = OptimizerUtils.getRemoteMemBudgetMap(false); 	
			_rm2 = OptimizerUtils.getRemoteMemBudgetReduce(); 	
		
			//correction of max parallelism if yarn enabled because yarn
			//does not have the notion of map/reduce slots and hence returns 
			//small constants of map=10*nodes, reduce=2*nodes
			//(not doing this correction would loose available degree of parallelism)
			if( InfrastructureAnalyzer.isYarnEnabled() ) {
				long tmprk = YarnClusterAnalyzer.getNumCores();
				_rk  = (int) Math.max( _rk, tmprk );
				_rk2 = (int) Math.max( _rk2, tmprk/2 );
			}
		}
		
		_rkmax   = (int) Math.ceil( PAR_K_FACTOR * _rk ); 
		_rkmax2  = (int) Math.ceil( PAR_K_FACTOR * _rk2 ); 
	}
	
	protected ExecType getRemoteExecType() {
		return ExecType.SPARK;
	}
	
	///////
	//REWRITE set data partitioner
	///

	protected boolean rewriteSetDataPartitioner(OptNode n, LocalVariableMap vars, HashMap<String, PartitionFormat> partitionedMatrices, double thetaM, boolean constrained ) 
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
			HashMap<String, PartitionFormat> cand2 = new HashMap<>();
			for( String c : pfsb.getReadOnlyParentMatrixVars() ) {
				PartitionFormat dpf = pfsb.determineDataPartitionFormat( c );
				double mem = getMemoryEstimate(c, vars);
				if( dpf != PartitionFormat.NONE 
					&& dpf._dpf != PDataPartitionFormat.BLOCK_WISE_M_N
					&& (constrained || (mem > _lm/2 && mem > _rm/2))
					&& vars.get(c) != null //robustness non-existing vars
					&& !vars.get(c).getDataType().isList() ) {
					cand2.put( c, dpf );
				}
			}
			apply = rFindDataPartitioningCandidates(n, cand2, vars, thetaM);
			if( apply )
				partitionedMatrices.putAll(cand2);
		}
		
		PDataPartitioner REMOTE = PDataPartitioner.REMOTE_SPARK;
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
			" ("+Arrays.toString(partitionedMatrices.keySet().toArray())+")" );
		
		return blockwise;
	}

	protected boolean rFindDataPartitioningCandidates( OptNode n, HashMap<String, PartitionFormat> cand, LocalVariableMap vars, double thetaM ) 
	{
		boolean ret = false;

		if( !n.isLeaf() ) {
			for( OptNode cn : n.getChilds() )
				if( cn.getNodeType() != NodeType.FUNCCALL ) //prevent conflicts with aliases
					ret |= rFindDataPartitioningCandidates( cn, cand, vars, thetaM );
		}
		else if( n.getNodeType()== NodeType.HOP
			&& n.getParam(ParamType.OPSTRING).equals(IndexingOp.OPSTRING) )
		{
			Hop h = OptTreeConverter.getAbstractPlanMapping().getMappedHop(n.getID());
			String inMatrix = h.getInput().get(0).getName();
			if( cand.containsKey(inMatrix) && h.getDataType().isMatrix() ) //Required: partitionable
			{
				PartitionFormat dpf = cand.get(inMatrix);
				double mnew = getNewRIXMemoryEstimate( n, inMatrix, dpf, vars );
				//NOTE: for the moment, we do not partition according to the remote mem, because we can execute 
				//it even without partitioning in CP. However, advanced optimizers should reason about this
				//double mold = h.getMemEstimate();
				if( n.getExecType() == getRemoteExecType()  //Opt Condition: MR/Spark
					|| h.getMemEstimate() > thetaM ) //Opt Condition: mem estimate > constraint to force partitioning
				{
					//NOTE: subsequent rewrites will still use the MR mem estimate
					//(guarded by subsequent operations that have at least the memory req of one partition)
					n.setExecType(ExecType.CP); //partition ref only (see below)
					n.addParam(ParamType.DATA_PARTITION_FORMAT, dpf.toString());
					h.setMemEstimate( mnew ); //CP vs CP_FILE in ProgramRecompiler bases on mem_estimate
					ret = true;
				}
				//keep track of nodes that allow conditional data partitioning and their mem
				else
				{
					n.addParam(ParamType.DATA_PARTITION_COND, String.valueOf(true));
					n.addParam(ParamType.DATA_PARTITION_COND_MEM, String.valueOf(mnew));
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
	 * @param n internal representation of a plan alternative for program blocks and instructions
	 * @param varName variable name
	 * @param dpf data partition format
	 * @param vars local variable map
	 * @return memory estimate
	 */
	protected double getNewRIXMemoryEstimate( OptNode n, String varName, PartitionFormat dpf, LocalVariableMap vars ) 
	{
		double mem = -1;
		
		//not all intermediates need to be known or existing on optimize
		Data dat = vars.get( varName );
		if( dat != null && dat instanceof MatrixObject )
		{
			MatrixObject mo = (MatrixObject) dat;
			
			//those are worst-case (dense) estimates
			switch( dpf._dpf )
			{
				case COLUMN_WISE:
					mem = OptimizerUtils.estimateSize(mo.getNumRows(), 1); 
					break;
				case ROW_WISE:
					mem = OptimizerUtils.estimateSize(1, mo.getNumColumns());
					break;
				case COLUMN_BLOCK_WISE_N:
					mem = OptimizerUtils.estimateSize(mo.getNumRows(), dpf._N); 
					break;
				case ROW_BLOCK_WISE_N:
					mem = OptimizerUtils.estimateSize(dpf._N, mo.getNumColumns()); 
					break;
				default:
					//do nothing
			}
		}
		
		return mem;
	}
	
	protected double getMemoryEstimate(String varName, LocalVariableMap vars) {
		Data dat = vars.get(varName);
		return (dat instanceof MatrixObject) ? 
			OptimizerUtils.estimateSize(((MatrixObject)dat).getDataCharacteristics()) :
			OptimizerUtils.DEFAULT_SIZE;
	}

	protected static LopProperties.ExecType getRIXExecType( MatrixObject mo, PDataPartitionFormat dpf, boolean withSparsity ) 
	{
		double mem = -1;
		
		long rlen = mo.getNumRows();
		long clen = mo.getNumColumns();
		long blen = mo.getBlocksize();
		long nnz = mo.getNnz();
		double lsparsity = ((double)nnz)/rlen/clen;		
		double sparsity = withSparsity ? lsparsity : 1.0;
		
		switch( dpf )
		{
			case COLUMN_WISE:
				mem = OptimizerUtils.estimateSizeExactSparsity(mo.getNumRows(), 1, sparsity); 
				break;
			case COLUMN_BLOCK_WISE:
				mem = OptimizerUtils.estimateSizeExactSparsity(mo.getNumRows(), blen, sparsity); 
				break;
			case ROW_WISE:
				mem = OptimizerUtils.estimateSizeExactSparsity(1, mo.getNumColumns(), sparsity);
				break;
			case ROW_BLOCK_WISE:
				mem = OptimizerUtils.estimateSizeExactSparsity(blen, mo.getNumColumns(), sparsity);
				break;
				
			default:
				//do nothing
		}
		
		if( mem < OptimizerUtils.getLocalMemBudget() )
			return LopProperties.ExecType.CP;
		else
			return LopProperties.ExecType.CP_FILE;
	}

	public static boolean allowsBinaryCellPartitions( MatrixObject mo, PartitionFormat dpf ) {
		return (getRIXExecType(mo, PDataPartitionFormat.COLUMN_BLOCK_WISE, false)==LopProperties.ExecType.CP );
	}
	
	///////
	//REWRITE set result partitioning
	///

	protected boolean rewriteSetResultPartitioning(OptNode n, double M, LocalVariableMap vars) {
		//preparations
		long id = n.getID();
		Object[] o = OptTreeConverter.getAbstractPlanMapping().getMappedProg(id);
		ParForProgramBlock pfpb = (ParForProgramBlock) o[1];
		
		//search for candidates
		Collection<OptNode> cand = n.getNodeList(getRemoteExecType());
		
		//determine if applicable
		boolean apply = M < _rm   //ops fit in remote memory budget
			&& !cand.isEmpty()    //at least one MR
			&& isResultPartitionableAll(cand,pfpb.getResultVariables(), 
				vars, pfpb.getIterVar()); // check candidates
		
		//recompile LIX
		if( apply )
		{
			try {
				for(OptNode lix : cand)
					recompileLIX( lix, vars );
			}
			catch(Exception ex) {
				throw new DMLRuntimeException("Unable to recompile LIX.", ex);
			}
		}
		
		_numEvaluatedPlans++;
		LOG.debug(getOptMode()+" OPT: rewrite 'set result partitioning' - result="+apply );
	
		return apply;
	}

	protected boolean isResultPartitionableAll( Collection<OptNode> nlist, ArrayList<ResultVar> resultVars, LocalVariableMap vars, String iterVarname ) {
		boolean ret = true;
		for( OptNode n : nlist ) {
			ret &= isResultPartitionable(n, resultVars, vars, iterVarname);
			if(!ret) //early abort
				break;
		}
		return ret;
	}

	protected boolean isResultPartitionable( OptNode n, ArrayList<ResultVar> resultVars, LocalVariableMap vars, String iterVarname ) 
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
			if( !ResultVar.contains(resultVars, base.getName()) )
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

	private static double estimateSizeSparseRowBlock( long rows ) {
		//see MatrixBlock.estimateSizeSparseInMemory
		return 44 + rows * 8;
	}

	private static double estimateSizeSparseRow( long cols, long nnz ) {
		//see MatrixBlock.estimateSizeSparseInMemory
		long cnnz = Math.max(SparseRowVector.initialCapacity, Math.max(cols, nnz));
		return ( 116 + 12 * cnnz ); //sparse row
	}

	private static  double estimateSizeSparseRowMin( long cols ) {
		//see MatrixBlock.estimateSizeSparseInMemory
		long cnnz = Math.min(SparseRowVector.initialCapacity, cols);
		return ( 116 + 12 * cnnz ); //sparse row
	}

	private static int estimateNumTasksSparseCol( double budget, long rows ) {
		//see MatrixBlock.estimateSizeSparseInMemory
		double lbudget = budget - rows * 116;
		return (int) Math.floor( lbudget / 12 );
	}

	protected void recompileLIX( OptNode n, LocalVariableMap vars ) {
		Hop h = OptTreeConverter.getAbstractPlanMapping().getMappedHop(n.getID());
		
		//set forced exec type
		h.setForcedExecType(LopProperties.ExecType.CP);
		n.setExecType(ExecType.CP);
		
		//recompile parent pb
		long pid = OptTreeConverter.getAbstractPlanMapping().getMappedParentID(n.getID());
		OptNode nParent = OptTreeConverter.getAbstractPlanMapping().getOptNode(pid);
		Object[] o = OptTreeConverter.getAbstractPlanMapping().getMappedProg(pid);
		StatementBlock sb = (StatementBlock) o[0];
		BasicProgramBlock pb = (BasicProgramBlock) o[1];
		
		//keep modified estimated of partitioned rix (in same dag as lix)
		HashMap<Hop, Double> estRix = getPartitionedRIXEstimates(nParent);
		
		//construct new instructions
		ArrayList<Instruction> newInst = Recompiler.recompileHopsDag(
			sb, sb.getHops(), vars, null, false, false, 0);
		pb.setInstructions( newInst );
		
		//reset all rix estimated (modified by recompile)
		resetPartitionRIXEstimates( estRix );
		
		//set new mem estimate (last, otherwise overwritten from recompile)
		h.setMemEstimate(_rm-1);
	}

	protected HashMap<Hop, Double> getPartitionedRIXEstimates(OptNode parent)
	{
		HashMap<Hop, Double> estimates = new HashMap<>();
		for( OptNode n : parent.getChilds() )
			if( n.getParam(ParamType.DATA_PARTITION_FORMAT) != null )
			{
				Hop h = OptTreeConverter.getAbstractPlanMapping().getMappedHop(n.getID());
				estimates.put( h, h.getMemEstimate() );
			}
		return estimates;
	}

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

	protected boolean rewriteSetExecutionStategy(OptNode n, double M0, double M, double M2, double M3, boolean flagLIX) {
		boolean isCPOnly = n.isCPOnly();
		boolean isCPOnlyPossible = isCPOnly || isCPOnlyPossible(n, _rm);

		String datapartitioner = n.getParam(ParamType.DATA_PARTITIONER);
		ExecType REMOTE = getRemoteExecType();
		PDataPartitioner REMOTE_DP = PDataPartitioner.REMOTE_SPARK;

		//deciding on the execution strategy
		if( ConfigurationManager.isParallelParFor()  //allowed remote parfor execution
			&& ( (isCPOnly && M <= _rm )             //Required: all inst already in cp and fit in remote mem
			   ||(isCPOnly && M3 <= _rm ) 	         //Required: all inst already in cp and fit partitioned in remote mem
			   ||(isCPOnlyPossible && M2 <= _rm)) )  //Required: all inst forced to cp fit in remote mem
		{
			//at this point all required conditions for REMOTE_MR given, now its an opt decision
			int cpk = (int) Math.min( _lk, Math.floor( _lm / M ) ); //estimated local exploited par  
			
			//MR if local par cannot be exploited due to mem constraints (this implies that we work on large data)
			//(the factor of 2 is to account for hyper-threading and in order prevent too eager remote parfor)
			if( 2*cpk < _lk && 2*cpk < _N && 2*cpk < _rk ) //incl conditional partitioning
			{
				n.setExecType( REMOTE ); //remote parfor
			}
			//MR if problem is large enough and remote parallelism is larger than local   
			else if( _lk < _N && _lk < _rk && M <= _rm && isLargeProblem(n, M0) )
			{
				n.setExecType( REMOTE ); //remote parfor
			}
			//MR if MR operations in local, but CP only in remote (less overall MR jobs)
			else if( !isCPOnly && isCPOnlyPossible )
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
		boolean requiresRecompile = (mode == PExecMode.REMOTE_SPARK && !isCPOnly);
		
		_numEvaluatedPlans++;
		LOG.debug(getOptMode()+" OPT: rewrite 'set execution strategy' - result="+mode+" (recompile="+requiresRecompile+")" );
		
		return requiresRecompile;
	}

	protected boolean isLargeProblem(OptNode pn, double M)
	{
		//TODO get a proper time estimate based to capture compute-intensive scenarios
		
		//rule-based decision based on number of outer iterations or maximum number of
		//inner iterations (w/ appropriately scaled minimum data size threshold); 
		boolean isCtxCreated = OptimizerUtils.isSparkExecutionMode()
				&& SparkExecutionContext.isSparkContextCreated();
		return (_N >= PROB_SIZE_THRESHOLD_REMOTE && M > PROB_SIZE_THRESHOLD_MB)
			|| (_Nmax >= 10 * PROB_SIZE_THRESHOLD_REMOTE
				&& M > PROB_SIZE_THRESHOLD_MB/(isCtxCreated?10:1));
	}

	protected boolean isCPOnlyPossible( OptNode n, double memBudget ) {
		ExecType et = n.getExecType();
		boolean ret = ( et == ExecType.CP);
		
		if( n.isLeaf() && et == getRemoteExecType() )
		{
			Hop h = OptTreeConverter.getAbstractPlanMapping().getMappedHop( n.getID() );
			if(    h.getForcedExecType()!=LopProperties.ExecType.SPARK 
				&& h.hasValidCPDimsAndSize() ) //integer dims
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

	protected void rewriteSetOperationsExecType(OptNode pn, boolean recompile) {
		//set exec type in internal opt tree
		int count = setOperationExecType(pn, ExecType.CP);
		
		//recompile program (actual programblock modification)
		if( recompile && count<=0 )
			LOG.warn("OPT: Forced set operations exec type 'CP', but no operation requires recompile.");
		ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
			.getAbstractPlanMapping().getMappedProg(pn.getID())[1];
		HashSet<String> fnStack = new HashSet<>();
		Recompiler.recompileProgramBlockHierarchy2Forced(pfpb.getChildBlocks(), 0, fnStack, LopProperties.ExecType.CP);
		
		//debug output
		LOG.debug(getOptMode()+" OPT: rewrite 'set operation exec type CP' - result="+count);
	}

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
	 * @param n internal representation of a plan alternative for program blocks and instructions
	 * @param vars local variable map
	 */
	protected void rewriteDataColocation( OptNode n, LocalVariableMap vars ) {
		// data colocation is beneficial if we have dp=REMOTE_MR, etype=REMOTE_MR
		// and there is at least one direct col-/row-wise access with the index variable
		// on the partitioned matrix
		boolean apply = false;
		String varname = null;
		ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
				.getAbstractPlanMapping().getMappedProg(n.getID())[1];
		
		//modify the runtime plan (apply true if at least one candidate)
		if( apply )
			pfpb.enableColocatedPartitionedMatrix( varname );
		
		_numEvaluatedPlans++;
		LOG.debug(getOptMode()+" OPT: rewrite 'enable data colocation' - result="+apply+((apply)?" ("+varname+")":"") );
	}

	protected void rFindDataColocationCandidates( OptNode n, HashSet<String> cand, String iterVarname ) {
		if( !n.isLeaf() )
		{
			for( OptNode cn : n.getChilds() )
				rFindDataColocationCandidates( cn, cand, iterVarname );
		}
		else if(    n.getNodeType()== NodeType.HOP
			     && n.getParam(ParamType.OPSTRING).equals(IndexingOp.OPSTRING)
			     && n.getParam(ParamType.DATA_PARTITION_FORMAT) != null )
		{
			PartitionFormat dpf = PartitionFormat.valueOf(n.getParam(ParamType.DATA_PARTITION_FORMAT));
			Hop h = OptTreeConverter.getAbstractPlanMapping().getMappedHop(n.getID());
			String inMatrix = h.getInput().get(0).getName();
			String indexAccess = null;
			switch( dpf._dpf )
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
	 * @param n internal representation of a plan alternative for program blocks and instructions
	 * @param partitionedMatrices map of data partition formats
	 * @param vars local variable map
	 */
	protected void rewriteSetPartitionReplicationFactor( OptNode n, HashMap<String, PartitionFormat> partitionedMatrices, LocalVariableMap vars ) 
	{
		boolean apply = false;
		double sizeReplicated = 0;
		int replication = ParForProgramBlock.WRITE_REPLICATION_FACTOR;
		
		ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
			.getAbstractPlanMapping().getMappedProg(n.getID())[1];
		
		if(((n.getExecType()==ExecType.SPARK && n.getParam(ParamType.DATA_PARTITIONER).equals(PDataPartitioner.REMOTE_SPARK.name())))
		    && n.hasNestedParallelism(false) 
		    && n.hasNestedPartitionReads(false) )
		{
			apply = true;
			
			//account for problem and cluster constraints
			replication = (int)Math.min( _N, _rnk );
			
			//account for internal max constraint (note hadoop will warn if max > 10)
			replication = (int)Math.min( replication, MAX_REPLICATION_FACTOR_PARTITIONING );
			
			//account for remaining hdfs capacity
			try {
				FileSystem fs = IOUtilFunctions.getFileSystem(ConfigurationManager.getCachedJobConf());
				long hdfsCapacityRemain = fs.getStatus().getRemaining();
				long sizeInputs = 0; //sum of all input sizes (w/o replication)
				for( String var : partitionedMatrices.keySet() ) {
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
	 * @param n internal representation of a plan alternative for program blocks and instructions
	 * @param vars local variable map
	 */
	protected void rewriteSetExportReplicationFactor( OptNode n, LocalVariableMap vars ) 
	{
		boolean apply = false;
		int replication = -1;
		
		ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
			.getAbstractPlanMapping().getMappedProg(n.getID())[1];
		
		//decide on the replication factor 
		if( n.getExecType()==getRemoteExecType() )
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

	/**
	 * Calculates the maximum memory needed in a CP only Parfor
	 * based on the {@link Hop#computeMemEstimate(MemoTable)}  } function
	 * called recursively for the "children" of the parfor {@link OptNode}.
	 *
	 * @param n the parfor {@link OptNode}
	 * @return the maximum memory needed for any operation inside a parfor in CP execution mode
	 */
	protected double getMaxCPOnlyBudget(OptNode n) {
		ExecType et = n.getExecType();
		double ret = 0;

		if (n.isLeaf() && et != getRemoteExecType()) {
			Hop h = OptTreeConverter.getAbstractPlanMapping().getMappedHop(n.getID());
			if ( h.getForcedExecType() != LopProperties.ExecType.SPARK) {
				double mem = _cost.getLeafNodeEstimate(TestMeasure.MEMORY_USAGE, n, LopProperties.ExecType.CP);
				if (mem >= OptimizerUtils.DEFAULT_SIZE) {
					// memory estimate for worst case scenario.
					// optimistically ignoring this
				} else {
					ret = Math.max(ret, mem);
				}
			}
		}

		if (!n.isLeaf()) {
			for (OptNode c : n.getChilds()) {
				ret = Math.max(ret, getMaxCPOnlyBudget(c));
			}
		}
		return ret;
	}

	///////
	//REWRITE set degree of parallelism
	///

	protected void rewriteSetDegreeOfParallelism(OptNode n, CostEstimator cost, LocalVariableMap vars, double M, boolean flagNested) 
	{
		ExecType type = n.getExecType();
		long id = n.getID();
		
		//special handling for different exec models (CP, MR, MR nested)
		Object[] map = OptTreeConverter.getAbstractPlanMapping().getMappedProg(id);
		ParForStatementBlock pfsb = (ParForStatementBlock)map[0];
		ParForProgramBlock pfpb = (ParForProgramBlock)map[1];
		
		if( type == ExecType.CP ) 
		{
			//determine local max parallelism constraint
			int kMax = ConfigurationManager.isParallelParFor() ?
				(n.isCPOnly() ? _lkmaxCP : _lkmaxMR) : 1;
			
			//compute memory budgets and partial estimates for handling shared reads
			double mem = (OptimizerUtils.isSparkExecutionMode() && !n.isCPOnly()) ? _lm/2 : _lm;
			double sharedM = 0, nonSharedM = M;
			if( computeMaxK(M, M, 0, mem) < kMax ) { //account for shared read if necessary
				sharedM = pfsb.getReadOnlyParentMatrixVars().stream().map(s -> vars.get(s))
					.filter(d -> d instanceof MatrixObject).mapToDouble(mo -> OptimizerUtils
					.estimateSize(((MatrixObject)mo).getDataCharacteristics())).sum();
				nonSharedM = cost.getEstimate(TestMeasure.MEMORY_USAGE, n, true,
					pfsb.getReadOnlyParentMatrixVars(), ExcludeType.SHARED_READ);
			}
			
			//ensure local memory constraint (for spark more conservative in order to 
			//prevent unnecessary guarded collect)
			kMax = Math.min( kMax, computeMaxK(M, nonSharedM, sharedM, mem) );
			kMax = Math.max( kMax, 1);
			
			//constrain max parfor parallelism by problem size
			int parforK = (int)((_N<kMax)? _N : kMax);
			
			// if gpu mode is enabled, the amount of parallelism is set to
			// the smaller of the number of iterations and the number of GPUs
			// otherwise it default to the number of CPU cores and the
			// operations are run in CP mode
			//FIXME rework for nested parfor parallelism and body w/o gpu ops
			if (DMLScript.USE_ACCELERATOR) {
				long perGPUBudget = GPUContextPool.initialGPUMemBudget();
				double maxMemUsage = getMaxCPOnlyBudget(n);
				if (maxMemUsage < perGPUBudget){
					parforK = GPUContextPool.getDeviceCount();
					parforK = Math.min(parforK, (int)_N);
					LOG.debug("Setting degree of parallelism + [" + parforK + "] for GPU; per GPU budget :[" +
							perGPUBudget + "], parfor budget :[" + maxMemUsage + "],  max parallelism per GPU : [" +
							parforK + "]");
				}
			}
			
			//set parfor degree of parallelism
			pfpb.setDegreeOfParallelism(parforK);
			n.setK(parforK);
			
			//distribute remaining parallelism 
			int remainParforK = getRemainingParallelismParFor(kMax, parforK);
			int remainOpsK = getRemainingParallelismOps(_lkmaxCP, parforK);
			rAssignRemainingParallelism( n, remainParforK, remainOpsK );
		}
		else // ExecType.MR/ExecType.SPARK
		{
			int kMax = -1;
			if( flagNested ) {
				//determine remote max parallelism constraint
				pfpb.setDegreeOfParallelism( _rnk ); //guaranteed <= _N (see nested)
				n.setK( _rnk );
				kMax = _rkmax / _rnk; //per node (CP only inside)
			}
			else { //not nested (default)
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
	
	private int computeMaxK(double M, double memNonShared, double memShared, double memBudget) {
		//note: we compute max K for both w/o and w/ shared reads and take the max, because
		//the latter might reduce the degree of parallelism if shared reads don't dominate
		int k1 = (int)Math.floor(memBudget / M);
		int k2 = (int)Math.floor(memBudget-memShared / memNonShared);
		return Math.max(k1, k2);
	}

	protected void rAssignRemainingParallelism(OptNode n, int parforK, int opsK) 
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
					ParForProgramBlock pfpb = (ParForProgramBlock) 
						OptTreeConverter.getAbstractPlanMapping().getMappedProg(id)[1];
					pfpb.setDegreeOfParallelism(tmpK);
					
					//distribute remaining parallelism
					int remainParforK = getRemainingParallelismParFor(parforK, tmpK);
					int remainOpsK = getRemainingParallelismOps(opsK, tmpK);
					rAssignRemainingParallelism(c, remainParforK, remainOpsK);
				}
				else if( c.getNodeType() == NodeType.HOP )
				{
					//set degree of parallelism for multi-threaded leaf nodes
					Hop h = OptTreeConverter.getAbstractPlanMapping().getMappedHop(c.getID());
					if(    ConfigurationManager.isParallelMatrixOperations() 
						&& h instanceof MultiThreadedHop //abop, datagenop, qop, paramop
						&& !( h instanceof ParameterizedBuiltinOp //paramop-grpagg, rexpand, paramserv
							 && !HopRewriteUtils.isValidOp(((ParameterizedBuiltinOp)h).getOp(), 
								ParamBuiltinOp.GROUPEDAGG, ParamBuiltinOp.REXPAND, ParamBuiltinOp.PARAMSERV))
						&& !( h instanceof UnaryOp //only unaryop-cumulativeagg
							 && !((UnaryOp)h).isCumulativeUnaryOperation()
							 && !((UnaryOp)h).isExpensiveUnaryOperation())
						&& !( h instanceof ReorgOp //only reorgop-transpose
							 && ((ReorgOp)h).getOp() != ReOrgOp.TRANS )
						&& !( h instanceof BinaryOp && h.getDataType().isScalar() ) )
					{
						MultiThreadedHop mhop = (MultiThreadedHop) h;
						mhop.setMaxNumThreads(opsK); //set max constraint in hop
						c.setK(opsK); //set optnode k (for explain)
						//need to recompile SB, if changed constraint
						recompileSB = true;
					}
					//for all other multi-threaded hops set k=1 to simply debugging
					else if( h instanceof MultiThreadedHop ) {
						MultiThreadedHop mhop = (MultiThreadedHop) h;
						mhop.setMaxNumThreads(1); //set max constraint in hop
						c.setK(1); //set optnode k (for explain)
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
	
	protected static int getRemainingParallelismParFor(int parforK, int tmpK) {
		//compute max remaining parfor parallelism k such that k * tmpK <= parforK
		return (int)Math.ceil((double)(parforK-tmpK+1) / tmpK);
	}
	
	protected static int getRemainingParallelismOps(int opsK, int tmpK) {
		//compute max remaining operations parallelism k with slight over-provisioning 
		//such that k * tmpK <= 1.5 * opsK; note that if parfor already exploits the
		//maximum parallelism, this will not introduce any over-provisioning.
		//(when running with native BLAS/DNN libraries, we disable over-provisioning
		//to avoid internal SIGFPE and allocation buffer issues w/ MKL and OpenBlas)
		return NativeHelper.isNativeLibraryLoaded() ?
			(int) Math.max(opsK / tmpK, 1) :
			(int) Math.max(Math.round((double)opsK / tmpK), 1);
	}
	
	///////
	//REWRITE set task partitioner
	///

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
		else if( pn.getExecType()==ExecType.SPARK && pn.hasOnlySimpleChilds() )
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
	 * @param pn internal representation of a plan alternative for program blocks and instructions
	 * @param M ?
	 * @param flagLIX ?
	 * @param partitionedMatrices map of data partition formats
	 * @param vars local variable map
	 */
	protected void rewriteSetFusedDataPartitioningExecution(OptNode pn, double M, boolean flagLIX, HashMap<String, PartitionFormat> partitionedMatrices, LocalVariableMap vars) 
	{
		//assertions (warnings of corrupt optimizer decisions)
		if( pn.getNodeType() != NodeType.PARFOR )
			LOG.warn(getOptMode()+" OPT: Fused data partitioning and execution is only applicable for a ParFor node.");
		
		boolean apply = false;
		String partitioner = pn.getParam(ParamType.DATA_PARTITIONER);
		PDataPartitioner REMOTE_DP = PDataPartitioner.REMOTE_SPARK;
		PExecMode REMOTE_DPE = PExecMode.REMOTE_SPARK_DP;
		
		//precondition: rewrite only invoked if exec type MR 
		// (this also implies that the body is CP only)
		
		// try to merge MR data partitioning and MR exec 
		if( pn.getExecType()==ExecType.SPARK //MR/SP EXEC and CP body
			&& partitioner!=null && partitioner.equals(REMOTE_DP.toString()) //MR/SP partitioning
			&& partitionedMatrices.size()==1 ) //only one partitioned matrix
		{
			ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
				.getAbstractPlanMapping().getMappedProg(pn.getID())[1];
			
			//partitioned matrix
			String moVarname = partitionedMatrices.keySet().iterator().next();
			PartitionFormat moDpf = partitionedMatrices.get(moVarname);
			MatrixObject mo = (MatrixObject)vars.get(moVarname);
			
			if( rIsAccessByIterationVariable(pn, moVarname, pfpb.getIterVar()) &&
			   ((moDpf==PartitionFormat.ROW_WISE && mo.getNumRows()==_N ) ||
				(moDpf==PartitionFormat.COLUMN_WISE && mo.getNumColumns()==_N) ||
				(moDpf._dpf==PDataPartitionFormat.ROW_BLOCK_WISE_N && mo.getNumRows()<=_N*moDpf._N)||
				(moDpf._dpf==PDataPartitionFormat.COLUMN_BLOCK_WISE_N && mo.getNumColumns()<=_N*moDpf._N)) )
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

	protected boolean rIsAccessByIterationVariable( OptNode n, String varName, String iterVarname ) 
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
			PartitionFormat dpf = PartitionFormat.valueOf(n.getParam(ParamType.DATA_PARTITION_FORMAT));
			Hop h = OptTreeConverter.getAbstractPlanMapping().getMappedHop(n.getID());
			String inMatrix = h.getInput().get(0).getName();
			String indexAccess = null;
			switch( dpf._dpf )
			{
				case ROW_WISE: //input 1 and 2 eq
					if( h.getInput().get(1) instanceof DataOp )
						indexAccess = h.getInput().get(1).getName();
					break;
				case ROW_BLOCK_WISE_N: //input 1 and 2 have same slope and var
					indexAccess = rGetVarFromExpression(h.getInput().get(1));
					break;
				case COLUMN_WISE: //input 3 and 4 eq
					if( h.getInput().get(3) instanceof DataOp )
						indexAccess = h.getInput().get(3).getName();
					break;
				case COLUMN_BLOCK_WISE_N: //input 3 and 4 have same slope and var
					indexAccess = rGetVarFromExpression(h.getInput().get(3));
					break;
					
				default:
					//do nothing
			}
			
			ret &= (   (inMatrix!=null && inMatrix.equals(varName)) 
				    && (indexAccess!=null && indexAccess.equals(iterVarname)));
		}
		
		return ret;
	}
	
	private static String rGetVarFromExpression(Hop current) {
		String var = null;
		for( Hop c : current.getInput() ) {
			var = rGetVarFromExpression(c);
			if( var != null )
				return var;
		}
		return (current instanceof DataOp) ?
			current.getName() : null;
	}
	
	///////
	//REWRITE transpose sparse vector operations
	///

	protected boolean rIsTransposeSafePartition( OptNode n, String varName ) 
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

	protected void rewriteSetInPlaceResultIndexing(OptNode pn, CostEstimator cost, LocalVariableMap vars, HashSet<ResultVar> inPlaceResultVars, ExecutionContext ec) 
	{
		//assertions (warnings of corrupt optimizer decisions)
		if( pn.getNodeType() != NodeType.PARFOR )
			LOG.warn(getOptMode()+" OPT: Set in-place result update is only applicable for a ParFor node.");
		
		boolean apply = false;
		
		ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
			.getAbstractPlanMapping().getMappedProg(pn.getID())[1];
		
		//note currently we decide for all result vars jointly, i.e.,
		//only if all fit pinned in remaining budget, we apply this rewrite.
		ArrayList<ResultVar> retVars = pfpb.getResultVariables();
		
		//basic correctness constraint
		double totalMem = -1;
		if( rHasOnlyInPlaceSafeLeftIndexing(pn, retVars) )
		{
			//compute total sum of pinned result variable memory 
			double sum = computeTotalSizeResultVariables(retVars, vars, pfpb.getDegreeOfParallelism());
		
			//compute memory estimate without result indexing, and total sum per worker
			double M = cost.getEstimate(TestMeasure.MEMORY_USAGE, pn, true, retVars.stream()
				.map(var -> var._name).collect(Collectors.toList()), ExcludeType.RESULT_LIX);
			totalMem = M + sum;
			
			//result update in-place for MR/Spark (w/ remote memory constraint)
			if( (pfpb.getExecMode() == PExecMode.REMOTE_SPARK_DP || pfpb.getExecMode() == PExecMode.REMOTE_SPARK) 
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
		
		//modify result variable meta data, if rewrite applied
		if( apply ) 
		{
			//add result vars to result and set state
			//will be serialized and transfered via symbol table 
			for( ResultVar var : retVars ){
				Data dat = vars.get(var._name);
				if( dat instanceof MatrixObject )
					((MatrixObject)dat).setUpdateType(UpdateType.INPLACE_PINNED);
			}
			inPlaceResultVars.addAll(retVars);
		}
		
		LOG.debug(getOptMode()+" OPT: rewrite 'set in-place result indexing' - result="+
			apply+" ("+Arrays.toString(inPlaceResultVars.toArray(new ResultVar[0]))+", M="+toMB(totalMem)+")" );
	}
	
	protected boolean rHasOnlyInPlaceSafeLeftIndexing( OptNode n, ArrayList<ResultVar> retVars ) 
	{
		boolean ret = true;
		if( !n.isLeaf() ) {
			for( OptNode cn : n.getChilds() )
				ret &= rHasOnlyInPlaceSafeLeftIndexing( cn, retVars );
		}
		else if( n.getNodeType()== NodeType.HOP) {
			Hop h = OptTreeConverter.getAbstractPlanMapping().getMappedHop(n.getID());
			if( h instanceof LeftIndexingOp && ResultVar.contains(retVars, h.getInput().get(0).getName() )
				&& !retVars.stream().anyMatch(rvar -> rvar._isAccum) )
				ret &= (h.getParent().size()==1 
					&& h.getParent().get(0).getName().equals(h.getInput().get(0).getName()));
		}
		return ret;
	}

	private static double computeTotalSizeResultVariables(ArrayList<ResultVar> retVars, LocalVariableMap vars, int k) {
		double sum = 1;
		for( ResultVar var : retVars ) {
			Data dat = vars.get(var._name);
			if( !(dat instanceof MatrixObject) )
				continue;
			MatrixObject mo = (MatrixObject)dat;
			// every worker will consume memory for at most (max_nnz/k + in_nnz)
			sum += (OptimizerUtils.estimateSizeExactSparsity(mo.getNumRows(), 
				mo.getNumColumns(), Math.min((1.0/k)+mo.getSparsity(), 1.0)));
		}
		return sum;
	}
	
	///////
	//REWRITE disable CP caching  
	///

	protected double rComputeSumMemoryIntermediates( OptNode n, HashSet<ResultVar> inplaceResultVars )
	{
		double sum = 0;
		
		if( !n.isLeaf() ) {
			for( OptNode cn : n.getChilds() )
				sum += rComputeSumMemoryIntermediates( cn, inplaceResultVars );
		}
		else if( n.getNodeType()== NodeType.HOP )
		{
			Hop h = OptTreeConverter.getAbstractPlanMapping().getMappedHop(n.getID());
			if( n.getParam(ParamType.OPSTRING).equals(IndexingOp.OPSTRING)
				&& n.getParam(ParamType.DATA_PARTITION_FORMAT) != null ) {
				//set during partitioning rewrite
				sum += h.getMemEstimate();
			}
			else {
				//base intermediate (worst-case w/ materialized intermediates)
				sum +=   h.getOutputMemEstimate()
					   + h.getIntermediateMemEstimate(); 
				//inputs not represented in the planopttree (worst-case no CSE)
				if( h.getInput() != null )
					for( Hop cn : h.getInput() )
						if( cn instanceof DataOp && ((DataOp)cn).isRead()  //read data
							&& !ResultVar.contains(inplaceResultVars, cn.getName())) //except in-place result vars
							sum += cn.getMemEstimate();
			}
		}
		
		return sum;
	}
	
	///////
	//REWRITE enable runtime piggybacking
	///

//	protected void rewriteEnableRuntimePiggybacking( OptNode n, LocalVariableMap vars, HashMap<String, PartitionFormat> partitionedMatrices ) 
//	{
//		ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
//				.getAbstractPlanMapping().getMappedProg(n.getID())[1];
//		HashSet<String> sharedVars = new HashSet<>();
//		boolean apply = false; 
//		
//		//enable runtime piggybacking if MR jobs on shared read-only data set
//		if( OptimizerUtils.ALLOW_RUNTIME_PIGGYBACKING )
//		{
//			//apply runtime piggybacking if hop in mr and shared input variable 
//			//(any input variabled which is not partitioned and is read only and applies)
//			apply = rHasSharedMRInput(n, vars.keySet(), partitionedMatrices.keySet(), sharedVars)
//					&& n.getTotalK() > 1; //apply only if degree of parallelism > 1
//		}
//		
//		if( apply )
//			pfpb.setRuntimePiggybacking(apply);
//		
//		_numEvaluatedPlans++;
//		LOG.debug(getOptMode()+" OPT: rewrite 'enable runtime piggybacking' - result="
//			+apply+" ("+Arrays.toString(sharedVars.toArray())+")" );
//	}
//
//	protected boolean rHasSharedMRInput( OptNode n, Set<String> inputVars, Set<String> partitionedVars, HashSet<String> sharedVars ) 
//	{
//		boolean ret = false;
//		
//		if( !n.isLeaf() )
//		{
//			for( OptNode cn : n.getChilds() )
//				ret |= rHasSharedMRInput( cn, inputVars, partitionedVars, sharedVars );
//		}
//		else if( n.getNodeType()== NodeType.HOP && n.getExecType()==ExecType.MR )
//		{
//			Hop h = OptTreeConverter.getAbstractPlanMapping().getMappedHop(n.getID());
//			for( Hop ch : h.getInput() )
//			{
//				//note: we replaxed the contraint of non-partitioned inputs for additional 
//				//latecy hiding and scan sharing of partitions which are read multiple times
//				
//				if(    ch instanceof DataOp && ch.getDataType() == DataType.MATRIX
//					&& inputVars.contains(ch.getName()) )
//				{
//					ret = true;
//					sharedVars.add(ch.getName());
//				}
//				else if( HopRewriteUtils.isTransposeOperation(ch)
//					&& ch.getInput().get(0) instanceof DataOp && ch.getInput().get(0).getDataType() == DataType.MATRIX
//					&& inputVars.contains(ch.getInput().get(0).getName()) )
//				{
//					ret = true;
//					sharedVars.add(ch.getInput().get(0).getName());
//				}
//			}
//		}
//
//		return ret;
//	}


	///////
	//REWRITE inject spark loop checkpointing
	///

	protected void rewriteInjectSparkLoopCheckpointing(OptNode n) 
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
			rewriter.rRewriteStatementBlockHopDAGs( pfsb, state );
			fs.setBody(rewriter.rRewriteStatementBlocks(fs.getBody(), state, true));
			
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

	protected void rewriteInjectSparkRepartition(OptNode n, LocalVariableMap vars) 
	{
		//get program blocks of root parfor
		Object[] progobj = OptTreeConverter.getAbstractPlanMapping().getMappedProg(n.getID());
		ParForStatementBlock pfsb = (ParForStatementBlock)progobj[0];
		ParForProgramBlock pfpb = (ParForProgramBlock)progobj[1];
		ArrayList<String> ret = new ArrayList<>();
		
		if(    OptimizerUtils.isSparkExecutionMode() //spark exec mode
			&& n.getExecType() == ExecType.CP		 //local parfor 
			&& _N > 1                            )   //at least 2 iterations
		{
			//collect candidates from zipmm spark instructions
			HashSet<String> cand = new HashSet<>();
			rCollectZipmmPartitioningCandidates(n, cand);
			
			//prune updated candidates
			HashSet<String> probe = new HashSet<>(pfsb.getReadOnlyParentMatrixVars());
			for( String var : cand )
				if( probe.contains( var ) )
					ret.add( var );
			
			//prune small candidates
			ArrayList<String> tmp = new ArrayList<>(ret);
			ret.clear();
			for( String var : tmp )
				if( vars.get(var) instanceof MatrixObject ) {
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
		LOG.debug(getOptMode()+" OPT: rewrite 'inject spark input repartition' - result="
			+ret.size()+" ("+Arrays.toString(ret.toArray())+")" );
	}

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
					else if( HopRewriteUtils.isTransposeOperation(in)
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

	protected void rewriteSetSparkEagerRDDCaching(OptNode n, LocalVariableMap vars) 
	{
		//get program blocks of root parfor
		Object[] progobj = OptTreeConverter.getAbstractPlanMapping().getMappedProg(n.getID());
		ParForStatementBlock pfsb = (ParForStatementBlock)progobj[0];
		ParForProgramBlock pfpb = (ParForProgramBlock)progobj[1];
		
		ArrayList<String> ret = new ArrayList<>();
		
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
					DataCharacteristics mc = mo.getDataCharacteristics();
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
		LOG.debug(getOptMode()+" OPT: rewrite 'set spark eager rdd caching' - result="
			+ret.size()+" ("+Arrays.toString(ret.toArray())+")" );
	}
	
	///////
	//REWRITE remove compare matrix (for result merge, needs to be invoked before setting result merge)
	///

	protected void rewriteRemoveUnnecessaryCompareMatrix( OptNode n, ExecutionContext ec ) 
	{
		ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
			.getAbstractPlanMapping().getMappedProg(n.getID())[1];

		ArrayList<ResultVar> cleanedVars = new ArrayList<>();
		ArrayList<ResultVar> resultVars = pfpb.getResultVariables();
		String itervar = pfpb.getIterVar();
		
		for( ResultVar rvar : resultVars ) {
			Data dat = ec.getVariable(rvar._name);
			if( dat instanceof MatrixObject && ((MatrixObject)dat).getNnz()!=0     //subject to result merge with compare
				&& n.hasOnlySimpleChilds()                                         //guaranteed no conditional indexing	
				&& rContainsResultFullReplace(n, rvar._name, itervar, (MatrixObject)dat) //guaranteed full matrix replace 
				&& !rIsReadInRightIndexing(n, rvar._name)                          //never read variable in loop body
				&& ((MatrixObject)dat).getNumRows()<=Integer.MAX_VALUE
				&& ((MatrixObject)dat).getNumColumns()<=Integer.MAX_VALUE )
			{
				//replace existing matrix object with empty matrix
				MatrixObject mo = (MatrixObject)dat;
				ec.cleanupCacheableData(mo);
				ec.setMatrixOutput(rvar._name, new MatrixBlock((int)mo.getNumRows(), (int)mo.getNumColumns(),false));
				
				//keep track of cleaned result variables
				cleanedVars.add(rvar);
			}
		}

		_numEvaluatedPlans++;
		LOG.debug(getOptMode()+" OPT: rewrite 'remove unnecessary compare matrix' - result="+(!cleanedVars.isEmpty())
			+" ("+ProgramConverter.serializeResultVariables(cleanedVars)+")" );
	}

	protected boolean rContainsResultFullReplace( OptNode n, String resultVar, String iterVarname, MatrixObject mo ) {
		boolean ret = false;
		//process hop node
		if( n.getNodeType()==NodeType.HOP )
			ret |= isResultFullReplace(n, resultVar, iterVarname, mo);
		//process childs recursively
		if( !n.isLeaf() )
			for( OptNode c : n.getChilds() ) 
				ret |= rContainsResultFullReplace(c, resultVar, iterVarname, mo);
		return ret;
	}

	protected boolean isResultFullReplace( OptNode n, String resultVar, String iterVarname, MatrixObject mo ) 
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

	protected void rewriteSetResultMerge( OptNode n, LocalVariableMap vars, boolean inLocal ) {
		ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
			.getAbstractPlanMapping().getMappedProg(n.getID())[1];
		
		PResultMerge REMOTE = PResultMerge.REMOTE_SPARK;
		PResultMerge ret = null;
		
		//investigate details of current parfor node
		boolean flagRemoteParFOR = (n.getExecType() == getRemoteExecType());
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

	protected boolean determineFlagCellFormatWoCompare( ArrayList<ResultVar> resultVars, LocalVariableMap vars  )
	{
		boolean ret = true;
		
		for( ResultVar rVar : resultVars )
		{
			Data dat = vars.get(rVar._name);
			if( dat == null || !(dat instanceof MatrixObject) )
			{
				ret = false; 
				break;
			}
			else
			{
				MatrixObject mo = (MatrixObject)dat;
				MetaDataFormat meta = (MetaDataFormat) mo.getMetaData();
				OutputInfo oi = meta.getOutputInfo();
				long nnz = meta.getDataCharacteristics().getNonZeros();
				
				if( oi == OutputInfo.BinaryBlockOutputInfo || nnz != 0 ) {
					ret = false; 
					break;
				}
			}
		}
		
		return ret;
	}

	protected boolean hasResultMRLeftIndexing( OptNode n, ArrayList<ResultVar> resultVars, LocalVariableMap vars, boolean checkSize ) 
	{
		boolean ret = false;
		
		if( n.isLeaf() )
		{
			String opName = n.getParam(ParamType.OPSTRING);
			//check opstring and exec type
			if( opName != null && opName.equals(LeftIndexingOp.OPSTRING) 
				&& n.getExecType() == getRemoteExecType() )
			{
				LeftIndexingOp hop = (LeftIndexingOp) OptTreeConverter.getAbstractPlanMapping().getMappedHop(n.getID());
				//check agains set of varname
				String varName = hop.getInput().get(0).getName();
				if( ResultVar.contains(resultVars, varName) )
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
	 * @param pn internal representation of a plan alternative for program blocks and instructions
	 * @param resultVars list of result variables
	 * @param vars local variable map
	 * @param checkSize ?
	 * @return true if result sizes larger than local memory budget
	 */
	protected boolean hasLargeTotalResults( OptNode pn, ArrayList<ResultVar> resultVars, LocalVariableMap vars, boolean checkSize ) 
	{
		double totalSize = 0;
		
		//get num tasks according to task partitioning 
		PTaskPartitioner tp = PTaskPartitioner.valueOf(pn.getParam(ParamType.TASK_PARTITIONER));
		int k = pn.getK();
		long W = estimateNumTasks(tp, _N, k); 
		
		for( ResultVar var : resultVars )
		{
			//Potential unknowns: for local result var of child parfor (but we're only interested in top level)
			//Potential scalars: for disabled dependency analysis and unbounded scoping
			Data dat = vars.get( var._name );
			if( dat != null && dat instanceof MatrixObject ) 
			{
				MatrixObject mo = (MatrixObject) dat;
				
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

	protected boolean hasOnlyInMemoryResults( OptNode n, ArrayList<ResultVar> resultVars, LocalVariableMap vars, boolean inLocal ) 
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
				if( ResultVar.contains(resultVars, varName) && vars.keySet().contains(varName) ) {
					Data dat = vars.get(hop.getInput().get(0).getName());
					//dims of result vars must be known at this point in time
					if( dat instanceof MatrixObject ) {
						MatrixObject mo = (MatrixObject) dat;
						long rows = mo.getNumRows();
						long cols = mo.getNumColumns();
						double memBudget = inLocal ? OptimizerUtils.getLocalMemBudget() : 
							                         OptimizerUtils.getRemoteMemBudgetMap();
						ret &= isInMemoryResultMerge(rows, cols, memBudget);
					}
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

	protected void rInvokeSetResultMerge( Collection<OptNode> nodes, LocalVariableMap vars, boolean inLocal) {
		for( OptNode n : nodes )
			if( n.getNodeType() == NodeType.PARFOR )
			{
				rewriteSetResultMerge(n, vars, inLocal);
				if( n.getExecType()==getRemoteExecType() )
					inLocal = false;
			}
			else if( n.getChilds()!=null )  
				rInvokeSetResultMerge(n.getChilds(), vars, inLocal);
	}

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

	protected void rewriteRemoveRecursiveParFor(OptNode n, LocalVariableMap vars) 
	{
		int count = 0; //num removed parfor
		
		//find recursive parfor
		HashSet<ParForProgramBlock> recPBs = new HashSet<>();
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

	protected void rFindAndUnfoldRecursiveFunction( OptNode n, ParForProgramBlock parfor, HashSet<ParForProgramBlock> recPBs, LocalVariableMap vars )
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
				HashSet<String> memo = new HashSet<>();
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

	protected void rReplaceFunctionNames( OptNode n, String oldName, String newName ) 
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
				BasicProgramBlock pb = (BasicProgramBlock) OptTreeConverter
					.getAbstractPlanMapping().getMappedProg(parentID)[1];
				
				ArrayList<Instruction> instArr = pb.getInstructions();
				for( int i=0; i<instArr.size(); i++ ) {
					Instruction inst = instArr.get(i);
					if( inst instanceof FunctionCallCPInstruction ) {
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

	protected int removeRecursiveParFor( OptNode n, HashSet<ParForProgramBlock> recPBs ) 
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

	protected void rewriteRemoveUnnecessaryParFor(OptNode n) {
		int count = removeUnnecessaryParFor( n );
		_numEvaluatedPlans++;
		LOG.debug(getOptMode()+" OPT: rewrite 'remove unnecessary parfor' - result="+count );
	}

	protected int removeUnnecessaryParFor( OptNode n ) {
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
	
	public static String toMB( double inB ) {
		return OptimizerUtils.toMB(inB) + "MB";
	}
}
