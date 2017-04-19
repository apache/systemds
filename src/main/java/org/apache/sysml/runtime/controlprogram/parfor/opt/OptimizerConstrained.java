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

import java.util.HashMap;
import java.util.HashSet;

import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.lops.LopProperties;
import org.apache.sysml.parser.ParForStatementBlock;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.LocalVariableMap;
import org.apache.sysml.runtime.controlprogram.ParForProgramBlock;
import org.apache.sysml.runtime.controlprogram.ParForProgramBlock.PDataPartitionFormat;
import org.apache.sysml.runtime.controlprogram.ParForProgramBlock.PDataPartitioner;
import org.apache.sysml.runtime.controlprogram.ParForProgramBlock.PExecMode;
import org.apache.sysml.runtime.controlprogram.ParForProgramBlock.POptMode;
import org.apache.sysml.runtime.controlprogram.ParForProgramBlock.PResultMerge;
import org.apache.sysml.runtime.controlprogram.ParForProgramBlock.PTaskPartitioner;
import org.apache.sysml.runtime.controlprogram.ParForProgramBlock.PartitionFormat;
import org.apache.sysml.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysml.runtime.controlprogram.parfor.opt.CostEstimator.TestMeasure;
import org.apache.sysml.runtime.controlprogram.parfor.opt.OptNode.ExecType;
import org.apache.sysml.runtime.controlprogram.parfor.opt.OptNode.ParamType;

/**
 * Rule-Based ParFor Optimizer (time: O(n)):
 *
 * Applied rule-based rewrites:
 * - see base class.
 *
 *
 * Checked constraints:
 * - 1) rewrite set data partitioner (incl. recompile RIX)
 * - 4) rewrite set execution strategy
 * - 9) rewrite set degree of parallelism
 * - 10) rewrite set task partitioner
 * - 11) rewrite set result merge 		 		
 *
 * TODO generalize for nested parfor (currently only awareness of top-level constraints, if present leave child as they are)
 *
 */
public class OptimizerConstrained extends OptimizerRuleBased
{

	@Override
	public POptMode getOptMode() {
		return POptMode.CONSTRAINED;
	}


	/**
	 * Main optimization procedure.
	 *
	 * Transformation-based heuristic (rule-based) optimization
	 * (no use of sb, direct change of pb).
	 */
	@Override
	public boolean optimize(ParForStatementBlock sb, ParForProgramBlock pb, OptTree plan, CostEstimator est, ExecutionContext ec)
		throws DMLRuntimeException
	{
		LOG.debug("--- "+getOptMode()+" OPTIMIZER -------");

		OptNode pn = plan.getRoot();

		//early abort for empty parfor body 
		if( pn.isLeaf() )
			return true;

		//ANALYZE infrastructure properties
		super.analyzeProblemAndInfrastructure( pn );

		_cost = est;

		//debug and warnings output
		LOG.debug(getOptMode()+" OPT: Optimize with local_max_mem="+toMB(_lm)+" and remote_max_mem="+toMB(_rm)+")." );
		if( _rnk<=0 || _rk<=0 )
			LOG.warn(getOptMode()+" OPT: Optimize for inactive cluster (num_nodes="+_rnk+", num_map_slots="+_rk+")." );

		//ESTIMATE memory consumption 
		ExecType oldET = pn.getExecType();
		int oldK = pn.getK();
		pn.setSerialParFor(); //for basic mem consumption 
		double M0a = _cost.getEstimate(TestMeasure.MEMORY_USAGE, pn);
		pn.setExecType(oldET);
		pn.setK(oldK);
		LOG.debug(getOptMode()+" OPT: estimated mem (serial exec) M="+toMB(M0a) );

		//OPTIMIZE PARFOR PLAN

		// rewrite 1: data partitioning (incl. log. recompile RIX)
		HashMap<String, PartitionFormat> partitionedMatrices = new HashMap<String, PartitionFormat>();
		rewriteSetDataPartitioner( pn, ec.getVariables(), partitionedMatrices, OptimizerUtils.getLocalMemBudget() );
		double M0b = _cost.getEstimate(TestMeasure.MEMORY_USAGE, pn); //reestimate

		// rewrite 2: remove unnecessary compare matrix
		rewriteRemoveUnnecessaryCompareMatrix(pn, ec);

		// rewrite 3: rewrite result partitioning (incl. log/phy recompile LIX) 
		boolean flagLIX = super.rewriteSetResultPartitioning( pn, M0b, ec.getVariables() );
		double M1 = _cost.getEstimate(TestMeasure.MEMORY_USAGE, pn); //reestimate 
		LOG.debug(getOptMode()+" OPT: estimated new mem (serial exec) M="+toMB(M1) );
		
		//determine memory consumption for what-if: all-cp or partitioned
		double M2 = _cost.getEstimate(TestMeasure.MEMORY_USAGE, pn, LopProperties.ExecType.CP);
		LOG.debug(getOptMode()+" OPT: estimated new mem (serial exec, all CP) M="+toMB(M2) );
		double M3 = _cost.getEstimate(TestMeasure.MEMORY_USAGE, pn, true);
		LOG.debug(getOptMode()+" OPT: estimated new mem (cond partitioning) M="+toMB(M3) );
		
		// rewrite 4: execution strategy
		PExecMode tmpmode = getPExecMode(pn); //keep old
		boolean flagRecompMR = rewriteSetExecutionStategy( pn, M0a, M1, M2, M3, flagLIX );

		//exec-type-specific rewrites
		if( pn.getExecType() == getRemoteExecType() )
		{
			if( M1 > _rm && M3 <= _rm  ) {
				// rewrite 1: data partitioning (apply conditional partitioning)
				rewriteSetDataPartitioner( pn, ec.getVariables(), partitionedMatrices, M3 );
				M1 = _cost.getEstimate(TestMeasure.MEMORY_USAGE, pn); //reestimate 		
			}
			
			if( flagRecompMR ){
				//rewrite 5: set operations exec type
				rewriteSetOperationsExecType( pn, flagRecompMR );
				M1 = _cost.getEstimate(TestMeasure.MEMORY_USAGE, pn); //reestimate 		
			}

			// rewrite 6: data colocation
			super.rewriteDataColocation( pn, ec.getVariables() );

			// rewrite 7: rewrite set partition replication factor
			super.rewriteSetPartitionReplicationFactor( pn, partitionedMatrices, ec.getVariables() );

			// rewrite 8: rewrite set partition replication factor
			super.rewriteSetExportReplicationFactor( pn, ec.getVariables() );

			// rewrite 9: nested parallelism (incl exec types)	
			boolean flagNested = super.rewriteNestedParallelism( pn, M1, flagLIX );

			// rewrite 10: determine parallelism
			rewriteSetDegreeOfParallelism( pn, M1, flagNested );

			// rewrite 11: task partitioning 
			rewriteSetTaskPartitioner( pn, flagNested, flagLIX );

			// rewrite 12: fused data partitioning and execution
			rewriteSetFusedDataPartitioningExecution(pn, M1, flagLIX, partitionedMatrices, ec.getVariables(), tmpmode);

			// rewrite 13: transpose sparse vector operations
			super.rewriteSetTranposeSparseVectorOperations(pn, partitionedMatrices, ec.getVariables());

			//rewrite 14:
			HashSet<String> inplaceResultVars = new HashSet<String>();
			super.rewriteSetInPlaceResultIndexing(pn, M1, ec.getVariables(), inplaceResultVars, ec);

			//rewrite 15:
			super.rewriteDisableCPCaching(pn, inplaceResultVars, ec.getVariables());

		}
		else //if( pn.getExecType() == ExecType.CP )
		{
			// rewrite 10: determine parallelism
			rewriteSetDegreeOfParallelism( pn, M1, false );

			// rewrite 11: task partitioning
			rewriteSetTaskPartitioner( pn, false, false ); //flagLIX always false 

			// rewrite 14: set in-place result indexing
			HashSet<String> inplaceResultVars = new HashSet<String>();
			super.rewriteSetInPlaceResultIndexing(pn, M1, ec.getVariables(), inplaceResultVars, ec);

			if( !OptimizerUtils.isSparkExecutionMode() ) {
				// rewrite 16: runtime piggybacking
				super.rewriteEnableRuntimePiggybacking( pn, ec.getVariables(), partitionedMatrices );
			}
			else {
				//rewrite 17: checkpoint injection for parfor loop body
				super.rewriteInjectSparkLoopCheckpointing( pn );

				//rewrite 18: repartition read-only inputs for zipmm 
				super.rewriteInjectSparkRepartition( pn, ec.getVariables() );

				//rewrite 19: eager caching for checkpoint rdds
				super.rewriteSetSparkEagerRDDCaching(pn, ec.getVariables() );
			}
		}

		//rewrite 20: set result merge
		rewriteSetResultMerge( pn, ec.getVariables(), true );

		//rewrite 21: set local recompile memory budget
		super.rewriteSetRecompileMemoryBudget( pn );

		///////
		//Final rewrites for cleanup / minor improvements

		// rewrite 22: parfor (in recursive functions) to for
		super.rewriteRemoveRecursiveParFor( pn, ec.getVariables() );

		// rewrite 23: parfor (par=1) to for 
		super.rewriteRemoveUnnecessaryParFor( pn );

		//info optimization result
		_numEvaluatedPlans = 1;
		return true;
	}


	///////
	//REWRITE set data partitioner
	///

	@Override
	protected boolean rewriteSetDataPartitioner(OptNode n, LocalVariableMap vars, HashMap<String,PartitionFormat> partitionedMatrices, double thetaM)
		throws DMLRuntimeException
	{
		boolean blockwise = false;

		// constraint awareness
		if( !n.getParam(ParamType.DATA_PARTITIONER).equals(PDataPartitioner.UNSPECIFIED.toString()) )
		{
			Object[] o = OptTreeConverter.getAbstractPlanMapping().getMappedProg(n.getID());
			ParForProgramBlock pfpb = (ParForProgramBlock) o[1];
			pfpb.setDataPartitioner(PDataPartitioner.valueOf(n.getParam(ParamType.DATA_PARTITIONER)));
			LOG.debug(getOptMode()+" OPT: forced 'set data partitioner' - result="+n.getParam(ParamType.DATA_PARTITIONER) );
		}
		else
			super.rewriteSetDataPartitioner(n, vars, partitionedMatrices, thetaM);

		return blockwise;
	}


	///////
	//REWRITE set execution strategy
	///

	@Override
	protected boolean rewriteSetExecutionStategy(OptNode n, double M0, double M, double M2, double M3, boolean flagLIX)
		throws DMLRuntimeException
	{
		boolean ret = false;

		// constraint awareness
		if( n.getExecType() != null && ConfigurationManager.isParallelParFor() )
		{
			ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
                    .getAbstractPlanMapping().getMappedProg(n.getID())[1];

			PExecMode mode = PExecMode.LOCAL;

			if (n.getExecType()==ExecType.MR) {
				mode = PExecMode.REMOTE_MR;
			}
			else if (n.getExecType() == ExecType.SPARK) {
				mode = PExecMode.REMOTE_SPARK;
			}

			pfpb.setExecMode( mode );
			LOG.debug(getOptMode()+" OPT: forced 'set execution strategy' - result="+mode );
		}
		else
			ret = super.rewriteSetExecutionStategy(n, M0, M, M2, M3, flagLIX);

		return ret;
	}


	///////
	//REWRITE set degree of parallelism
	///

	@Override
	protected void rewriteSetDegreeOfParallelism(OptNode n, double M, boolean flagNested)
		throws DMLRuntimeException
	{
		// constraint awareness
		if( n.getK() > 0 && ConfigurationManager.isParallelParFor() )
		{
			ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
					.getAbstractPlanMapping().getMappedProg(n.getID())[1];
			pfpb.setDegreeOfParallelism(n.getK());
			LOG.debug(getOptMode()+" OPT: forced 'set degree of parallelism' - result=(see EXPLAIN)" );
		}
		else
			super.rewriteSetDegreeOfParallelism(n, M, flagNested);
	}


	///////
	//REWRITE set task partitioner
	///

	@Override
	protected void rewriteSetTaskPartitioner(OptNode pn, boolean flagNested, boolean flagLIX)
	{
		// constraint awareness
		if( !pn.getParam(ParamType.TASK_PARTITIONER).equals(PTaskPartitioner.UNSPECIFIED.toString()) )
		{
			ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
                    .getAbstractPlanMapping().getMappedProg(pn.getID())[1];
			pfpb.setTaskPartitioner(PTaskPartitioner.valueOf(pn.getParam(ParamType.TASK_PARTITIONER)));
			String tsExt = "";
			if( pn.getParam(ParamType.TASK_SIZE)!=null )
			{
				pfpb.setTaskSize( Integer.parseInt(pn.getParam(ParamType.TASK_SIZE)) );
				tsExt+= "," + pn.getParam(ParamType.TASK_SIZE);
			}
			LOG.debug(getOptMode()+" OPT: forced 'set task partitioner' - result="+pn.getParam(ParamType.TASK_PARTITIONER)+tsExt );
		}
		else
		{
			 if( pn.getParam(ParamType.TASK_SIZE)!=null )
				LOG.warn("Cannot force task size without forcing task partitioner.");

			super.rewriteSetTaskPartitioner(pn, flagNested, flagLIX);
		}
	}


	///////
	//REWRITE set result merge
	///

	@Override
	protected void rewriteSetResultMerge( OptNode n, LocalVariableMap vars, boolean inLocal )
		throws DMLRuntimeException
	{
		// constraint awareness
		if( !n.getParam(ParamType.RESULT_MERGE).equals(PResultMerge.UNSPECIFIED.toString()) )
		{
			ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
				    .getAbstractPlanMapping().getMappedProg(n.getID())[1];
			pfpb.setResultMerge(PResultMerge.valueOf(n.getParam(ParamType.RESULT_MERGE)));
			LOG.debug(getOptMode()+" OPT: force 'set result merge' - result="+n.getParam(ParamType.RESULT_MERGE) );
		}
		else
			super.rewriteSetResultMerge(n, vars, inLocal);
	}


	///////
	//REWRITE set fused data partitioning / execution
	///

	protected void rewriteSetFusedDataPartitioningExecution(OptNode pn, double M, boolean flagLIX, HashMap<String, PartitionFormat> partitionedMatrices, LocalVariableMap vars, PExecMode emode)
		throws DMLRuntimeException
	{
		if( emode == PExecMode.REMOTE_MR_DP || emode == PExecMode.REMOTE_SPARK_DP )
		{
			ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
	                  .getAbstractPlanMapping().getMappedProg(pn.getID())[1];

			//partitioned matrix
			if( partitionedMatrices.size()<=0 )
			{
				LOG.debug(getOptMode()+" OPT: unable to force 'set fused data partitioning and execution' - result="+false );
				return;
			}

			String moVarname = partitionedMatrices.keySet().iterator().next();
			PartitionFormat moDpf = partitionedMatrices.get(moVarname);
			MatrixObject mo = (MatrixObject)vars.get(moVarname);

			//check if access via iteration variable and sizes match
			String iterVarname = pfpb.getIterablePredicateVars()[0];

			if( rIsAccessByIterationVariable(pn, moVarname, iterVarname) &&
			   ((moDpf==PartitionFormat.ROW_WISE && mo.getNumRows()==_N ) ||
				(moDpf==PartitionFormat.COLUMN_WISE && mo.getNumColumns()==_N) ||
				(moDpf._dpf==PDataPartitionFormat.ROW_BLOCK_WISE_N && mo.getNumRows()<=_N*moDpf._N)||
				(moDpf._dpf==PDataPartitionFormat.COLUMN_BLOCK_WISE_N && mo.getNumColumns()<=_N*moDpf._N)) )
			{
				int k = (int)Math.min(_N,_rk2);

				if (emode == PExecMode.REMOTE_MR_DP) {
					pn.addParam(ParamType.DATA_PARTITIONER, "REMOTE_MR(fused)");
					pfpb.setExecMode(PExecMode.REMOTE_MR_DP); //set fused exec type
				}
				else {
					pn.addParam(ParamType.DATA_PARTITIONER, "REMOTE_SPARK(fused)");
					pfpb.setExecMode(PExecMode.REMOTE_SPARK_DP); //set fused exec type
				}

				pn.setK( k );

				pfpb.setDataPartitioner(PDataPartitioner.NONE);
				pfpb.enableColocatedPartitionedMatrix( moVarname );
				pfpb.setDegreeOfParallelism(k);
			}

			LOG.debug(getOptMode()+" OPT: force 'set fused data partitioning and execution' - result="+true );
		}
		else
			super.rewriteSetFusedDataPartitioningExecution(pn, M, flagLIX, partitionedMatrices, vars);
	}

	private PExecMode getPExecMode( OptNode pn )
	{
		ParForProgramBlock pfpb = (ParForProgramBlock) OptTreeConverter
			    .getAbstractPlanMapping().getMappedProg(pn.getID())[1];
		return pfpb.getExecMode();
	}
}
