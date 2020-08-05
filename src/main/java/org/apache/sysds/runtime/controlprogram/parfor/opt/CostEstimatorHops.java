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

package org.apache.sysds.runtime.controlprogram.parfor.opt;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.Hop;
import org.apache.sysds.hops.LeftIndexingOp;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.parfor.opt.OptNode.NodeType;
import org.apache.sysds.runtime.controlprogram.parfor.opt.Optimizer.CostModelType;

public class CostEstimatorHops extends CostEstimator
{
	
	public static final double DEFAULT_MEM_SP = 20*1024*1024;
	
	private OptTreePlanMappingAbstract _map = null;
	
	public CostEstimatorHops( OptTreePlanMappingAbstract map ) {
		_map = map;
	}

	@Override
	public double getLeafNodeEstimate(TestMeasure measure, OptNode node)
	{
		if( node.getNodeType() != NodeType.HOP )
			return 0; //generic optnode but no childs (e.g., PB for rmvar inst)
		
		if( measure != TestMeasure.MEMORY_USAGE )
			throw new DMLRuntimeException( "Testmeasure "+measure+" not supported by cost model "+CostModelType.STATIC_MEM_METRIC+"." );
		
		//core mem estimation (use hops estimate)
		Hop h = _map.getMappedHop( node.getID() );
		double value = h.getMemEstimate();
		
		//correction for disabled shared read accounting
		value = (_exclVars!=null && _exclType==ExcludeType.SHARED_READ) ?
			h.getInputOutputSize(_exclVars) : value;
		
		//handle specific cases 
		double DEFAULT_MEM_REMOTE = OptimizerUtils.isSparkExecutionMode() ? DEFAULT_MEM_SP : 0;
		boolean forcedExec =  DMLScript.getGlobalExecMode() == ExecMode.SINGLE_NODE || h.getForcedExecType()!=null;
		
		if( value >= DEFAULT_MEM_REMOTE )
		{
			//check for CP estimate but Spark type (include broadcast requirements)
			if( h.getExecType()==ExecType.SPARK ) {
				value = DEFAULT_MEM_REMOTE + h.getSpBroadcastSize();
			}
			//check for invalid cp memory estimate
			else if ( h.getExecType()==ExecType.CP && value >= OptimizerUtils.getLocalMemBudget() ) {
				if( !forcedExec )
					LOG.warn("Memory estimate larger than budget but CP exec type (op="+h.getOpString()+", name="+h.getName()+", memest="+h.getMemEstimate()+").");
				value = DEFAULT_MEM_REMOTE;
			}
			//check for non-existing exec type
			else if ( h.getExecType()==null) {
				//note: if exec type is 'null' lops have never been created (e.g., r(T) for tsmm),
				//in that case, we do not need to raise a warning 
				value = DEFAULT_MEM_REMOTE;
			}
		}
		
		//check for forced runtime platform
		if( h.getForcedExecType()==ExecType.SPARK) {
			value = DEFAULT_MEM_REMOTE;
		}
		
		if( value <= 0 && !forcedExec ) { //no mem estimate
			LOG.warn("Cannot get memory estimate for hop (op="+h.getOpString()+", name="+h.getName()+", memest="+h.getMemEstimate()+").");
			value = CostEstimator.DEFAULT_MEM_ESTIMATE_CP;
		}
		
		//correction for disabled result indexing
		value = (_exclVars!=null && _exclType==ExcludeType.RESULT_LIX 
			&& h instanceof LeftIndexingOp && _exclVars.contains(h.getName())) ? 0 : value;
		
		if( LOG.isTraceEnabled() ) {
			LOG.trace("Memory estimate "+h.getName()+", "+h.getOpString()
				+"("+node.getExecType()+")"+"="+OptimizerRuleBased.toMB(value));
		}
		
		return value;
	}

	@Override
	public double getLeafNodeEstimate(TestMeasure measure, OptNode node, ExecType et)
	{
		if( node.getNodeType() != NodeType.HOP )
			return 0; //generic optnode but no childs (e.g., PB for rmvar inst)
		
		if( measure != TestMeasure.MEMORY_USAGE )
			throw new DMLRuntimeException( "Testmeasure "+measure+" not supported by cost model "+CostModelType.STATIC_MEM_METRIC+"." );
		
		//core mem estimation (use hops estimate)
		Hop h = _map.getMappedHop( node.getID() );
		double value = h.getMemEstimate();
		if( et != ExecType.CP ) //MR, null
			value = DEFAULT_MEM_SP;
		if( value <= 0 ) //no mem estimate
			value = CostEstimator.DEFAULT_MEM_ESTIMATE_CP;
		
		if( LOG.isTraceEnabled() ) {
			LOG.trace("Memory estimate (forced exec type) "+h.getName()+", "
				+h.getOpString()+"("+node.getExecType()+")"+"="+OptimizerRuleBased.toMB(value));
		}
		
		return value;
	}
}
