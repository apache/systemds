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

package org.tugraz.sysds.runtime.controlprogram.parfor.opt;

import org.tugraz.sysds.hops.cost.CostEstimationWrapper;
import org.tugraz.sysds.lops.LopProperties.ExecType;
import org.tugraz.sysds.runtime.controlprogram.LocalVariableMap;
import org.tugraz.sysds.runtime.controlprogram.ProgramBlock;
import org.tugraz.sysds.runtime.controlprogram.context.ExecutionContext;
import org.tugraz.sysds.runtime.controlprogram.context.ExecutionContextFactory;

/**
 * Cost estimator for runtime programs. Previously this estimator used an offline created
 * performance profile. Since SystemDS 1.0, this estimator uses a time-based cost model
 * that relies on floating operations and I/O, which does not require explicit profiling.
 * 
 */
public class CostEstimatorRuntime extends CostEstimator
{	
	private final CostEstimatorHops _costMem;
	private final OptTreePlanMappingAbstract _map;
	private final ExecutionContext _ec;
	
	public CostEstimatorRuntime(OptTreePlanMappingAbstract map, LocalVariableMap vars ) {
		_costMem = new CostEstimatorHops(map);
		_map = map;
		
		//construct execution context as wrapper to hand over
		//deep copied symbol table to cost estimator
		_ec = ExecutionContextFactory.createContext();
		_ec.setVariables(vars);
	}
	
	@Override
	public double getLeafNodeEstimate( TestMeasure measure, OptNode node ) {
		//use CostEstimatorHops to get the memory estimate
		if( measure == TestMeasure.MEMORY_USAGE )
			return _costMem.getLeafNodeEstimate(measure, node);
		
		//redirect to exec-type-specific estimate
		return getLeafNodeEstimate(measure, node, node.isCPOnly() ? ExecType.CP : ExecType.SPARK);
	}
	
	@Override
	public double getLeafNodeEstimate( TestMeasure measure, OptNode node, ExecType et ) {
		//use CostEstimatorHops to get the memory estimate
		if( measure == TestMeasure.MEMORY_USAGE )
			return _costMem.getLeafNodeEstimate(measure, node, et);
		
		//use static cost estimator based on floating point operations
		//(currently only called for entire parfor program in order to
		//decide for LOCAL vs REMOTE parfor execution)
		double ret = DEFAULT_TIME_ESTIMATE;
		boolean isCP = (et == ExecType.CP || et == null);
		if( !node.isLeaf() && isCP ) {
			ProgramBlock pb = (ProgramBlock)_map.getMappedProg(node.getID())[1];
			ret = CostEstimationWrapper.getTimeEstimate(pb, _ec, true);
		}
		return ret;
	}
}
