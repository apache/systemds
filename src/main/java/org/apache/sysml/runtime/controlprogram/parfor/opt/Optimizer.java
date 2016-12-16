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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysml.parser.ParForStatementBlock;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.controlprogram.ParForProgramBlock;
import org.apache.sysml.runtime.controlprogram.ParForProgramBlock.POptMode;
import org.apache.sysml.runtime.controlprogram.context.ExecutionContext;


/**
 * Generic optimizer super class that defines the interface of all implemented optimizers.
 * Furthermore it implements basic primitives, used by all optimizers such as the enumeration
 * of plan alternatives and specific rewrites.
 * 
 * Optimization objective: \phi: \min T(prog) | k \leq ck \wedge m(prog) \leq cm 
 *                                      with T(p)=max_(1\leq i\leq k)(T(prog_i). 
 * 
 */
public abstract class Optimizer 
{

	
	protected static final Log LOG = LogFactory.getLog(Optimizer.class.getName());
	
	protected long _numTotalPlans     = -1;
	protected long _numEvaluatedPlans = -1;
	
	public enum PlanInputType {
		ABSTRACT_PLAN,
		RUNTIME_PLAN
	}
	
	public enum CostModelType {
		STATIC_MEM_METRIC,
		RUNTIME_METRICS
	}
	
	protected Optimizer()
	{
		_numTotalPlans     = 0;
		_numEvaluatedPlans = 0;
	}

	/**
	 * Optimize
	 * 
	 * @param sb parfor statement block
	 * @param pb parfor program block
	 * @param plan  complete plan of a top-level parfor
	 * @param est cost estimator
	 * @param ec execution context
	 * @return true if plan changed, false otherwise
	 * @throws DMLRuntimeException if DMLRuntimeException occurs
	 */
	public abstract boolean optimize(ParForStatementBlock sb, ParForProgramBlock pb, OptTree plan, CostEstimator est, ExecutionContext ec) 
		throws DMLRuntimeException;	

	public abstract PlanInputType getPlanInputType();

	public abstract CostModelType getCostModelType();

	public abstract POptMode getOptMode();
	
	
	///////
	//methods for evaluating the overall properties and costing  

	public long getNumTotalPlans()
	{
		return _numTotalPlans;
	}

	public long getNumEvaluatedPlans()
	{
		return _numEvaluatedPlans;
	}

}
