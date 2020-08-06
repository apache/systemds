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

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.ParForProgramBlock.POptMode;
import org.apache.sysds.runtime.controlprogram.parfor.opt.CostEstimator.TestMeasure;


/**
 * Heuristic ParFor Optimizer: This optimizer extends the rule-based
 * optimizer by a time-based cost estimate for execution type decisions.
 * 
 *  
 */
public class OptimizerHeuristic extends OptimizerRuleBased {
	private static final Log LOG = LogFactory.getLog(OptimizerHeuristic.class.getName());
	public static final double EXEC_TIME_THRESHOLD = 30000; //in ms
			
	@Override
	public CostModelType getCostModelType() {
		return CostModelType.RUNTIME_METRICS;
	}
	
	@Override
	public POptMode getOptMode() {
		return POptMode.HEURISTIC;
	}

	////////////////////////////////////////////////////
	// Overwritten rewrites (see rulebased optimizer) //
	////////////////////////////////////////////////////
	
	/**
	 * Used as one condition in rewriteSetExecutionStategy in order to decide if
	 * MR execution makes sense if all the other constraints are given. 
	 */
	@Override
	protected boolean isLargeProblem(OptNode pn, double M)
	{
		boolean ret = false;
		
		try 
		{
			double T = _cost.getEstimate(TestMeasure.EXEC_TIME, pn);
			ret = (T >= EXEC_TIME_THRESHOLD) && (M > PROB_SIZE_THRESHOLD_MB );
		} 
		catch (DMLRuntimeException e) 
		{
			LOG.error("Failed to estimate execution time.", e);
		}
		
		return ret;
	}

}
