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

package org.apache.sysds.api.ropt.cost;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.LocalVariableMap;
import org.apache.sysds.runtime.controlprogram.Program;
import org.apache.sysds.runtime.controlprogram.ProgramBlock;
import org.apache.sysds.runtime.controlprogram.context.ExecutionContext;
import org.apache.sysds.runtime.controlprogram.parfor.stat.Timing;

import java.util.HashMap;

public class CostEstimationWrapper 
{
	public enum CostType { 
		NUM_MRJOBS, //based on number of MR jobs, [number MR jobs]
		STATIC // based on FLOPS, read/write, etc, [time in sec]
	}
	
	private static final Log LOG = LogFactory.getLog(CostEstimationWrapper.class.getName());
	private static final CostType DEFAULT_COSTTYPE = CostType.STATIC;
	
	private static CostEstimator _costEstim = null;
	
	
	static  {
		//create cost estimator
		try {
			//TODO config parameter?
			_costEstim = createCostEstimator(DEFAULT_COSTTYPE);
		}
		catch(Exception ex) {
			LOG.error("Failed cost estimator initialization.", ex);
		}
	}
	
	public static double getTimeEstimate(ExecutionContext ec) {
		double costs = _costEstim.getTimeEstimate(ec);
		// TODO: maybe do some logging here
		return costs;
	}

	private static CostEstimator createCostEstimator(CostType type ) {
		switch( type )
		{
			case STATIC:
				return new CostEstimatorStaticRuntime();
			default:
				throw new DMLRuntimeException("Unknown cost type: "+type);
		}
	}
}
