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

package org.apache.sysml.yarn.ropt;

import org.apache.sysml.conf.ConfigurationManager;
import org.apache.sysml.conf.DMLConfig;
import org.apache.sysml.hops.OptimizerUtils;

public class YarnOptimizerUtils 
{
	public enum GridEnumType{
		EQUI_GRID,
		EXP_GRID,
		MEM_EQUI_GRID,
		HYBRID_MEM_EXP_GRID,
	}

	/**
	 * 
	 * @return
	 */
	public static double getRemoteMemBudgetMap(long jobLookupId)
	{
		return getRemoteMemBudgetMap(false, jobLookupId);
	}
	
	/**
	 * 
	 * @return
	 */
	public static double getRemoteMemBudgetMap(boolean substractSortBuffer, long jobLookupId)
	{
		double ret = YarnClusterAnalyzer.getRemoteMaxMemoryMap(jobLookupId);
		if( substractSortBuffer )
			ret -= YarnClusterAnalyzer.getRemoteMaxMemorySortBuffer();
		return ret * OptimizerUtils.MEM_UTIL_FACTOR;
	}
	
	/**
	 * 
	 * @return
	 */
	public static double getRemoteMemBudgetReduce(long jobLookupId)
	{
		double ret = YarnClusterAnalyzer.getRemoteMaxMemoryReduce(jobLookupId);
		return ret * OptimizerUtils.MEM_UTIL_FACTOR;
	}
	
	/**
	 * Returns the number of reducers that potentially run in parallel.
	 * This is either just the configured value (SystemML config) or
	 * the minimum of configured value and available reduce slots. 
	 * 
	 * @param configOnly
	 * @return
	 */
	public static int getNumReducers(boolean configOnly, long jobLookupId)
	{
		int ret = ConfigurationManager.getConfig().getIntValue(DMLConfig.NUM_REDUCERS);
		if( !configOnly )
			ret = Math.min(ret,YarnClusterAnalyzer.getRemoteParallelReduceTasks(jobLookupId));
		
		return ret;
	}
	
	/**
	 * 
	 * @param mb
	 * @return
	 */
	public static long toB( long mb )
	{
		return 1024 * 1024 * mb; 
	}
	
	/**
	 * 
	 * @param b
	 * @return
	 */
	public static long toMB( long b )
	{
		return b / (1024 * 1024); 
	}
	
	/**
	 * 
	 * @param minAlloc
	 * @param maxAlloc
	 * @param numCores
	 * @return
	 */
	public static long computeMinContraint( long minAlloc, long maxAlloc, long numCores )
	{
		return ((long)(Math.max(minAlloc, maxAlloc/numCores )/minAlloc)*minAlloc); 	
	}
}
