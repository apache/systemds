/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.yarn.ropt;

import com.ibm.bi.dml.conf.ConfigurationManager;
import com.ibm.bi.dml.conf.DMLConfig;
import com.ibm.bi.dml.hops.OptimizerUtils;

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
