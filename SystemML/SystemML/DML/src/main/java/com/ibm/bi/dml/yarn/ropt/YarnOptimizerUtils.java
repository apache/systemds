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
import com.ibm.bi.dml.utils.Explain.ExplainType;

public class YarnOptimizerUtils 
{
	private static CompilationMode _compileMode = CompilationMode.O0_COMPILE_AND_EXECUTION;
	
	public enum GridEnumType{
		EQUI_GRID,
		EXP_GRID,
		HYBRID_MEM_EQUI_GRID,
		HYBRID2_MEM_EXP_GRID,
	}
	
	public enum CompilationMode { 
		O0_COMPILE_AND_EXECUTION,
		O1_COMPILE_ONLY_SILENT,
		O2_COMPILE_ONLY_AND_HOP_ESTIMATE,
		O3_COMPILE_ONLY_AND_EXPLAIN_RUNTIME,
	};
	
	public static CompilationMode getCompileMode() {
		return _compileMode;
	}
	
	public static void setCompileMode(CompilationMode mode) {
		_compileMode = mode;
	}
	
	public static boolean executeAfterCompile() {
		return _compileMode == CompilationMode.O0_COMPILE_AND_EXECUTION;
	}
	
	public static boolean silentCompile() {
		return (_compileMode == CompilationMode.O1_COMPILE_ONLY_SILENT) ||
				(_compileMode == CompilationMode.O0_COMPILE_AND_EXECUTION);
	}
	
	public static ExplainType getCompileExplainType() {
		if (_compileMode == CompilationMode.O3_COMPILE_ONLY_AND_EXPLAIN_RUNTIME)
			return ExplainType.RUNTIME;
		return ExplainType.NONE;
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
		long minL = (long)minAlloc; 
		return ((long)(Math.max(minAlloc, maxAlloc/numCores )/minL)*minAlloc); 	
	}
}
