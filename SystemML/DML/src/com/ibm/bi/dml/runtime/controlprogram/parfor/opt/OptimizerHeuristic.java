/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.runtime.controlprogram.parfor.opt;

import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.POptMode;
import com.ibm.bi.dml.runtime.controlprogram.parfor.opt.PerfTestTool.TestMeasure;
import com.ibm.bi.dml.utils.DMLRuntimeException;


/**
 * Heuristic ParFor Optimizer (time: O(n)):
 * 
 *  
 */
public class OptimizerHeuristic extends OptimizerRuleBased
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public static final double EXEC_TIME_THRESHOLD = 60000; //in ms
			
	@Override
	public CostModelType getCostModelType() 
	{
		return CostModelType.RUNTIME_METRICS;
	}

	@Override
	public PlanInputType getPlanInputType() 
	{
		return PlanInputType.RUNTIME_PLAN;
	}
	
	@Override
	public POptMode getOptMode() 
	{
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
	protected boolean isLargeProblem(OptNode pn)
	{
		boolean ret = false;
		
		try 
		{
			double T = _cost.getEstimate(TestMeasure.EXEC_TIME, pn);
			ret = (T >= EXEC_TIME_THRESHOLD);
		} 
		catch (DMLRuntimeException e) 
		{
			LOG.error("Failed to estimate execution time.", e);
		}
		
		return ret;
	}

}
