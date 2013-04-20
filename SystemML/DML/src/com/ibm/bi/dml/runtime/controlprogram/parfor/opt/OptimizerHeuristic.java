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
