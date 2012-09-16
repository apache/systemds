package com.ibm.bi.dml.runtime.controlprogram.parfor.opt;

import com.ibm.bi.dml.parser.ParForStatementBlock;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;

/**
 * 
 * 
 */
class OptimizerDPEnum extends Optimizer
{

	@Override
	public CostModelType getCostModelType() 
	{
		return CostModelType.RUNTIME_METRICS;
	}

	@Override
	public PlanInputType getPlanInputType() 
	{
		return PlanInputType.ABSTRACT_PLAN;
	}

	@Override
	public boolean optimize(ParForStatementBlock sb, ParForProgramBlock pb, OptTree plan, CostEstimator est) 
		throws DMLRuntimeException, DMLUnsupportedOperationException 
	{
		throw new DMLRuntimeException("not implemented yet.");
	}

}
