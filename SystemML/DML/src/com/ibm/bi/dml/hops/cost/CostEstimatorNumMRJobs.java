package com.ibm.bi.dml.hops.cost;

import com.ibm.bi.dml.runtime.instructions.Instruction;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;

/**
 * 
 */
public class CostEstimatorNumMRJobs extends CostEstimator
{
	@Override
	protected double getCPInstTimeEstimate( Instruction inst, VarStats[] vs, String[] args  ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		return 0;
	}
	
	@Override
	protected double getMRJobInstTimeEstimate( Instruction inst, VarStats[] vs, String[] args  ) 
		throws DMLRuntimeException, DMLUnsupportedOperationException
	{
		return 1;
	}
}
