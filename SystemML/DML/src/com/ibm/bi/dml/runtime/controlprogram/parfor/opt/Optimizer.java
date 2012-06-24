package com.ibm.bi.dml.runtime.controlprogram.parfor.opt;

import com.ibm.bi.dml.parser.ParForStatementBlock;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock;
import com.ibm.bi.dml.utils.DMLRuntimeException;
import com.ibm.bi.dml.utils.DMLUnsupportedOperationException;


/**
 * Generic optimizer super class that defines the interface of all implemented optimizers.
 * Optimization objective: \phi: \min T(prog) | k \leq ck \wedge m(prog) \leq cm 
 *                                      with T(p)=max_(1\leq i\leq k)(T(prog_i). 
 * 
 * Known implementation classes: OptimizerHeuristic (time: O(m)), OptimizerGreedyEnum 
 * (time: O(m^2)), and OptimizerDPEnum (time: O(2^m)) 
 * 
 * 
 */
public abstract class Optimizer 
{
	protected long _numTotalPlans     = -1;
	protected long _numEvaluatedPlans = -1;
	
	protected Optimizer()
	{
		_numTotalPlans     = 0;
		_numEvaluatedPlans = 0;
	}
	
	/**
	 * 
	 * @param plan
	 * @return true if plan changed, false otherwise
	 * @throws DMLUnsupportedOperationException 
	 * @throws DMLRuntimeException 
	 */
	public abstract boolean optimize(ParForStatementBlock sb, ParForProgramBlock pb, OptTree plan) 
		throws DMLRuntimeException, DMLUnsupportedOperationException;	
	
	///////
	//methods for evaluating the overall properties and costing  

	/**
	 *
	 * @return
	 */
	public long getNumTotalPlans()
	{
		return _numTotalPlans;
	}
	
	/**
	 * 
	 * @return
	 */
	public long getNumEvaluatedPlans()
	{
		return _numEvaluatedPlans;
	}
}
