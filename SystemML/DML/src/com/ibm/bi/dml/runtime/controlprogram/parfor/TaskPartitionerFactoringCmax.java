package com.ibm.bi.dml.runtime.controlprogram.parfor;

import com.ibm.bi.dml.runtime.instructions.CPInstructions.IntObject;

/**
 * Factoring with maximum constraint (e.g., if LIX matrix out-of-core and we need
 * to bound the maximum number of iterations per map task -> memory bounds) 
 */
public class TaskPartitionerFactoringCmax extends TaskPartitionerFactoring
{
	protected int _constraint = -1;
	
	public TaskPartitionerFactoringCmax( int taskSize, int numThreads, int constraint, String iterVarName, IntObject fromVal, IntObject toVal, IntObject incrVal ) 
	{
		super(taskSize, numThreads, iterVarName, fromVal, toVal, incrVal);
		
		_constraint = constraint;
	}

	@Override
	protected int determineNextBatchSize(int R, int P) 
	{
		int x = 2;
		int K = (int) Math.ceil((double)R / ( x * P )); //NOTE: round creates more tasks
		
		if( K > _constraint ) //account for rounding errors
			K = _constraint;
		
		return K;
	}
	
}
