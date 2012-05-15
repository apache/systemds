package dml.runtime.controlprogram.parfor;

import dml.runtime.instructions.CPInstructions.IntObject;

/**
 * 
 * @author mboehm
 */
public class TaskPartitionerFactoringConstrained extends TaskPartitionerFactoring
{
	protected int _constraint = -1;
	
	public TaskPartitionerFactoringConstrained( int taskSize, int numThreads, int constraint, String iterVarName, IntObject fromVal, IntObject toVal, IntObject incrVal ) 
	{
		super(taskSize, numThreads, iterVarName, fromVal, toVal, incrVal);
		
		_constraint = constraint;
	}

	@Override
	protected int determineNextBatchSize(int R, int P) 
	{
		int x = 2;
		int K = (int) Math.ceil((double)R / ( x * P )); //NOTE: round creates more tasks
		
		if( K < _constraint ) //account for rounding errors
			K = _constraint;
		
		return K;
	}
	
}
