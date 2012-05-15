package dml.runtime.controlprogram.parfor;

import dml.runtime.instructions.CPInstructions.IntObject;

/**
 * This static task partitioner virtually iterates over the given FOR loop (from, to, incr),
 * creates iterations and group them to tasks according to a task size of numIterations/numWorkers. 
 * There, all tasks are equally sized.
 * 
 * @author mboehm
 */
public class TaskPartitionerStatic extends TaskPartitionerFixedsize
{
	public TaskPartitionerStatic( int taskSize, int numThreads, String iterVarName, IntObject fromVal, IntObject toVal, IntObject incrVal ) 
	{
		super(taskSize, iterVarName, fromVal, toVal, incrVal);
	
		//compute the new static task size
		int lFrom = _fromVal.getIntValue();
		int lTo   = _toVal.getIntValue();
		int lIncr = _incrVal.getIntValue();
		
		int numIterations = (lTo-lFrom+1)/lIncr; 
		_taskSize = numIterations / numThreads;
		_firstnPlus1 = numIterations % numThreads;
	}	
}
