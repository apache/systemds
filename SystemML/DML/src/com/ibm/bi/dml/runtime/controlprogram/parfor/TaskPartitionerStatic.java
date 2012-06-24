package com.ibm.bi.dml.runtime.controlprogram.parfor;

import com.ibm.bi.dml.runtime.instructions.CPInstructions.IntObject;

/**
 * This static task partitioner virtually iterates over the given FOR loop (from, to, incr),
 * creates iterations and group them to tasks according to a task size of numIterations/numWorkers. 
 * There, all tasks are equally sized.
 * 
 */
public class TaskPartitionerStatic extends TaskPartitionerFixedsize
{
	public TaskPartitionerStatic( int taskSize, int numThreads, String iterVarName, IntObject fromVal, IntObject toVal, IntObject incrVal ) 
	{
		super(taskSize, iterVarName, fromVal, toVal, incrVal);
	
		_taskSize = _numIter / numThreads;
		_firstnPlus1 = _numIter % numThreads;
	}	
}
