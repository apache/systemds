package com.ibm.bi.dml.runtime.controlprogram.parfor;

import com.ibm.bi.dml.runtime.instructions.CPInstructions.IntObject;

/**
 * This static task partitioner virtually iterates over the given FOR loop (from, to, incr),
 * creates iterations and group them to tasks according to a task size of numIterations/numWorkers. 
 * There, all tasks are equally sized.
 * 
 */
public class TaskPartitionerNaive extends TaskPartitionerFixedsize
{
	public TaskPartitionerNaive( int taskSize, String iterVarName, IntObject fromVal, IntObject toVal, IntObject incrVal ) 
	{
		super(taskSize, iterVarName, fromVal, toVal, incrVal);
	
		//compute the new task size
		_taskSize = 1;
	}	
}
