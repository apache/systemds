/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

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
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	public TaskPartitionerStatic( int taskSize, int numThreads, String iterVarName, IntObject fromVal, IntObject toVal, IntObject incrVal ) 
	{
		super(taskSize, iterVarName, fromVal, toVal, incrVal);
	
		_taskSize = _numIter / numThreads;
		_firstnPlus1 = _numIter % numThreads;
	}	
}
