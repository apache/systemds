/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.components.runtime.controlprogram;


import java.util.Collection;
import java.util.LinkedList;

import junit.framework.Assert;

import org.junit.Test;

import com.ibm.bi.dml.runtime.controlprogram.parfor.LocalTaskQueue;
import com.ibm.bi.dml.runtime.controlprogram.parfor.Task;
import com.ibm.bi.dml.runtime.controlprogram.parfor.Task.TaskType;
import com.ibm.bi.dml.runtime.instructions.cp.IntObject;

/**
 * TestCases for local task queue.
 * 
 */
public class ParForLocalTaskQueueTest 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private static final long _N = 101;
	
	@Test
	public void testCompleteness() 
		throws InterruptedException 
	{ 
		//create N tasks
		Collection<Task> tasks = new LinkedList<Task>();
		for( long i=1; i<=_N; i++ )
		{
			Task ltask = new Task(TaskType.SET);
			ltask.addIteration(new IntObject("i",i));
			tasks.add(ltask);
 		}
		
		//put all tasks into queue
		LocalTaskQueue<Task> q = new LocalTaskQueue<Task>(); 
		for( Task ltask : tasks )
			q.enqueueTask(ltask);
		
		//read and compare tasks (FIFO semantics of queue)
		boolean ret = (tasks.size() == q.size());
		for( int i=1; i<=_N; i++ )
		{
			Task ltask = q.dequeueTask();
			long val = ltask.getIterations().getFirst().getLongValue();
			ret &= (i == val);
 		}
		
		if( !ret )
			Assert.fail("Wrong order of dequeued tasks.");
	}
	
}
