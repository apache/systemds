package com.ibm.bi.dml.test.components.runtime.controlprogram;


import java.util.Collection;
import java.util.LinkedList;

import junit.framework.Assert;

import org.junit.Test;

import com.ibm.bi.dml.runtime.controlprogram.parfor.LocalTaskQueue;
import com.ibm.bi.dml.runtime.controlprogram.parfor.Task;
import com.ibm.bi.dml.runtime.controlprogram.parfor.Task.TaskType;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.IntObject;

/**
 * TestCases for local task queue.
 * 
 */
public class ParForLocalTaskQueueTest 
{
	private static final int _N = 101;
	
	@Test
	public void testCompleteness() 
		throws InterruptedException 
	{ 
		//create N tasks
		Collection<Task> tasks = new LinkedList<Task>();
		for( int i=1; i<=_N; i++ )
		{
			Task ltask = new Task(TaskType.SET);
			ltask.addIteration(new IntObject("i",i));
			tasks.add(ltask);
 		}
		
		//put all tasks into queue
		LocalTaskQueue q = new LocalTaskQueue(); 
		for( Task ltask : tasks )
			q.enqueueTask(ltask);
		
		//read and compare tasks (FIFO semantics of queue)
		boolean ret = (tasks.size() == q.size());
		for( int i=1; i<=_N; i++ )
		{
			Task ltask = q.dequeueTask();
			int val = ltask.getIterations().getFirst().getIntValue();
			ret &= (i == val);
 		}
		
		if( !ret )
			Assert.fail("Wrong order of dequeued tasks.");
	}
	
}
