package com.ibm.bi.dml.test.components.runtime.controlprogram;

import junit.framework.Assert;

import org.junit.Test;

import com.ibm.bi.dml.runtime.controlprogram.parfor.Task;
import com.ibm.bi.dml.runtime.controlprogram.parfor.Task.TaskType;
import com.ibm.bi.dml.runtime.instructions.CPInstructions.IntObject;

/**
 * Different test cases for serialization and parsing of all kinds of task representations.
 * 
 * @author mboehm
 *
 */
public class ParForTaskSerializationTest 
{
	@Test
	public void testTaskSetStringSerilization() 
	{ 
		int i = 7;
		
		Task t1 = new Task(TaskType.ITERATION_SET);
		t1.addIteration(new IntObject("i",i));
		
		String str = t1.toCompactString();
		Task t2 = Task.parseCompactString(str);
		
		IntObject val = t2.getIterations().getFirst(); 
		
		Assert.assertEquals(i, val.getIntValue());
	}
	
	@Test
	public void testTaskSetStringSerilizationMultiple() 
	{ 
		int i1 = 3;
		int i2 = 7;
		
		Task t1 = new Task(TaskType.ITERATION_SET);
		t1.addIteration(new IntObject("i",i1));
		t1.addIteration(new IntObject("i",i2));
		
		String str = t1.toCompactString();
		Task t2 = Task.parseCompactString(str);
		
		IntObject val1 = t2.getIterations().get(0); 
		IntObject val2 = t2.getIterations().get(1);
		
		Assert.assertEquals(i1, val1.getIntValue());
		Assert.assertEquals(i2, val2.getIntValue());
	}
	
	@Test
	public void testTaskSetBinarySerilization() 
	{ 
		int i = 7;
		
		Task t1 = new Task(TaskType.ITERATION_SET);
		t1.addIteration(new IntObject("i",i));
		
		byte[] b = t1.toBinary();
		Task t2 = Task.parseBinary(b);
		
		IntObject val = t2.getIterations().getFirst(); 
		
		Assert.assertEquals(i, val.getIntValue());
	}

	@Test
	public void testTaskSetBinarySerilizationMultiple() 
	{ 
		int i1 = 3;
		int i2 = 7;
		
		Task t1 = new Task(TaskType.ITERATION_SET);
		t1.addIteration(new IntObject("i",i1));
		t1.addIteration(new IntObject("i",i2));
		
		byte[] b = t1.toBinary();
		Task t2 = Task.parseBinary(b);
		
		IntObject val1 = t2.getIterations().get(0); 
		IntObject val2 = t2.getIterations().get(1);
		
		Assert.assertEquals(i1, val1.getIntValue());
		Assert.assertEquals(i2, val2.getIntValue());
	}

	
	@Test
	public void testTaskRangeStringSerilization() 
	{ 
		int to = 1;
		int from = 10;
		int incr = 2;
		
		Task t1 = new Task(TaskType.ITERATION_RANGE);
		t1.addIteration(new IntObject("i",to));
		t1.addIteration(new IntObject("i",from));
		t1.addIteration(new IntObject("i",incr));
		
		String str = t1.toCompactString();
		Task t2 = Task.parseCompactString(str);
		
		IntObject val1 = t2.getIterations().get(0); 
		IntObject val2 = t2.getIterations().get(1);
		IntObject val3 = t2.getIterations().get(2);
		
		Assert.assertEquals(to, val1.getIntValue());
		Assert.assertEquals(from, val2.getIntValue());
		Assert.assertEquals(incr, val3.getIntValue());
	}
	
	@Test
	public void testTaskRangeBinarySerilization() 
	{ 
		int to = 1;
		int from = 10;
		int incr = 2;
		
		Task t1 = new Task(TaskType.ITERATION_RANGE);
		t1.addIteration(new IntObject("i",to));
		t1.addIteration(new IntObject("i",from));
		t1.addIteration(new IntObject("i",incr));
		
		byte[] b = t1.toBinary();
		Task t2 = Task.parseBinary(b);
		
		IntObject val1 = t2.getIterations().get(0); 
		IntObject val2 = t2.getIterations().get(1);
		IntObject val3 = t2.getIterations().get(2);
		
		Assert.assertEquals(to, val1.getIntValue());
		Assert.assertEquals(from, val2.getIntValue());
		Assert.assertEquals(incr, val3.getIntValue());
	}

}
