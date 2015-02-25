/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.components.runtime.controlprogram;

import junit.framework.Assert;

import org.junit.Test;

import com.ibm.bi.dml.runtime.controlprogram.parfor.Task;
import com.ibm.bi.dml.runtime.controlprogram.parfor.Task.TaskType;
import com.ibm.bi.dml.runtime.instructions.cp.IntObject;

/**
 * Different test cases for serialization and parsing of all kinds of task representations.
 * 
 *
 */
public class ParForTaskSerializationTest 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	@Test
	public void testTaskSetStringSerilization() 
	{ 
		long i = 7;
		
		Task t1 = new Task(TaskType.SET);
		t1.addIteration(new IntObject("i",i));
		
		String str = t1.toCompactString();
		Task t2 = Task.parseCompactString(str);
		
		IntObject val = t2.getIterations().getFirst(); 
		
		Assert.assertEquals(i, val.getLongValue());
	}
	
	@Test
	public void testTaskSetStringSerilizationMultiple() 
	{ 
		long i1 = 3;
		long i2 = 7;
		
		Task t1 = new Task(TaskType.SET);
		t1.addIteration(new IntObject("i",i1));
		t1.addIteration(new IntObject("i",i2));
		
		String str = t1.toCompactString();
		Task t2 = Task.parseCompactString(str);
		
		IntObject val1 = t2.getIterations().get(0); 
		IntObject val2 = t2.getIterations().get(1);
		
		Assert.assertEquals(i1, val1.getLongValue());
		Assert.assertEquals(i2, val2.getLongValue());
	}
	
	@Test
	public void testTaskRangeStringSerilization() 
	{ 
		long from = 1;
		long to = 10;
		long incr = 2;
		
		Task t1 = new Task(TaskType.RANGE);
		t1.addIteration(new IntObject("i",from));
		t1.addIteration(new IntObject("i",to));
		t1.addIteration(new IntObject("i",incr));
		
		String str = t1.toCompactString();
		Task t2 = Task.parseCompactString(str);
		
		IntObject val1 = t2.getIterations().get(0); 
		IntObject val2 = t2.getIterations().get(1);
		IntObject val3 = t2.getIterations().get(2);
		
		Assert.assertEquals(from, val1.getLongValue());
		Assert.assertEquals(to, val2.getLongValue());
		Assert.assertEquals(incr, val3.getLongValue());
	}

	
	
	@Test
	public void testTaskStringNumberLength() 
	{ 
		long val = 7;
		String valStr = "007";
		
		Task t1 = new Task(TaskType.RANGE);
		t1.addIteration(new IntObject("i",val));
		
		String str = t1.toCompactString( valStr.length() );
		Assert.assertEquals(valStr, str.substring(9, 12));
		
		Task t2 = Task.parseCompactString(str);		
		IntObject valRet = t2.getIterations().get(0);
		Assert.assertEquals(val, valRet.getLongValue());
	}
}
