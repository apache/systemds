package com.ibm.bi.dml.test.components.runtime.controlprogram;

import junit.framework.Assert;

import org.junit.Test;

import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDHandler;
import com.ibm.bi.dml.runtime.controlprogram.parfor.util.IDSequence;

/**
 * Different test cases for IDHandler and IDSequence.
 * 
 *
 */
public class ParForIDUtilsTest 
{
	@Test
	public void testIDSequence() 
	{ 
		int testCount = 100;
		
		IDSequence seq = new IDSequence();
		for( int i=1; i<=testCount; i++ )
			if( i!=seq.getNextID() )
				Assert.fail("Not a sequence number at iteration "+i);
	}
	
	@Test
	public void testLocalIntIDExtraction() 
	{ 
		String str = getLocalTaskID();
		int intVal = getLocalIntID();
		int eIntVal = IDHandler.extractIntID(str);
		
		if( intVal != eIntVal )
			Assert.fail("Different numbers "+intVal+" "+eIntVal);
	}
	
	@Test
	public void testLocalLongIDExtraction() 
	{ 
		String str = getLocalTaskID();
		long longVal = getLocalLongID();
		long eLongVal = IDHandler.extractLongID(str);
		
		if( longVal != eLongVal )
			Assert.fail("Different numbers "+longVal+" "+eLongVal);
	}

	@Test
	public void testClusterIntIDExtraction() 
	{ 
		String str = getClusterTaskID();
		int intVal = getClusterIntID();
		int eIntVal = IDHandler.extractIntID(str);
		
		if( intVal != eIntVal )
			Assert.fail("Different numbers "+intVal+" "+eIntVal);
	}
	
	@Test
	public void testClusterLongIDExtraction() 
	{ 
		String str = getClusterTaskID();
		long longVal = getClusterLongID();
		long eLongVal = IDHandler.extractLongID(str);
		
		if( longVal != eLongVal )
			Assert.fail("Different numbers "+longVal+" "+eLongVal);
	}
	
	@Test
	public void testIDConcat() 
	{ 
		int val1 = 3;
		int val2 = 7;
		long val3  = getConcatID(val1, val2);
		long ret = IDHandler.concatIntIDsToLong(val1, val2);
		
		if( ret != val3 )
			Assert.fail("Different numbers "+ret+" "+val3);
	}
	
	@Test
	public void testIDConcatExtract() 
	{ 
		int val1 = 3;
		int val2 = 7;
		long val3  = getConcatID(val1, val2);
		
		int ret1 = IDHandler.extractIntIDFromLong(val3, 1);
		int ret2 = IDHandler.extractIntIDFromLong(val3, 2);
		
		if( ret1 != val1 )
			Assert.fail("Different numbers "+ret1+" "+val1);
		
		if( ret2 != val2 )
			Assert.fail("Different numbers "+ret2+" "+val2);
	}
	
	
	
	private long getConcatID( int val1, int val2 )
	{
		return val1*(long)Math.pow(2, 32)+val2;
	}
	
	private String getLocalTaskID()
	{
		return "task_local_0002_m_000009";
	}
	
	private int getLocalIntID()
	{
		return 2000009;
	}
	
	private long getLocalLongID()
	{
		return 2000009L;
	}
	
	private String getClusterTaskID()
	{
		return "task_201213111647_0898_m_000001";
	}
	
	private int getClusterIntID()
	{
		//cluster ID: 2012131116470898000001
		//MAX INT (2^31-1):       2147483647
		//cluster int ID:          898000001 
		
		return 898000001;
	}
	
	private long getClusterLongID()
	{
		//cluster ID:       2012131116470898000001
		//MAX LONG (2^63-1):   9223372036854775807 
		//cluster ID:           131116470898000001

		return 131116470898000001L;
	}
}
