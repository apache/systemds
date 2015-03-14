/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.binary.scalar;

import java.io.IOException;

import junit.framework.Assert;

import org.junit.Test;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;

/**
 * The main purpose of this test is to verify all combinations of
 * scalar string comparisons.
 * 
 */
public class FullStringComparisonTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME1 = "FullStringComparisonTest";
	private final static String TEST_DIR = "functions/binary/scalar/";
	
	public enum Type{
		GREATER,
		LESS,
		EQUALS,
		NOT_EQUALS,
		GREATER_EQUALS,
		LESS_EQUALS,
	}
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_DIR, TEST_NAME1, new String[] { "B" })   ); 
	}

	@Test
	public void testStringCompareEqualsTrue() 
	{
		runStringComparison(Type.EQUALS, true);
	}
	
	@Test
	public void testStringCompareEqualsFalse() 
	{
		runStringComparison(Type.EQUALS, false);
	}
	
	@Test
	public void testStringCompareNotEqualsTrue() 
	{
		runStringComparison(Type.NOT_EQUALS, true);
	}
	
	@Test
	public void testStringCompareNotEqualsFalse() 
	{
		runStringComparison(Type.NOT_EQUALS, false);
	}
	
	@Test
	public void testStringCompareGreaterTrue() 
	{
		runStringComparison(Type.GREATER, true);
	}
	
	@Test
	public void testStringCompareGreaterFalse() 
	{
		runStringComparison(Type.GREATER, false);
	}
	
	@Test
	public void testStringCompareGreaterEqualsTrue() 
	{
		runStringComparison(Type.GREATER_EQUALS, true);
	}
	
	@Test
	public void testStringCompareGreaterEqualsFalse() 
	{
		runStringComparison(Type.GREATER_EQUALS, false);
	}
	
	@Test
	public void testStringCompareLessTrue() 
	{
		runStringComparison(Type.LESS, true);
	}
	
	@Test
	public void testStringCompareLessFalse() 
	{
		runStringComparison(Type.LESS, false);
	}
	
	@Test
	public void testStringCompareLessEqualsTrue() 
	{
		runStringComparison(Type.LESS_EQUALS, true);
	}
	
	@Test
	public void testStringCompareLessEqualsFalse() 
	{
		runStringComparison(Type.LESS_EQUALS, false);
	}
	
	
	
	/**
	 * 
	 * @param type
	 * @param instType
	 * @param sparse
	 */
	private void runStringComparison( Type type, boolean trueCondition )
	{
		String TEST_NAME = TEST_NAME1;
		
		String string1 = "abcd";
		String string2 = null;
		switch( type ){
			case EQUALS:     string2 = trueCondition ? "abcd" : "xyz"; break;
			case NOT_EQUALS: string2 = trueCondition ? "xyz" : "abcd"; break;
			case LESS:       string2 = trueCondition ? "dcba" : "aabbccdd"; break;
			case LESS_EQUALS: string2 = trueCondition ? "abce" : "aabbccdd"; break;
			case GREATER:     string2 = trueCondition ? "aabbccdd" : "dcba"; break;
			case GREATER_EQUALS: string2 = trueCondition ? "aabbccdd" : "abce"; break;
		}
		
		TestConfiguration config = getTestConfiguration(TEST_NAME);
			
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[]{"-args", 
				                        string1,
				                        string2,
				                        Integer.toString(type.ordinal()),
				                        HOME + OUTPUT_DIR + "B"    };
		
		loadTestConfiguration(config);

		//run tests
		runTest(true, false, null, -1); 
		
		//compare result
		try {
			boolean retCondition = MapReduceTool.readBooleanFromHDFSFile(HOME + OUTPUT_DIR + "B");
			Assert.assertEquals(trueCondition, retCondition);
		} 
		catch (IOException e) {
			Assert.fail(e.getMessage());
		}
	}
}