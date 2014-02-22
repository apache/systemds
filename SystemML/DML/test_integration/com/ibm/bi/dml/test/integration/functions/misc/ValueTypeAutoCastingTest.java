/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.misc;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLException;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;

/**
 *   
 */
public class ValueTypeAutoCastingTest extends AutomatedTestBase
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_DIR = "functions/misc/";

	private final static String TEST_NAME1 = "iterablePredicate";
	private final static String TEST_NAME2 = "conditionalPredicateWhile";
	private final static String TEST_NAME3 = "conditionalPredicateIf";
	private final static String TEST_NAME4 = "functionInlining";
	private final static String TEST_NAME5 = "functionNoInlining";
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_DIR, TEST_NAME1, new String[] {"R"}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_DIR, TEST_NAME2, new String[] {"R"}));
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_DIR, TEST_NAME3, new String[] {"R"}));
		addTestConfiguration(TEST_NAME4, new TestConfiguration(TEST_DIR, TEST_NAME4, new String[] {"R"}));
		addTestConfiguration(TEST_NAME5, new TestConfiguration(TEST_DIR, TEST_NAME5, new String[] {"R"}));
	}
	
	@Test
	public void testIterablePredicateDouble() 
	{ 
		runTest( TEST_NAME1, ValueType.DOUBLE, false ); 
	}
	
	@Test
	public void testIterablePredicateInteger() 
	{ 
		runTest( TEST_NAME1, ValueType.INT, false ); 
	}
	
	@Test
	public void testIterablePredicateBoolean() 
	{ 
		runTest( TEST_NAME1, ValueType.BOOLEAN, true ); 
	}
	
	@Test
	public void testConditionalPredicateWhileDouble() 
	{ 
		runTest( TEST_NAME2, ValueType.DOUBLE, false ); 
	}
	
	@Test
	public void testConditionalPredicateWhileInteger() 
	{ 
		runTest( TEST_NAME2, ValueType.INT, false ); 
	}
	
	@Test
	public void testConditionalPredicateWhileBoolean() 
	{ 
		runTest( TEST_NAME2, ValueType.BOOLEAN, false ); 
	}
	
	@Test
	public void testConditionalPredicateIfDouble() 
	{ 
		runTest( TEST_NAME3, ValueType.DOUBLE, false ); 
	}
	
	@Test
	public void testConditionalPredicateIfInteger() 
	{ 
		runTest( TEST_NAME3, ValueType.INT, false ); 
	}
	
	@Test
	public void testConditionalPredicateIfBoolean() 
	{ 
		runTest( TEST_NAME3, ValueType.BOOLEAN, false ); 
	}
	
	@Test
	public void testFunctionInliningDouble() 
	{ 
		runTest( TEST_NAME4, ValueType.DOUBLE, false ); 
	}
	
	@Test
	public void testFunctionInliningInteger() 
	{ 
		runTest( TEST_NAME4, ValueType.INT, false ); 
	}
	
	@Test
	public void testFunctionInliningBoolean() 
	{ 
		runTest( TEST_NAME4, ValueType.BOOLEAN, false ); 
	}
	
	@Test
	public void testFunctionNoInliningDouble() 
	{ 
		runTest( TEST_NAME5, ValueType.DOUBLE, false ); 
	}
	
	@Test
	public void testFunctionNoInliningInteger() 
	{ 
		runTest( TEST_NAME5, ValueType.INT, false ); 
	}
	
	@Test
	public void testFunctionNoInliningBoolean() 
	{ 
		runTest( TEST_NAME5, ValueType.BOOLEAN, false ); 
	}

	
	/**
	 * 
	 * @param cfc
	 * @param vt
	 */
	private void runTest( String testName, ValueType vt, boolean exceptionExpected ) 
	{
		String TEST_NAME = testName;
		
		try
		{		
			TestConfiguration config = getTestConfiguration(TEST_NAME);
		    
			//create input data
			double[][] V = getRandomMatrix(1, 2, 3, 7, 1.0, System.nanoTime());
			String val1 = null, val2 = null;
			switch(vt) {
				case DOUBLE:
					val1 = Double.toString(V[0][0]); 
					val2 = Double.toString(V[0][1]); break;
				case INT:
					val1 = Integer.toString((int)V[0][0]); 
					val2 = Integer.toString((int)V[0][1]); break;
				case BOOLEAN:
					val1 = Boolean.toString( (V[0][0]!=0) ); 
					val2 = Boolean.toString( (V[0][1]!=0) ); break;
			}
			
		    String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", val1, val2 };
			
			loadTestConfiguration(config);
			
			//run tests
	        runTest(true, exceptionExpected, DMLException.class, 0);
		}
		catch(Exception ex)
		{
			throw new RuntimeException(ex);
		}
	}
}
