/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.parfor;

import java.util.HashMap;

import junit.framework.Assert;

import org.junit.Test;

import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;

public class ForLoopPredicateTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME1 = "for_pred1a"; //const
	private final static String TEST_NAME2 = "for_pred1b"; //const seq
	private final static String TEST_NAME3 = "for_pred2a"; //var
	private final static String TEST_NAME4 = "for_pred2b"; //var seq
	private final static String TEST_NAME5 = "for_pred3a"; //expression
	private final static String TEST_NAME6 = "for_pred3b"; //expression seq
	private final static String TEST_DIR = "functions/parfor/";
	
	private final static double from = 1;
	private final static double to = 10.2;
	private final static int increment = 1;
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_DIR, TEST_NAME1, new String[]{"R"}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_DIR, TEST_NAME2, new String[]{"R"}));
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_DIR, TEST_NAME3, new String[]{"R"}));
		addTestConfiguration(TEST_NAME4, new TestConfiguration(TEST_DIR, TEST_NAME4, new String[]{"R"}));
		addTestConfiguration(TEST_NAME5, new TestConfiguration(TEST_DIR, TEST_NAME5, new String[]{"R"}));
		addTestConfiguration(TEST_NAME6, new TestConfiguration(TEST_DIR, TEST_NAME6, new String[]{"R"}));
	}

	@Test
	public void testForConstIntegerPredicate() 
	{
		runForPredicateTest(1, true);
	}
	
	@Test
	public void testForConstIntegerSeqPredicate() 
	{
		runForPredicateTest(2, true);
	}
	
	@Test
	public void testForVariableIntegerPredicate() 
	{
		runForPredicateTest(3, true);
	}
	
	@Test
	public void testForVariableIntegerSeqPredicate() 
	{
		runForPredicateTest(4, true);
	}
	
	@Test
	public void testForExpressionIntegerPredicate() 
	{
		runForPredicateTest(5, true);
	}
	
	@Test
	public void testForExpressionIntegerSeqPredicate() 
	{
		runForPredicateTest(6, true);
	}

	@Test
	public void testForConstDoublePredicate() 
	{
		runForPredicateTest(1, false);
	}
	
	@Test
	public void testForConstDoubleSeqPredicate() 
	{
		runForPredicateTest(2, false);
	}
	
	@Test
	public void testForVariableDoublePredicate() 
	{
		runForPredicateTest(3, false);
	}
	
	@Test
	public void testForVariableDoubleSeqPredicate() 
	{
		runForPredicateTest(4, false);
	}
	
	@Test
	public void testForExpressionDoublePredicate() 
	{
		runForPredicateTest(5, false);
	}
	
	@Test
	public void testForExpressionDoubleSeqPredicate() 
	{
		runForPredicateTest(6, false);
	}
	
	/**
	 * 
	 * @param testNum
	 * @param intScalar
	 */
	private void runForPredicateTest( int testNum, boolean intScalar )
	{
		String TEST_NAME = null;
		switch( testNum )
		{
			case 1: TEST_NAME = TEST_NAME1; break;
			case 2: TEST_NAME = TEST_NAME2; break;
			case 3: TEST_NAME = TEST_NAME3; break;
			case 4: TEST_NAME = TEST_NAME4; break;
			case 5: TEST_NAME = TEST_NAME5; break;
			case 6: TEST_NAME = TEST_NAME6; break;
		}
		
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		
		Object valFrom = null;
		Object valTo = null;
		Object valIncrement = null;
		if( intScalar )
		{
			valFrom = new Integer((int)Math.round(from));
			valTo = new Integer((int)Math.round(to));
			valIncrement = new Integer(increment);
		}
		else
		{
			valFrom = new Double(from);
			valTo = new Double(to);
			valIncrement = new Double(increment);
		}
		
		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[]{"-args", 
				                        String.valueOf(valFrom),
				                        String.valueOf(valTo),
				                        String.valueOf(valIncrement),
				                        HOME + OUTPUT_DIR + "R" };
		fullRScriptName = HOME + TEST_NAME1 + ".R";
		rCmd = null;
		
		loadTestConfiguration(config);


		runTest(true, false, null, -1);
		
		//compare matrices
		HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
		Assert.assertEquals( Math.ceil((Math.round(to)-Math.round(from)+1)/increment),
				             dmlfile.get(new CellIndex(1,1)));
	}
	
}