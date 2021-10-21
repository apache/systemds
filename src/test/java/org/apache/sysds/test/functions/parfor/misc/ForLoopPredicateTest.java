/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.sysds.test.functions.parfor.misc;

import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;

public class ForLoopPredicateTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "for_pred1a"; //const
	private final static String TEST_NAME2 = "for_pred1b"; //const seq
	private final static String TEST_NAME3 = "for_pred2a"; //var
	private final static String TEST_NAME4 = "for_pred2b"; //var seq
	private final static String TEST_NAME5 = "for_pred3a"; //expression
	private final static String TEST_NAME6 = "for_pred3b"; //expression seq
	private final static String TEST_NAME7 = "for_pred1a_seq"; //const seq two parameters (this tests is only for parser)
	private final static String TEST_NAME8 = "parfor_pred1_neg"; //to:from (negative increment)
	private final static String TEST_NAME9 = "parfor_pred2_neg"; //2:1 (negative increment, two steps)
	private final static String TEST_DIR = "functions/parfor/";
	private final static String TEST_CLASS_DIR = TEST_DIR + ForLoopPredicateTest.class.getSimpleName() + "/";
	
	private final static double from = 1;
	private final static double to = 10.2;
	private final static int increment = 1;
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[]{"R"}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[]{"R"}));
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[]{"R"}));
		addTestConfiguration(TEST_NAME4, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4, new String[]{"R"}));
		addTestConfiguration(TEST_NAME5, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME5, new String[]{"R"}));
		addTestConfiguration(TEST_NAME6, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME6, new String[]{"R"}));
		addTestConfiguration(TEST_NAME7, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME7, new String[]{"R"}));
		addTestConfiguration(TEST_NAME8, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME8, new String[]{"R"}));
		addTestConfiguration(TEST_NAME9, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME9, new String[]{"R"}));
	}

	@Test
	public void testForConstIntegerPredicate() {
		runForPredicateTest(1, true);
	}
	
	@Test
	public void testForConstIntegerSeq2ParametersPredicate() {
		runForPredicateTest(7, true);
	}
	
	@Test
	public void testForConstIntegerSeqPredicate() {
		runForPredicateTest(2, true);
	}
	
	@Test
	public void testForVariableIntegerPredicate() {
		runForPredicateTest(3, true);
	}
	
	@Test
	public void testForVariableIntegerSeqPredicate() {
		runForPredicateTest(4, true);
	}
	
	@Test
	public void testForExpressionIntegerPredicate() {
		runForPredicateTest(5, true);
	}
	
	@Test
	public void testForExpressionIntegerSeqPredicate() {
		runForPredicateTest(6, true);
	}

	@Test
	public void testForConstDoublePredicate() {
		runForPredicateTest(1, false);
	}
	
	@Test
	public void testForConstDoubleSeq2ParametersPredicate() {
		runForPredicateTest(7, false);
	}
	
	@Test
	public void testForConstDoubleSeqPredicate() {
		runForPredicateTest(2, false);
	}
	
	@Test
	public void testForVariableDoublePredicate() {
		runForPredicateTest(3, false);
	}
	
	@Test
	public void testForVariableDoubleSeqPredicate() {
		runForPredicateTest(4, false);
	}
	
	@Test
	public void testForExpressionDoublePredicate() {
		runForPredicateTest(5, false);
	}
	
	@Test
	public void testForExpressionDoubleSeqPredicate() {
		runForPredicateTest(6, false);
	}
	
	@Test
	public void testParFor1IntNegativeIncrement() {
		runForPredicateTest(8, true, true);
	}
	
	@Test
	public void testParFor1DoubleNegativeIncrement() {
		runForPredicateTest(8, false, true);
	}
	
	@Test
	public void testParFor2IntNegativeIncrement() {
		runForPredicateTest(9, true, true);
	}
	
	private void runForPredicateTest( int testNum, boolean intScalar ) {
		runForPredicateTest(testNum, intScalar, false);
	}
	
	private void runForPredicateTest( int testNum, boolean intScalar, boolean negative )
	{
		String TEST_NAME = null;
		switch( testNum ) {
			case 1: TEST_NAME = TEST_NAME1; break;
			case 2: TEST_NAME = TEST_NAME2; break;
			case 3: TEST_NAME = TEST_NAME3; break;
			case 4: TEST_NAME = TEST_NAME4; break;
			case 5: TEST_NAME = TEST_NAME5; break;
			case 6: TEST_NAME = TEST_NAME6; break;
			case 7: TEST_NAME = TEST_NAME7; break;
			case 8: TEST_NAME = TEST_NAME8; break;
			case 9: TEST_NAME = TEST_NAME9; break;
		}
		
		getAndLoadTestConfiguration(TEST_NAME);
		
		Object valFrom = null;
		Object valTo = null;
		Object valIncrement = null;
		if( intScalar ) {
			valFrom = Integer.valueOf((int)Math.round(from));
			valTo = Integer.valueOf((int)Math.round(to));
			valIncrement = Integer.valueOf(increment);
		}
		else {
			valFrom = Double.valueOf(from);
			valTo = Double.valueOf(to);
			valIncrement = Double.valueOf(increment);
		}
		
		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[]{"-args",
			String.valueOf(valFrom), 
			String.valueOf(valTo), 
			String.valueOf(valIncrement), 
			output("R") };
		
		fullRScriptName = HOME + TEST_NAME1 + ".R";
		rCmd = null;

		runTest(true, false, null, -1);
		
		//compare matrices
		HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");
		Assert.assertEquals( Double.valueOf(Math.ceil((Math.round(to)-Math.round(from)+1)/increment)),
				             dmlfile.get(new CellIndex(1,1)));
	}
}
