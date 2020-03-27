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

package org.apache.sysds.test.functions.unary.matrix;

import java.util.HashMap;

import org.junit.Test;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class ReplaceTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "replace_value";
	private final static String TEST_NAME2 = "replace_NaN";
	private final static String TEST_NAME3 = "replace_Infinity";
	private final static String TEST_NAME4 = "replace_NInfinity";
	private final static String TEST_NAME5 = "replace_maxmin";
	
	private final static String TEST_DIR = "functions/unary/matrix/";
	private static final String TEST_CLASS_DIR = TEST_DIR + ReplaceTest.class.getSimpleName() + "/";

	private final static int rows = 1577;
	private final static int cols = 37;
	
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.07;
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME1, 
				new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "C" }) ); 
		addTestConfiguration(TEST_NAME2, 
				new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "C" }) ); 
		addTestConfiguration(TEST_NAME3, 
				new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] { "C" }) ); 
		addTestConfiguration(TEST_NAME4, 
				new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4, new String[] { "C" }) ); 
		addTestConfiguration(TEST_NAME5, 
				new TestConfiguration(TEST_CLASS_DIR, TEST_NAME5, new String[] { "C" }) ); 
	}
	
	@Test
	public void testReplaceZeroDenseCP() 
	{
		runTestReplace( TEST_NAME1, 0, false, ExecType.CP );
	}
	
	@Test
	public void testReplaceValueDenseCP() 
	{
		runTestReplace( TEST_NAME1, 7, false, ExecType.CP );
	}
	
	@Test
	public void testReplaceNaNDenseCP() 
	{
		runTestReplace( TEST_NAME2, Double.NaN, false, ExecType.CP );
	}
	
	@Test
	public void testReplacePInfinityDenseCP() 
	{
		runTestReplace( TEST_NAME3, Double.POSITIVE_INFINITY, false, ExecType.CP );
	}
	
	@Test
	public void testReplaceNInfinityDenseCP() 
	{
		runTestReplace( TEST_NAME4, Double.NEGATIVE_INFINITY, false, ExecType.CP );
	}
	
	@Test
	public void testReplaceMaxMinDenseCP() 
	{
		runTestReplace( TEST_NAME5, -1, false, ExecType.CP );
	}

	@Test
	public void testReplaceZeroSparseCP() 
	{
		runTestReplace( TEST_NAME1, 0, true, ExecType.CP );
	}
	
	@Test
	public void testReplaceValueSparseCP() 
	{
		runTestReplace( TEST_NAME1, 7, true, ExecType.CP );
	}
	
	@Test
	public void testReplaceNaNSparseCP() 
	{
		runTestReplace( TEST_NAME2, Double.NaN, true, ExecType.CP );
	}
	
	@Test
	public void testReplacePInfinitySparseCP() 
	{
		runTestReplace( TEST_NAME3, Double.POSITIVE_INFINITY, true, ExecType.CP );
	}
	
	@Test
	public void testReplaceNInfinitySparseCP() 
	{
		runTestReplace( TEST_NAME4, Double.NEGATIVE_INFINITY, true, ExecType.CP );
	}
	
	@Test
	public void testReplaceMaxMinSparseCP() 
	{
		runTestReplace( TEST_NAME5, -1, true, ExecType.CP );
	}
	
	// ------------------------------------------------------------------------

	@Test
	public void testReplaceZeroDenseSP() 
	{	
		runTestReplace( TEST_NAME1, 0, false, ExecType.SPARK );
	}
	
	@Test
	public void testReplaceValueDenseSP() 
	{	
		runTestReplace( TEST_NAME1, 7, false, ExecType.SPARK );
	}
	
	@Test
	public void testReplaceNaNDenseSP() 
	{	
		runTestReplace( TEST_NAME2, Double.NaN, false, ExecType.SPARK );
	}
	
	@Test
	public void testReplacePInfinityDenseSP() 
	{	
		runTestReplace( TEST_NAME3, Double.POSITIVE_INFINITY, false, ExecType.SPARK );
	}
	
	@Test
	public void testReplaceNInfinityDenseSP() 
	{	
		runTestReplace( TEST_NAME4, Double.NEGATIVE_INFINITY, false, ExecType.SPARK );
	}
	
	@Test
	public void testReplaceMaxMinDenseSP() 
	{	
		runTestReplace( TEST_NAME5, -1, false, ExecType.SPARK );
	}

	@Test
	public void testReplaceZeroSparseSP() 
	{
		runTestReplace( TEST_NAME1, 0, true, ExecType.SPARK );
	}
	
	@Test
	public void testReplaceValueSparseSP() 
	{	
		runTestReplace( TEST_NAME1, 7, true, ExecType.SPARK );
	}
	
	@Test
	public void testReplaceNaNSparseSP() 
	{
		runTestReplace( TEST_NAME2, Double.NaN, true, ExecType.SPARK );
	}
	
	@Test
	public void testReplacePInfinitySparseSP() 
	{
		runTestReplace( TEST_NAME3, Double.POSITIVE_INFINITY, true, ExecType.SPARK );
	}
	
	@Test
	public void testReplaceNInfinitySparseSP() 
	{
		runTestReplace( TEST_NAME4, Double.NEGATIVE_INFINITY, true, ExecType.SPARK );
	}
	
	@Test
	public void testReplaceMaxMinSparseSP() 
	{
		runTestReplace( TEST_NAME5, -1, true, ExecType.SPARK );
	}
	
	private void runTestReplace( String test, double pattern, boolean sparse, ExecType etype )
	{
		ExecMode platformOld = setExecMode(etype);
		
		try
		{
			double sparsity = (sparse)? sparsity2 : sparsity1;
			
			//register test configuration
			TestConfiguration config = getTestConfiguration(test);
			config.addVariable("rows", rows);
			config.addVariable("cols", cols);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + test + ".dml";
			programArgs = new String[]{"-args", input("A"), String.valueOf(rows), String.valueOf(cols), 
				output("C"), String.valueOf(pattern) }; //only respected for TEST_NAME1
			
			fullRScriptName = HOME + test + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + 
				pattern + " " + expectedDir();
			
			double[][] A = getRandomMatrix(rows, cols, 0, 1, sparsity, 7);
			TestUtils.replaceRandom(A, rows, cols, pattern, 10);
			writeInputMatrix("A", A, true);
			writeExpectedMatrix("A", A);

			runTest(true, false, null, -1);
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("C");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("C");
			TestUtils.compareMatrices(dmlfile, rfile, 1e-14, "Stat-DML", "Stat-R");
		}
		finally {
			resetExecMode(platformOld);
		}
	}
}