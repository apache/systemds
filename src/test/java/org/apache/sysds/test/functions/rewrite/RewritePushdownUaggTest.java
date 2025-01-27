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

package org.apache.sysds.test.functions.rewrite;

import java.util.HashMap;

import org.apache.sysds.common.Opcodes;
import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;

/**
 * Regression test for function recompile-once issue with literal replacement.
 * 
 */
public class RewritePushdownUaggTest extends AutomatedTestBase 
{	
	//two aggregation functions as examples
	private static final String TEST_NAME1 = "RewritePushdownColsums";
	private static final String TEST_NAME2 = "RewritePushdownRowsums";
	private static final String TEST_NAME3 = "RewritePushdownColmins";
	private static final String TEST_NAME4 = "RewritePushdownRowmins";
	
	private static final String TEST_DIR = "functions/rewrite/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RewritePushdownUaggTest.class.getSimpleName() + "/";
	
	private static final int rows = 192;
	private static final int cols = 293;
	private static final double eps = Math.pow(10, -10);
	
	@Override
	public void setUp() 
	{
		TestUtils.clearAssertionInformation();
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME4, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4, new String[] { "R" }) );
	}

	@Test
	public void testRewriteColSumsNoRewrite()  {
		testRewritePushdownUagg( TEST_NAME1, false );
	}
	
	@Test
	public void testRewriteRowSumsNoRewrite()  {
		testRewritePushdownUagg( TEST_NAME2, false );
	}
	
	@Test
	public void testRewriteColMinsNoRewrite()  {
		testRewritePushdownUagg( TEST_NAME3, false );
	}
	
	@Test
	public void testRewriteRowMinsNoRewrite()  {
		testRewritePushdownUagg( TEST_NAME4, false );
	}
	
	@Test
	public void testRewriteColSums()  {
		testRewritePushdownUagg( TEST_NAME1, true );
	}
	
	@Test
	public void testRewriteRowSums()  {
		testRewritePushdownUagg( TEST_NAME2, true );
	}
	
	@Test
	public void testRewriteColMins()  {
		testRewritePushdownUagg( TEST_NAME3, true );
	}
	
	@Test
	public void testRewriteRowMins()  {
		testRewritePushdownUagg( TEST_NAME4, true );
	}


	private void testRewritePushdownUagg( String testname, boolean rewrites )
	{	
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		
		try
		{
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{ "-stats","-args", input("X"), output("R") };
			
			fullRScriptName = HOME + testname + ".R";
			rCmd = getRCmd(inputDir(), expectedDir());

			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;

			double[][] X = getRandomMatrix(rows, cols, -1, 1, 0.56d, 7);
			writeInputMatrixWithMTD("X", X, true);
			
			runTest(true, false, null, -1); 
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("R");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			
			//check matrix mult existence
			String check = null;
			if( testname.equals(TEST_NAME1) ) //colsums
				check = rewrites ? Opcodes.UARKP.toString() : Opcodes.UACKP.toString();
			else if( testname.equals(TEST_NAME2) ) //rowsums
				check = rewrites ? Opcodes.UACKP.toString() : Opcodes.UARKP.toString();
			else if( testname.equals(TEST_NAME3) ) //colmins
				check = rewrites ? Opcodes.UARMIN.toString() : Opcodes.UACMIN.toString();
			else if( testname.equals(TEST_NAME4) ) //rowmins
				check = rewrites ? Opcodes.UACMIN.toString() : Opcodes.UARMIN.toString();

			String gpuCheck = "gpu_" + check;
			boolean containsOpcode = Statistics.getCPHeavyHitterOpCodes().contains(check) || Statistics.getCPHeavyHitterOpCodes().contains(gpuCheck);

			Assert.assertTrue( "Missing opcode: "+check, containsOpcode);
		}
		finally
		{
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
		}
	}	
}