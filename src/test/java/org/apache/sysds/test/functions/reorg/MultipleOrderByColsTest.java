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

package org.apache.sysds.test.functions.reorg;

import java.util.HashMap;

import org.apache.sysds.common.Opcodes;
import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;

public class MultipleOrderByColsTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "OrderMultiBy";
	private final static String TEST_NAME2 = "OrderMultiBy2";
	
	private final static String TEST_DIR = "functions/reorg/";
	private static final String TEST_CLASS_DIR = TEST_DIR + MultipleOrderByColsTest.class.getSimpleName() + "/";
	private final static double eps = 1e-10;
	
	private final static int rows = 1017;
	private final static int cols = 736;
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.07;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1,new String[]{"B"}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2,new String[]{"B"}));
	}
	
	@Test
	public void testOrderDenseAscDataCP() {
		runOrderTest(TEST_NAME1, false, false, false, ExecType.CP);
	}
	
	@Test
	public void testOrderDenseAscIxCP() {
		runOrderTest(TEST_NAME1, false, false, true, ExecType.CP);
	}
	
	@Test
	public void testOrderDenseDescDataCP() {
		runOrderTest(TEST_NAME1, false, true, false, ExecType.CP);
	}
	
	@Test
	public void testOrderDenseDescIxCP() {
		runOrderTest(TEST_NAME1, false, true, true, ExecType.CP);
	}
	
	@Test
	public void testOrderSparseAscDataCP() {
		runOrderTest(TEST_NAME1, true, false, false, ExecType.CP);
	}
	
	@Test
	public void testOrderSparseAscIxCP() {
		runOrderTest(TEST_NAME1, true, false, true, ExecType.CP);
	}
	
	@Test
	public void testOrderSparseDescDataCP() {
		runOrderTest(TEST_NAME1, true, true, false, ExecType.CP);
	}
	
	@Test
	public void testOrderSparseDescIxCP() {
		runOrderTest(TEST_NAME1, true, true, true, ExecType.CP);
	}

	@Test
	public void testOrder2DenseAscDataCP() {
		runOrderTest(TEST_NAME2, false, false, false, ExecType.CP);
	}
	
	@Test
	public void testOrder2DenseDescDataCP() {
		runOrderTest(TEST_NAME2, false, true, false, ExecType.CP);
	}
	
	@Test
	public void testOrder2SparseAscDataCP() {
		runOrderTest(TEST_NAME2, true, false, false, ExecType.CP);
	}
	
	@Test
	public void testOrder2SparseDescDataCP() {
		runOrderTest(TEST_NAME2, true, true, false, ExecType.CP);
	}
	
	@Test
	public void testOrderDenseAscDataSP() {
		runOrderTest(TEST_NAME1, false, false, false, ExecType.SPARK);
	}
	
	@Test
	public void testOrderDenseAscIxSP() {
		runOrderTest(TEST_NAME1, false, false, true, ExecType.SPARK);
	}
	
	@Test
	public void testOrderDenseDescDataSP() {
		runOrderTest(TEST_NAME1, false, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testOrderDenseDescIxSP() {
		runOrderTest(TEST_NAME1, false, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testOrderSparseAscDataSP() {
		runOrderTest(TEST_NAME1, true, false, false, ExecType.SPARK);
	}
	
	@Test
	public void testOrderSparseAscIxSP() {
		runOrderTest(TEST_NAME1, true, false, true, ExecType.SPARK);
	}
	
	@Test
	public void testOrderSparseDescDataSP() {
		runOrderTest(TEST_NAME1, true, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testOrderSparseDescIxSP() {
		runOrderTest(TEST_NAME1, true, true, true, ExecType.SPARK);
	}
	
	private void runOrderTest( String testname, boolean sparse, boolean desc, boolean ixret, ExecType et)
	{
		ExecMode platformOld = rtplatform;
		switch( et ){
			case SPARK: rtplatform = ExecMode.SPARK; break;
			default: rtplatform = ExecMode.HYBRID; break;
		}
	
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == ExecMode.SPARK || rtplatform == ExecMode.HYBRID )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		
		try
		{
			String TEST_NAME = testname;
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-stats","-args", input("A"), 
				String.valueOf(desc).toUpperCase(), String.valueOf(ixret).toUpperCase(), output("B") };
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " +
				String.valueOf(desc).toUpperCase()+" "+String.valueOf(ixret).toUpperCase()+" "+expectedDir();
			
			double sparsity = (sparse) ? sparsity2 : sparsity1; //with rounding for duplicates
			double[][] A = TestUtils.round(getRandomMatrix(rows, cols, -10, 10, sparsity, 7));
			writeInputMatrixWithMTD("A", A, true);
			
			runTest(true, false, null, -1);
			runRScript(true);
			
			//compare matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("B");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("B");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			
			//check for applied rewrite
			if( testname.equals(TEST_NAME2) && !ixret )
				Assert.assertTrue(Statistics.getCPHeavyHitterCount(Opcodes.SORT.toString())==1);
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}
