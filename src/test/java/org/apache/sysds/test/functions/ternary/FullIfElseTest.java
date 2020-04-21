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

package org.apache.sysds.test.functions.ternary;

import java.util.HashMap;

import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class FullIfElseTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "TernaryIfElse";
	
	private final static String TEST_DIR = "functions/ternary/";
	private final static String TEST_CLASS_DIR = TEST_DIR + FullIfElseTest.class.getSimpleName() + "/";
	
	private final static int rows = 2111;
	private final static int cols = 30;
	
	private final static double sparsity1 = 0.6;
	private final static double sparsity2 = 0.1;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "R" }) );
	}

	@Test
	public void testScalarScalarScalarDenseCP() {
		runIfElseTest(false, false, false, false, ExecType.CP);
	}
	
	@Test
	public void testMatrixScalarScalarDenseCP() {
		runIfElseTest(true, false, false, false, ExecType.CP);
	}
	
	@Test
	public void testScalarMatrixScalarDenseCP() {
		runIfElseTest(false, true, false, false, ExecType.CP);
	}
	
	@Test
	public void testMatrixMatrixScalarDenseCP() {
		runIfElseTest(true, true, false, false, ExecType.CP);
	}
	
	@Test
	public void testScalarScalarMatrixDenseCP() {
		runIfElseTest(false, false, true, false, ExecType.CP);
	}
	
	@Test
	public void testMatrixScalarMatrixDenseCP() {
		runIfElseTest(true, false, true, false, ExecType.CP);
	}
	
	@Test
	public void testScalarMatrixMatrixDenseCP() {
		runIfElseTest(false, true, true, false, ExecType.CP);
	}
	
	@Test
	public void testMatrixMatrixMatrixDenseCP() {
		runIfElseTest(true, true, true, false, ExecType.CP);
	}

	@Test
	public void testScalarScalarScalarSparseCP() {
		runIfElseTest(false, false, false, true, ExecType.CP);
	}
	
	@Test
	public void testMatrixScalarScalarSparseCP() {
		runIfElseTest(true, false, false, true, ExecType.CP);
	}
	
	@Test
	public void testScalarMatrixScalarSparseCP() {
		runIfElseTest(false, true, false, true, ExecType.CP);
	}
	
	@Test
	public void testMatrixMatrixScalarSparseCP() {
		runIfElseTest(true, true, false, true, ExecType.CP);
	}
	
	@Test
	public void testScalarScalarMatrixSparseCP() {
		runIfElseTest(false, false, true, true, ExecType.CP);
	}
	
	@Test
	public void testMatrixScalarMatrixSparseCP() {
		runIfElseTest(true, false, true, true, ExecType.CP);
	}
	
	@Test
	public void testScalarMatrixMatrixSparseCP() {
		runIfElseTest(false, true, true, true, ExecType.CP);
	}
	
	@Test
	public void testMatrixMatrixMatrixSparseCP() {
		runIfElseTest(true, true, true, true, ExecType.CP);
	}

	//SPARK
	
	@Test
	public void testScalarScalarScalarDenseSP() {
		runIfElseTest(false, false, false, false, ExecType.SPARK);
	}
	
	@Test
	public void testMatrixScalarScalarDenseSP() {
		runIfElseTest(true, false, false, false, ExecType.SPARK);
	}
	
	@Test
	public void testScalarMatrixScalarDenseSP() {
		runIfElseTest(false, true, false, false, ExecType.SPARK);
	}
	
	@Test
	public void testMatrixMatrixScalarDenseSP() {
		runIfElseTest(true, true, false, false, ExecType.SPARK);
	}
	
	@Test
	public void testScalarScalarMatrixDenseSP() {
		runIfElseTest(false, false, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testMatrixScalarMatrixDenseSP() {
		runIfElseTest(true, false, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testScalarMatrixMatrixDenseSP() {
		runIfElseTest(false, true, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testMatrixMatrixMatrixDenseSP() {
		runIfElseTest(true, true, true, false, ExecType.SPARK);
	}

	@Test
	public void testScalarScalarScalarSparseSP() {
		runIfElseTest(false, false, false, true, ExecType.SPARK);
	}
	
	@Test
	public void testMatrixScalarScalarSparseSP() {
		runIfElseTest(true, false, false, true, ExecType.SPARK);
	}
	
	@Test
	public void testScalarMatrixScalarSparseSP() {
		runIfElseTest(false, true, false, true, ExecType.SPARK);
	}
	
	@Test
	public void testMatrixMatrixScalarSparseSP() {
		runIfElseTest(true, true, false, true, ExecType.SPARK);
	}
	
	@Test
	public void testScalarScalarMatrixSparseSP() {
		runIfElseTest(false, false, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testMatrixScalarMatrixSparseSP() {
		runIfElseTest(true, false, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testScalarMatrixMatrixSparseSP() {
		runIfElseTest(false, true, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testMatrixMatrixMatrixSparseSP() {
		runIfElseTest(true, true, true, true, ExecType.SPARK);
	}

	private void runIfElseTest(boolean matrix1, boolean matrix2, boolean matrix3, boolean sparse, ExecType et)
	{
		//rtplatform for MR
		ExecMode platformOld = rtplatform;
		switch( et ){
			case SPARK: rtplatform = ExecMode.SPARK; break;
			default: rtplatform = ExecMode.HYBRID; break;
		}
	
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		boolean rewritesOld = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		if( rtplatform == ExecMode.SPARK || rtplatform == ExecMode.HYBRID )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = false; //test runtime ops
		
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME1);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME1 + ".dml";
			programArgs = new String[]{"-explain","-args", input("A"), input("B"), input("C"), output("R")};
			fullRScriptName = HOME + TEST_NAME1 + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();
	
			//generate actual datasets (matrices and scalars)
			double sparsity = sparse ? sparsity2 : sparsity1;
			double[][] A = matrix1 ? getRandomMatrix(rows, cols, 0, 1, sparsity, 1) : getScalar(1);
			writeInputMatrixWithMTD("A", A, true);
			double[][] B = matrix2 ? getRandomMatrix(rows, cols, 0, 1, sparsity, 2) : getScalar(2);
			writeInputMatrixWithMTD("B", B, true);
			double[][] C = matrix3 ? getRandomMatrix(rows, cols, 0, 1, sparsity, 3) : getScalar(3);
			writeInputMatrixWithMTD("C", C, true);
			
			//run test cases
			runTest(true, false, null, -1);
			runRScript(true); 
			
			//compare output matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("R");
			TestUtils.compareMatrices(dmlfile, rfile, 0, "Stat-DML", "Stat-R");
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewritesOld;
		}
	}
	
	private static double[][] getScalar(int input) {
		return new double[][]{{7d*input}};
	}
}
