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
import org.apache.sysds.common.Types.ExecType;
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
	
	private enum MatType {
		MATRIX, COL, ROW, SCALAR
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "R" }) );
	}

	@Test
	public void testScalarScalarScalarDenseCP() {
		runIfElseTest(MatType.SCALAR, MatType.SCALAR, MatType.SCALAR, false, ExecType.CP);
	}
	
	@Test
	public void testMatrixScalarScalarDenseCP() {
		runIfElseTest(MatType.MATRIX, MatType.SCALAR, MatType.SCALAR, false, ExecType.CP);
	}
	
	@Test
	public void testScalarMatrixScalarDenseCP() {
		runIfElseTest(MatType.SCALAR, MatType.MATRIX, MatType.SCALAR, false, ExecType.CP);
	}
	
	@Test
	public void testMatrixMatrixScalarDenseCP() {
		runIfElseTest(MatType.MATRIX, MatType.MATRIX, MatType.SCALAR, false, ExecType.CP);
	}
	
	@Test
	public void testScalarScalarMatrixDenseCP() {
		runIfElseTest(MatType.SCALAR, MatType.SCALAR, MatType.MATRIX, false, ExecType.CP);
	}
	
	@Test
	public void testMatrixScalarMatrixDenseCP() {
		runIfElseTest(MatType.MATRIX, MatType.SCALAR, MatType.MATRIX, false, ExecType.CP);
	}
	
	@Test
	public void testScalarMatrixMatrixDenseCP() {
		runIfElseTest(MatType.SCALAR, MatType.MATRIX, MatType.MATRIX, false, ExecType.CP);
	}
	
	@Test
	public void testMatrixMatrixMatrixDenseCP() {
		runIfElseTest(MatType.MATRIX, MatType.MATRIX, MatType.MATRIX, false, ExecType.CP);
	}

	@Test
	public void testScalarScalarScalarSparseCP() {
		runIfElseTest(MatType.SCALAR, MatType.SCALAR, MatType.SCALAR, true, ExecType.CP);
	}
	
	@Test
	public void testMatrixScalarScalarSparseCP() {
		runIfElseTest(MatType.MATRIX, MatType.SCALAR, MatType.SCALAR, true, ExecType.CP);
	}
	
	@Test
	public void testScalarMatrixScalarSparseCP() {
		runIfElseTest(MatType.SCALAR, MatType.MATRIX, MatType.SCALAR, true, ExecType.CP);
	}
	
	@Test
	public void testMatrixMatrixScalarSparseCP() {
		runIfElseTest(MatType.MATRIX, MatType.MATRIX, MatType.SCALAR, true, ExecType.CP);
	}
	
	@Test
	public void testScalarScalarMatrixSparseCP() {
		runIfElseTest(MatType.SCALAR, MatType.SCALAR, MatType.MATRIX, true, ExecType.CP);
	}
	
	@Test
	public void testMatrixScalarMatrixSparseCP() {
		runIfElseTest(MatType.MATRIX, MatType.SCALAR, MatType.MATRIX, true, ExecType.CP);
	}
	
	@Test
	public void testScalarMatrixMatrixSparseCP() {
		runIfElseTest(MatType.SCALAR, MatType.MATRIX, MatType.MATRIX, true, ExecType.CP);
	}
	
	@Test
	public void testMatrixMatrixMatrixSparseCP() {
		runIfElseTest(MatType.MATRIX, MatType.MATRIX, MatType.MATRIX, true, ExecType.CP);
	}

	//SPARK
	
	@Test
	public void testScalarScalarScalarDenseSP() {
		runIfElseTest(MatType.SCALAR, MatType.SCALAR, MatType.SCALAR, false, ExecType.SPARK);
	}
	
	@Test
	public void testMatrixScalarScalarDenseSP() {
		runIfElseTest(MatType.MATRIX, MatType.SCALAR, MatType.SCALAR, false, ExecType.SPARK);
	}
	
	@Test
	public void testScalarMatrixScalarDenseSP() {
		runIfElseTest(MatType.SCALAR, MatType.MATRIX, MatType.SCALAR, false, ExecType.SPARK);
	}
	
	@Test
	public void testMatrixMatrixScalarDenseSP() {
		runIfElseTest(MatType.MATRIX, MatType.MATRIX, MatType.SCALAR, false, ExecType.SPARK);
	}
	
	@Test
	public void testScalarScalarMatrixDenseSP() {
		runIfElseTest(MatType.SCALAR, MatType.SCALAR, MatType.MATRIX, false, ExecType.SPARK);
	}
	
	@Test
	public void testMatrixScalarMatrixDenseSP() {
		runIfElseTest(MatType.MATRIX, MatType.SCALAR, MatType.MATRIX, false, ExecType.SPARK);
	}
	
	@Test
	public void testScalarMatrixMatrixDenseSP() {
		runIfElseTest(MatType.SCALAR, MatType.MATRIX, MatType.MATRIX, false, ExecType.SPARK);
	}
	
	@Test
	public void testMatrixMatrixMatrixDenseSP() {
		runIfElseTest(MatType.MATRIX, MatType.MATRIX, MatType.MATRIX, false, ExecType.SPARK);
	}

	@Test
	public void testScalarScalarScalarSparseSP() {
		runIfElseTest(MatType.SCALAR, MatType.SCALAR, MatType.SCALAR, true, ExecType.SPARK);
	}
	
	@Test
	public void testMatrixScalarScalarSparseSP() {
		runIfElseTest(MatType.MATRIX, MatType.SCALAR, MatType.SCALAR, true, ExecType.SPARK);
	}
	
	@Test
	public void testScalarMatrixScalarSparseSP() {
		runIfElseTest(MatType.SCALAR, MatType.MATRIX, MatType.SCALAR, true, ExecType.SPARK);
	}
	
	@Test
	public void testMatrixMatrixScalarSparseSP() {
		runIfElseTest(MatType.MATRIX, MatType.MATRIX, MatType.SCALAR, true, ExecType.SPARK);
	}
	
	@Test
	public void testScalarScalarMatrixSparseSP() {
		runIfElseTest(MatType.SCALAR, MatType.SCALAR, MatType.MATRIX, true, ExecType.SPARK);
	}
	
	@Test
	public void testMatrixScalarMatrixSparseSP() {
		runIfElseTest(MatType.MATRIX, MatType.SCALAR, MatType.MATRIX, true, ExecType.SPARK);
	}
	
	@Test
	public void testScalarMatrixMatrixSparseSP() {
		runIfElseTest(MatType.SCALAR, MatType.MATRIX, MatType.MATRIX, true, ExecType.SPARK);
	}
	
	@Test
	public void testMatrixMatrixMatrixSparseSP() {
		runIfElseTest(MatType.MATRIX, MatType.MATRIX, MatType.MATRIX, true, ExecType.SPARK);
	}

	private void runIfElseTest(MatType mtype1, MatType mtype2, MatType mtype3, boolean sparse, ExecType et){
		setOutputBuffering(true);
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
			double[][] A = getMatrixOfType(mtype1, sparse, 1);
			writeInputMatrixWithMTD("A", A, true);
			double[][] B = getMatrixOfType(mtype2, sparse, 2);
			writeInputMatrixWithMTD("B", B, true);
			double[][] C = getMatrixOfType(mtype2, sparse, 3);
			writeInputMatrixWithMTD("C", C, true);
			
			//run test cases
			runTest(null);
			runRScript(true); 
			
			//compare output matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("R");
			TestUtils.compareMatrices(dmlfile, rfile, 0, "Stat-DML", "Stat-R");
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewritesOld;
		}
	}
	
	private double[][] getMatrixOfType(MatType mtype, boolean sparse, long seed) {
		double[][] ret = null;
		double sparsity = sparse ? sparsity2 : sparsity1;
		switch(mtype) {
			case SCALAR:
				ret = getRandomMatrix(1, 1, 0, 1, sparsity, seed);
				break;
			case MATRIX:
				ret = getRandomMatrix(rows, cols, 0, 1, sparsity, seed);
				break;
			case COL:
				ret = getRandomMatrix(1, cols, 0, 1, sparsity, seed);
				break;
			case ROW:
				ret = getRandomMatrix(rows, 1, 0, 1, sparsity, seed);
				break;
			default:
		}
		return ret;
	}
}
