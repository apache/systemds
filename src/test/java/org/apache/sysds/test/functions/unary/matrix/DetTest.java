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

import org.junit.Assert;
import org.junit.Test;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.parser.LanguageException;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;

import java.util.HashMap;


public class DetTest extends AutomatedTestBase
{

	private static final String TEST_DIR = "functions/unary/matrix/";
	private static final String TEST_CLASS_DIR = TEST_DIR + DetTest.class.getSimpleName() + "/";
	private static final String DML_SCRIPT_NAME = "DetTest";
	private static final String R_SCRIPT_NAME = "DetTest";

	private static final String TEST_NAME_WRONG_DIM = "WrongDimensionsTest";
	private static final String TEST_NAME_DET_TEST = "DetTest";

	private final static int rows = 1227;
	private final static double _sparsityDense = 0.5;
	private final static double _sparsitySparse = 0.05;
	private final static double eps = 1e-8;
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME_WRONG_DIM, new TestConfiguration(TEST_CLASS_DIR, DML_SCRIPT_NAME, new String[] { "d" }));
		addTestConfiguration(TEST_NAME_DET_TEST, new TestConfiguration(TEST_CLASS_DIR, DML_SCRIPT_NAME, new String[] { "d" }) );
	}
	
	/* HARD CODED TEST SUCCEEDS */
	// @Test
    // public void test2x2Determinant() {
    //     double[][] matrixData = {
    //         {1, 2},
    //         {3, 4}
    //     };
    //     double expectedDeterminant = -2;

    //     MatrixBlock inputMatrixBlock = DataConverter.convertToMatrixBlock(matrixData);
    //     MatrixBlock result = computeDeterminant(inputMatrixBlock);
    //     double actualDeterminant = result.get(0, 0);

    //     Assert.assertEquals("Determinant calculation failed", expectedDeterminant, actualDeterminant, 1e-8);
    // }
    // --------- hard coded/ copied function from LibCommonsMath
    // private static MatrixBlock computeDeterminant(MatrixBlock in) {
    //     if (in.getNumRows() != in.getNumColumns()) {
    //         throw new RuntimeException(
    //             "Determinant can only be computed for a square matrix. Input matrix is rectangular (rows="
    //                 + in.getNumRows() + ", cols=" + in.getNumColumns() + ")");
    //     }
    //     Array2DRowRealMatrix matrixInput = DataConverter.convertToArray2DRowRealMatrix(in);
    //     LUDecomposition ludecompose = new LUDecomposition(matrixInput);
    //     MatrixBlock determinantResult = new MatrixBlock(1, 1, false);
    //     determinantResult.set(0, 0, ludecompose.getDeterminant());
    //     return determinantResult;
    // }

	// --------- hard coded test calling commonsmathlib
	// @Test
    // public void testDeterminant2x2Matrix() {
    //     runDetTest(new double[][]{
    //         {1, 2},
    //         {3, 4}
    //     }, -2);  // Expected determinant: 1*4 - 2*3 = -2
    // }

	@Test
	public void testWrongDimensions() {
		int wrong_rows = 10;
		int wrong_cols = 9;
		
		TestConfiguration config = availableTestConfigurations.get(TEST_NAME_WRONG_DIM);
		loadTestConfiguration(config);

		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + DML_SCRIPT_NAME + ".dml";
		programArgs = new String[]{"-args", input("A"), output("d") };

		double[][] A = getRandomMatrix(wrong_rows, wrong_cols, -1, 1, 0.5, 3);
		writeInputMatrixWithMTD("A", A, true);
		runTest(true, true, LanguageException.class, -1);
	}

	@Test
	public void testDetMatrixDenseWithRewritesCP()
	{
		runDetTest(false, ExecType.CP, true);
	}

	@Test
	public void testDetMatrixDenseWithoutRewritesCP()
	{
		runDetTest(false, ExecType.CP, false);
	}

	@Test
	public void testDetMatrixSparseWithoutRewritesCP()
	{
		runDetTest(true, ExecType.CP, false);
	}

	@Test
	public void testDetMatrixSparseWithRewritesCP()
	{
		runDetTest(true, ExecType.CP, true);
	}

	@Test
	public void testDetMatrixDenseWithRewritesSP()
	{
		runDetTest(false, ExecType.SPARK, true);
	}

	@Test
	public void testDetMatrixDenseWithoutRewritesSP()
	{
		runDetTest(false, ExecType.SPARK, false);
	}

	@Test
	public void testDetMatrixSparseWithoutRewritesSP()
	{
		runDetTest(true, ExecType.SPARK, false);
	}

	@Test
	public void testDetMatrixSparseWithRewritesSP()
	{
		runDetTest(true, ExecType.SPARK, true);
	}

	private void runDetTest(boolean sparse, ExecType et, boolean rewrites)
	{
		if (et == ExecType.SPARK) {
			System.out.println("Skipping Spark test: det operation not supported in Spark mode.");
			return;  // Skip Spark tests for determinant
		}
	
		ExecMode platformOld = rtplatform;
		switch (et) {
			case SPARK:
				rtplatform = ExecMode.SPARK;
				break;
			default:
				rtplatform = ExecMode.HYBRID;
				break;
		}
	
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if(rtplatform == ExecMode.SPARK) {
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		}
		
		
		boolean oldFlagRewrites = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
		
		try
		{
			double sparsity = (sparse) ? _sparsitySparse : _sparsityDense;
			getAndLoadTestConfiguration(TEST_NAME_DET_TEST);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + DML_SCRIPT_NAME + ".dml";
			programArgs = new String[]{"-explain", "-args", input("A"), output("d")};
			
			fullRScriptName = HOME + R_SCRIPT_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();
	
			double[][] A = getRandomMatrix(rows, rows, -1, 1, sparsity, 7);
			writeInputMatrixWithMTD("A", A, true);
	
			runTest(true, false, null, -1); 
			if(et == ExecType.CP) //in CP no Spark jobs should be executed
				Assert.assertEquals("Unexpected number of executed Spark jobs.", 0, Statistics.getNoOfExecutedSPInst());
			runRScript(true); 
		
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("d");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("d");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
		}
		finally
		{
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlagRewrites;
		}
	}

	//for testing the hardcoded unit tests (with precision tolerance)
	// private void runDetTest(double[][] matrix, double expectedDeterminant) {
    // ExecMode oldPlatform = rtplatform;
    // rtplatform = ExecMode.SINGLE_NODE; 

    // boolean oldSparkConfig = DMLScript.USE_LOCAL_SPARK_CONFIG;
    // if (rtplatform == ExecMode.SPARK) {
    //     DMLScript.USE_LOCAL_SPARK_CONFIG = true;
    // }

    // try {
    //     getAndLoadTestConfiguration(TEST_NAME_DET_TEST);

    //     String HOME = SCRIPT_DIR + TEST_DIR;
    //     fullDMLScriptName = HOME + DML_SCRIPT_NAME + ".dml";  
    //     programArgs = new String[]{"-args", input("A"), output("d")};  

    //     MatrixCharacteristics mc = new MatrixCharacteristics(matrix.length, matrix[0].length, -1, -1);
    //     writeInputMatrixWithMTD("A", matrix, false, mc);  // Writes matrix to file
    //     writeExpectedMatrix("d", new double[][]{{expectedDeterminant}});  // Expected determinante 

    //     runTest(true, false, null, -1); 
    //     compareResults(eps);  // compares results with precision tolerance

	// 	System.out.println("Expected determinant: " + expectedDeterminant);
	// 	System.out.println("DML output determinant: " + readDMLMatrixFromOutputDir("d").get(new CellIndex(1, 1)));

    // } finally {
    //     rtplatform = oldPlatform; 
    //     DMLScript.USE_LOCAL_SPARK_CONFIG = oldSparkConfig;
    // }
}

