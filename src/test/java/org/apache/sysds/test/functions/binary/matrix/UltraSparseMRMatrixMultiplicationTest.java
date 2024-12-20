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

package org.apache.sysds.test.functions.binary.matrix;

import java.util.HashMap;

import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.AggBinaryOp;
import org.apache.sysds.hops.AggBinaryOp.MMultMethod;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.controlprogram.ProgramBlock;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

/**
 * Test for MMCJ MR because otherwise seldom (if at all) executed in our testsuite, ultrasparse 
 * in order to account for 'empty block rejection' optimization.
 * 
 * Furthermore, it is at the same time a test for removeEmpty-diag which has special
 * physical operators.
 */
public class UltraSparseMRMatrixMultiplicationTest extends AutomatedTestBase 
{
	
	private final static String TEST_NAME1 = "UltraSparseMatrixMultiplication";
	private final static String TEST_NAME2 = "UltraSparseMatrixMultiplication2";
	private final static String TEST_DIR = "functions/binary/matrix/";
	private final static String TEST_CLASS_DIR = TEST_DIR + 
		UltraSparseMRMatrixMultiplicationTest.class.getSimpleName() + "/";
	private final static double eps = 1e-10;
	
	private final static int rows = 4045; 
	private final static int cols = 23;
	
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.1;
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME1, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "C" }) ); 
		addTestConfiguration(TEST_NAME2, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "C" }) ); 
	}

	@Test
	public void testMMRowDenseCP() 
	{
		runMatrixMatrixMultiplicationTest(false, false, ExecType.CP, true, false);
	}
	
	@Test
	public void testMMRowSparseCP() 
	{
		runMatrixMatrixMultiplicationTest(false, true, ExecType.CP, true, false);
	}
	
	@Test
	public void testMMColDenseCP() 
	{
		runMatrixMatrixMultiplicationTest(false, false, ExecType.CP, false, false);
	}
	
	@Test
	public void testMMColSparseCP() 
	{
		runMatrixMatrixMultiplicationTest(false, true, ExecType.CP, false, false);
	}

	@Test
	public void testMMRowDenseSpark_PMMJ() 
	{
		runMatrixMatrixMultiplicationTest(false, false, ExecType.SPARK, true, true);
	}
	
	@Test
	public void testMMRowSparseSpark_PMMJ() 
	{
		runMatrixMatrixMultiplicationTest(false, true, ExecType.SPARK, true, true);
	}

	private void runMatrixMatrixMultiplicationTest( boolean sparseM1, boolean sparseM2, ExecType instType, boolean rowwise, boolean forcePMMJ)
	{
		//setup exec type, rows, cols

		//rtplatform for MR
		ExecMode platformOld = rtplatform;
		switch( instType ){
			case SPARK: rtplatform = ExecMode.SPARK; break;
			default: rtplatform = ExecMode.HYBRID; break;
		}
	
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == ExecMode.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		ProgramBlock.CHECK_MATRIX_PROPERTIES = true;
		
		if(forcePMMJ)
			AggBinaryOp.FORCED_MMULT_METHOD = MMultMethod.PMM;
			
		try
		{
			String TEST_NAME = (rowwise) ? TEST_NAME1 : TEST_NAME2;
			getAndLoadTestConfiguration(TEST_NAME);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", input("A"),
				Integer.toString(rows), Integer.toString(cols), input("B"), output("C") };
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();
	
			//generate actual dataset
			double[][] A = getRandomMatrix(rows, cols, 0, 1, sparseM1?sparsity2:sparsity1, 7); 
			writeInputMatrix("A", A, true);
			double[][] B = getRandomMatrix(rows, 1, 0.51, 3.49, 1.0, 3); 
			B = TestUtils.round(B);
			writeInputMatrix("B", B, true);
	
			boolean exceptionExpected = false;
			runTest(true, exceptionExpected, null, -1); 
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("C");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("C");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			AggBinaryOp.FORCED_MMULT_METHOD = null;
			ProgramBlock.CHECK_MATRIX_PROPERTIES = false;
		}
	}
}