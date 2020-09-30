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

package org.apache.sysds.test.functions.binary.matrix_full_other;

import java.util.HashMap;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

/**
 * The main purpose of this test is to verify various input combinations for
 * matrix-scalar logical operations that internally translate to binary operations.
 *
 */
public class FullLogicalMatrixTest extends AutomatedTestBase
{

	private final static String TEST_NAME1 = "LogicalMatrixTest";
	private final static String TEST_DIR = "functions/binary/matrix_full_other/";
	private final static String TEST_CLASS_DIR = TEST_DIR + FullLogicalMatrixTest.class.getSimpleName() + "/";
	private final static double eps = 1e-10;

	private final static int rows1 = 1183;
	private final static int cols1 = 1032;

	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.01;

	public enum Type{
		GREATER,
		LESS,
		EQUALS,
		NOT_EQUALS,
		GREATER_EQUALS,
		LESS_EQUALS,
	}

	@Override
	public void setUp()
	{
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "C" }) );
		TestUtils.clearAssertionInformation();
		if (TEST_CACHE_ENABLED) {
			setOutAndExpectedDeletionDisabled(true);
		}
	}

	@BeforeClass
	public static void init()
	{
		TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
	}

	@AfterClass
	public static void cleanUp()
	{
		if (TEST_CACHE_ENABLED) {
			TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
		}
	}

	@Test
	public void testLogicalGreaterDenseDenseCP()
	{
		runLogicalTest(Type.GREATER, false, false, ExecType.CP);
	}

	@Test
	public void testLogicalGreaterDenseSparseCP()
	{
		runLogicalTest(Type.GREATER, false, true, ExecType.CP);
	}

	@Test
	public void testLogicalGreaterSparseDenseCP()
	{
		runLogicalTest(Type.GREATER, true, false, ExecType.CP);
	}

	@Test
	public void testLogicalGreaterSparseSparseCP()
	{
		runLogicalTest(Type.GREATER, true, true, ExecType.CP);
	}

	@Test
	public void testLogicalGreaterEqualsDenseDenseCP()
	{
		runLogicalTest(Type.GREATER_EQUALS, false, false, ExecType.CP);
	}

	@Test
	public void testLogicalGreaterEqualsDenseSparseCP()
	{
		runLogicalTest(Type.GREATER_EQUALS, false, true, ExecType.CP);
	}

	@Test
	public void testLogicalGreaterEqualsSparseDenseCP()
	{
		runLogicalTest(Type.GREATER_EQUALS, true, false, ExecType.CP);
	}

	@Test
	public void testLogicalGreaterEqualsSparseSparseCP()
	{
		runLogicalTest(Type.GREATER_EQUALS, true, true, ExecType.CP);
	}

	@Test
	public void testLogicalEqualsDenseDenseCP()
	{
		runLogicalTest(Type.EQUALS, false, false, ExecType.CP);
	}

	@Test
	public void testLogicalEqualsDenseSparseCP()
	{
		runLogicalTest(Type.EQUALS, false, true, ExecType.CP);
	}

	@Test
	public void testLogicalEqualsSparseDenseCP()
	{
		runLogicalTest(Type.EQUALS, true, false, ExecType.CP);
	}

	@Test
	public void testLogicalEqualsSparseSparseCP()
	{
		runLogicalTest(Type.EQUALS, true, true, ExecType.CP);
	}

	@Test
	public void testLogicalNotEqualsDenseDenseCP()
	{
		runLogicalTest(Type.NOT_EQUALS, false, false, ExecType.CP);
	}

	@Test
	public void testLogicalNotEqualsDenseSparseCP()
	{
		runLogicalTest(Type.NOT_EQUALS, false, true, ExecType.CP);
	}

	@Test
	public void testLogicalNotEqualsSparseDenseCP()
	{
		runLogicalTest(Type.NOT_EQUALS, true, false, ExecType.CP);
	}

	@Test
	public void testLogicalNotEqualsSparseSparseCP()
	{
		runLogicalTest(Type.NOT_EQUALS, true, true, ExecType.CP);
	}

	@Test
	public void testLogicalLessDenseDenseCP()
	{
		runLogicalTest(Type.LESS, false, false, ExecType.CP);
	}

	@Test
	public void testLogicalLessDenseSparseCP()
	{
		runLogicalTest(Type.LESS, false, true, ExecType.CP);
	}

	@Test
	public void testLogicalLessSparseDenseCP()
	{
		runLogicalTest(Type.LESS, true, false, ExecType.CP);
	}

	@Test
	public void testLogicalLessSparseSparseCP()
	{
		runLogicalTest(Type.LESS, true, true, ExecType.CP);
	}

	@Test
	public void testLogicalLessEqualsDenseDenseCP()
	{
		runLogicalTest(Type.LESS_EQUALS, false, false, ExecType.CP);
	}

	@Test
	public void testLogicalLessEqualsDenseSparseCP()
	{
		runLogicalTest(Type.LESS_EQUALS, false, true, ExecType.CP);
	}

	@Test
	public void testLogicalLessEqualsSparseDenseCP()
	{
		runLogicalTest(Type.LESS_EQUALS, true, false, ExecType.CP);
	}

	@Test
	public void testLogicalLessEqualsSparseSparseCP()
	{
		runLogicalTest(Type.LESS_EQUALS, true, true, ExecType.CP);
	}


	// ------------------------
	@Test
	public void testLogicalGreaterDenseDenseSP()
	{
		runLogicalTest(Type.GREATER, false, false, ExecType.SPARK);
	}

	@Test
	public void testLogicalGreaterDenseSparseSP()
	{
		runLogicalTest(Type.GREATER, false, true, ExecType.SPARK);
	}

	@Test
	public void testLogicalGreaterSparseDenseSP()
	{
		runLogicalTest(Type.GREATER, true, false, ExecType.SPARK);
	}

	@Test
	public void testLogicalGreaterSparseSparseSP()
	{
		runLogicalTest(Type.GREATER, true, true, ExecType.SPARK);
	}

	@Test
	public void testLogicalGreaterEqualsDenseDenseSP()
	{
		runLogicalTest(Type.GREATER_EQUALS, false, false, ExecType.SPARK);
	}

	@Test
	public void testLogicalGreaterEqualsDenseSparseSP()
	{
		runLogicalTest(Type.GREATER_EQUALS, false, true, ExecType.SPARK);
	}

	@Test
	public void testLogicalGreaterEqualsSparseDenseSP()
	{
		runLogicalTest(Type.GREATER_EQUALS, true, false, ExecType.SPARK);
	}

	@Test
	public void testLogicalGreaterEqualsSparseSparseSP()
	{
		runLogicalTest(Type.GREATER_EQUALS, true, true, ExecType.SPARK);
	}

	@Test
	public void testLogicalEqualsDenseDenseSP()
	{
		runLogicalTest(Type.EQUALS, false, false, ExecType.SPARK);
	}

	@Test
	public void testLogicalEqualsDenseSparseSP()
	{
		runLogicalTest(Type.EQUALS, false, true, ExecType.SPARK);
	}

	@Test
	public void testLogicalEqualsSparseDenseSP()
	{
		runLogicalTest(Type.EQUALS, true, false, ExecType.SPARK);
	}

	@Test
	public void testLogicalEqualsSparseSparseSP()
	{
		runLogicalTest(Type.EQUALS, true, true, ExecType.SPARK);
	}

	@Test
	public void testLogicalNotEqualsDenseDenseSP()
	{
		runLogicalTest(Type.NOT_EQUALS, false, false, ExecType.SPARK);
	}

	@Test
	public void testLogicalNotEqualsDenseSparseSP()
	{
		runLogicalTest(Type.NOT_EQUALS, false, true, ExecType.SPARK);
	}

	@Test
	public void testLogicalNotEqualsSparseDenseSP()
	{
		runLogicalTest(Type.NOT_EQUALS, true, false, ExecType.SPARK);
	}

	@Test
	public void testLogicalNotEqualsSparseSparseSP()
	{
		runLogicalTest(Type.NOT_EQUALS, true, true, ExecType.SPARK);
	}

	@Test
	public void testLogicalLessDenseDenseSP()
	{
		runLogicalTest(Type.LESS, false, false, ExecType.SPARK);
	}

	@Test
	public void testLogicalLessDenseSparseSP()
	{
		runLogicalTest(Type.LESS, false, true, ExecType.SPARK);
	}

	@Test
	public void testLogicalLessSparseDenseSP()
	{
		runLogicalTest(Type.LESS, true, false, ExecType.SPARK);
	}

	@Test
	public void testLogicalLessSparseSparseSP()
	{
		runLogicalTest(Type.LESS, true, true, ExecType.SPARK);
	}

	@Test
	public void testLogicalLessEqualsDenseDenseSP()
	{
		runLogicalTest(Type.LESS_EQUALS, false, false, ExecType.SPARK);
	}

	@Test
	public void testLogicalLessEqualsDenseSparseSP()
	{
		runLogicalTest(Type.LESS_EQUALS, false, true, ExecType.SPARK);
	}

	@Test
	public void testLogicalLessEqualsSparseDenseSP()
	{
		runLogicalTest(Type.LESS_EQUALS, true, false, ExecType.SPARK);
	}

	@Test
	public void testLogicalLessEqualsSparseSparseSP()
	{
		runLogicalTest(Type.LESS_EQUALS, true, true, ExecType.SPARK);
	}
	// ----------------------

	
	private void runLogicalTest( Type type, boolean sp1, boolean sp2, ExecType et )
	{
		String TEST_NAME = TEST_NAME1;
		int rows = rows1;
		int cols = cols1;

	    ExecMode platformOld = rtplatform;
		switch( et ){
			case SPARK: rtplatform = ExecMode.SPARK; break;
			default: rtplatform = ExecMode.HYBRID; break;
		}

		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
	    if( rtplatform == ExecMode.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		double sparsityLeft = sp1 ? sparsity2 : sparsity1;
		double sparsityRight = sp2 ? sparsity2 : sparsity1;

		String TEST_CACHE_DIR = "";
		if (TEST_CACHE_ENABLED) {
			TEST_CACHE_DIR = type.ordinal() + "_" + rows + "_" + cols + "_" + sparsityLeft + "_" + sparsityRight + "/";
		}

		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config, TEST_CACHE_DIR);

			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", input("A"), input("B"),
				Integer.toString(type.ordinal()), output("C") };

			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + type.ordinal() + " " + expectedDir();

			//generate actual dataset
			double[][] A = getRandomMatrix(rows, cols, -10, 10, sparsityLeft, 7);
			writeInputMatrixWithMTD("A", A, true);
			double[][] B = getRandomMatrix(rows, cols, -15, 15, sparsityRight, 3);
			writeInputMatrixWithMTD("B", B, true);

			//run tests
			
			runTest(true, false, null, -1);
			runRScript(true);

			//compare matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("C");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("C");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
		}
		finally
		{
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}