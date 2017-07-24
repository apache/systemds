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

package org.apache.sysml.test.integration.functions.binary.matrix_full_other;

import java.util.HashMap;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

/**
 * The main purpose of this test is to verify the internal optimization
 * regarding sparse-safeness of logical operations for various input
 * combinations. (logical operations not sparse-safe in general, but for certain
 * instance involving 0 scalar they are).
 *
 * Furthermore, it is used to test all combinations of matrix-scalar,
 * scalar-matrix logical operations in all execution types.
 *
 */
public class FullLogicalScalarLeftTest extends AutomatedTestBase
{

	private final static String TEST_NAME1 = "LogicalScalarLeftTest";
	private final static String TEST_DIR = "functions/binary/matrix_full_other/";
	private final static String TEST_CLASS_DIR = TEST_DIR + FullLogicalScalarLeftTest.class.getSimpleName() + "/";
	private final static double eps = 1e-10;

	private final static int rows1 = 1072;
	private final static int cols1 = 1009;

	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.1;

	public enum Type{
		GREATER,
		LESS,
		EQUALS,
		NOT_EQUALS,
		GREATER_EQUALS,
		LESS_EQUALS,
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

	@Override
	public void setUp()
	{
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "B" })   );
		if (TEST_CACHE_ENABLED) {
			setOutAndExpectedDeletionDisabled(true);
		}
	}


	@Test
	public void testLogicalGreaterZeroDenseCP()
	{
		runLogicalTest(Type.GREATER, true, false, ExecType.CP);
	}

	@Test
	public void testLogicalLessZeroDenseCP()
	{
		runLogicalTest(Type.LESS, true, false, ExecType.CP);
	}

	@Test
	public void testLogicalEqualsZeroDenseCP()
	{
		runLogicalTest(Type.EQUALS, true, false, ExecType.CP);
	}

	@Test
	public void testLogicalNotEqualsZeroDenseCP()
	{
		runLogicalTest(Type.NOT_EQUALS, true, false, ExecType.CP);
	}

	@Test
	public void testLogicalGreaterEqualsZeroDenseCP()
	{
		runLogicalTest(Type.GREATER_EQUALS, true, false, ExecType.CP);
	}

	@Test
	public void testLogicalLessEqualsZeroDenseCP()
	{
		runLogicalTest(Type.LESS_EQUALS, true, false, ExecType.CP);
	}

	@Test
	public void testLogicalGreaterNonZeroDenseCP()
	{
		runLogicalTest(Type.GREATER, false, false, ExecType.CP);
	}

	@Test
	public void testLogicalLessNonZeroDenseCP()
	{
		runLogicalTest(Type.LESS, false, false, ExecType.CP);
	}

	@Test
	public void testLogicalEqualsNonZeroDenseCP()
	{
		runLogicalTest(Type.EQUALS, false, false, ExecType.CP);
	}

	@Test
	public void testLogicalNotEqualsNonZeroDenseCP()
	{
		runLogicalTest(Type.NOT_EQUALS, false, false, ExecType.CP);
	}

	@Test
	public void testLogicalGreaterEqualsNonZeroDenseCP()
	{
		runLogicalTest(Type.GREATER_EQUALS, false, false, ExecType.CP);
	}

	@Test
	public void testLogicalLessEqualsNonZeroDenseCP()
	{
		runLogicalTest(Type.LESS_EQUALS, false, false, ExecType.CP);
	}

	@Test
	public void testLogicalGreaterZeroSparseCP()
	{
		runLogicalTest(Type.GREATER, true, true, ExecType.CP);
	}

	@Test
	public void testLogicalLessZeroSparseCP()
	{
		runLogicalTest(Type.LESS, true, true, ExecType.CP);
	}

	@Test
	public void testLogicalEqualsZeroSparseCP()
	{
		runLogicalTest(Type.EQUALS, true, true, ExecType.CP);
	}

	@Test
	public void testLogicalNotEqualsZeroSparseCP()
	{
		runLogicalTest(Type.NOT_EQUALS, true, true, ExecType.CP);
	}

	@Test
	public void testLogicalGreaterEqualsZeroSparseCP()
	{
		runLogicalTest(Type.GREATER_EQUALS, true, true, ExecType.CP);
	}

	@Test
	public void testLogicalLessEqualsZeroSparseCP()
	{
		runLogicalTest(Type.LESS_EQUALS, true, true, ExecType.CP);
	}

	@Test
	public void testLogicalGreaterNonZeroSparseCP()
	{
		runLogicalTest(Type.GREATER, false, true, ExecType.CP);
	}

	@Test
	public void testLogicalLessNonZeroSparseCP()
	{
		runLogicalTest(Type.LESS, false, true, ExecType.CP);
	}

	@Test
	public void testLogicalEqualsNonZeroSparseCP()
	{
		runLogicalTest(Type.EQUALS, false, true, ExecType.CP);
	}

	@Test
	public void testLogicalNotEqualsNonZeroSparseCP()
	{
		runLogicalTest(Type.NOT_EQUALS, false, true, ExecType.CP);
	}

	@Test
	public void testLogicalGreaterEqualsNonZeroSparseCP()
	{
		runLogicalTest(Type.GREATER_EQUALS, false, true, ExecType.CP);
	}

	@Test
	public void testLogicalLessEqualsNonZeroSparseCP()
	{
		runLogicalTest(Type.LESS_EQUALS, false, true, ExecType.CP);
	}

	@Test
	public void testLogicalGreaterZeroDenseMR()
	{
		runLogicalTest(Type.GREATER, true, false, ExecType.MR);
	}

	@Test
	public void testLogicalLessZeroDenseMR()
	{
		runLogicalTest(Type.LESS, true, false, ExecType.MR);
	}

	@Test
	public void testLogicalEqualsZeroDenseMR()
	{
		runLogicalTest(Type.EQUALS, true, false, ExecType.MR);
	}

	@Test
	public void testLogicalNotEqualsZeroDenseMR()
	{
		runLogicalTest(Type.NOT_EQUALS, true, false, ExecType.MR);
	}

	@Test
	public void testLogicalGreaterEqualsZeroDenseMR()
	{
		runLogicalTest(Type.GREATER_EQUALS, true, false, ExecType.MR);
	}

	@Test
	public void testLogicalLessEqualsZeroDenseMR()
	{
		runLogicalTest(Type.LESS_EQUALS, true, false, ExecType.MR);
	}

	@Test
	public void testLogicalGreaterNonZeroDenseMR()
	{
		runLogicalTest(Type.GREATER, false, false, ExecType.MR);
	}

	@Test
	public void testLogicalLessNonZeroDenseMR()
	{
		runLogicalTest(Type.LESS, false, false, ExecType.MR);
	}

	@Test
	public void testLogicalEqualsNonZeroDenseMR()
	{
		runLogicalTest(Type.EQUALS, false, false, ExecType.MR);
	}

	@Test
	public void testLogicalNotEqualsNonZeroDenseMR()
	{
		runLogicalTest(Type.NOT_EQUALS, false, false, ExecType.MR);
	}

	@Test
	public void testLogicalGreaterEqualsNonZeroDenseMR()
	{
		runLogicalTest(Type.GREATER_EQUALS, false, false, ExecType.MR);
	}

	@Test
	public void testLogicalLessEqualsNonZeroDenseMR()
	{
		runLogicalTest(Type.LESS_EQUALS, false, false, ExecType.MR);
	}

	@Test
	public void testLogicalGreaterZeroSparseMR()
	{
		runLogicalTest(Type.GREATER, true, true, ExecType.MR);
	}

	@Test
	public void testLogicalLessZeroSparseMR()
	{
		runLogicalTest(Type.LESS, true, true, ExecType.MR);
	}

	@Test
	public void testLogicalEqualsZeroSparseMR()
	{
		runLogicalTest(Type.EQUALS, true, true, ExecType.MR);
	}

	@Test
	public void testLogicalNotEqualsZeroSparseMR()
	{
		runLogicalTest(Type.NOT_EQUALS, true, true, ExecType.MR);
	}

	@Test
	public void testLogicalGreaterEqualsZeroSparseMR()
	{
		runLogicalTest(Type.GREATER_EQUALS, true, true, ExecType.MR);
	}

	@Test
	public void testLogicalLessEqualsZeroSparseMR()
	{
		runLogicalTest(Type.LESS_EQUALS, true, true, ExecType.MR);
	}

	@Test
	public void testLogicalGreaterNonZeroSparseMR()
	{
		runLogicalTest(Type.GREATER, false, true, ExecType.MR);
	}

	@Test
	public void testLogicalLessNonZeroSparseMR()
	{
		runLogicalTest(Type.LESS, false, true, ExecType.MR);
	}

	@Test
	public void testLogicalEqualsNonZeroSparseMR()
	{
		runLogicalTest(Type.EQUALS, false, true, ExecType.MR);
	}

	@Test
	public void testLogicalNotEqualsNonZeroSparseMR()
	{
		runLogicalTest(Type.NOT_EQUALS, false, true, ExecType.MR);
	}

	@Test
	public void testLogicalGreaterEqualsNonZeroSparseMR()
	{
		runLogicalTest(Type.GREATER_EQUALS, false, true, ExecType.MR);
	}

	@Test
	public void testLogicalLessEqualsNonZeroSparseMR()
	{
		runLogicalTest(Type.LESS_EQUALS, false, true, ExecType.MR);
	}

	private void runLogicalTest( Type type, boolean zero, boolean sparse, ExecType et )
	{
		String TEST_NAME = TEST_NAME1;
		int rows = rows1;
		int cols = cols1;
		double sparsity = sparse ? sparsity2 : sparsity1;
		double constant = zero ? 0 : 0.5;

		String TEST_CACHE_DIR = "";
		if (TEST_CACHE_ENABLED) {
			TEST_CACHE_DIR = type.ordinal() + "_" + constant + "_" + sparsity + "/";
		}

		//rtplatform for MR
		RUNTIME_PLATFORM platformOld = rtplatform;
		rtplatform = (et==ExecType.MR) ? RUNTIME_PLATFORM.HADOOP : RUNTIME_PLATFORM.HYBRID;

		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);

			loadTestConfiguration(config, TEST_CACHE_DIR);

			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain","-args", input("A"),
				Integer.toString(type.ordinal()), Double.toString(constant), output("B") };

			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " +  inputDir() + " " +
				type.ordinal() + " " + constant + " " + expectedDir();

			//generate actual dataset
			double[][] A = getRandomMatrix(rows, cols, -1, 1, sparsity, 7);
			writeInputMatrixWithMTD("A", A, true);

			//run tests
			runTest(true, false, null, -1);
			runRScript(true);

			//compare matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("B");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("B");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
		}
		finally
		{
			rtplatform = platformOld;
		}
	}
}