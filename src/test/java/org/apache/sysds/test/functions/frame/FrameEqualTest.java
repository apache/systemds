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

package org.apache.sysds.test.functions.frame;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.FileFormat;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.HashMap;

public class FrameEqualTest extends AutomatedTestBase {
	private final static String TEST_NAME = "frameComparisonTest";
	private final static String TEST_DIR = "functions/binary/frame/";
	private static final String TEST_CLASS_DIR = TEST_DIR + FrameEqualTest.class.getSimpleName() + "/";

	private final static int rows = 100;
	private final static Types.ValueType[] schemaStrings1 = {Types.ValueType.FP64, Types.ValueType.BOOLEAN, Types.ValueType.INT64, Types.ValueType.STRING, Types.ValueType.STRING, Types.ValueType.FP64};
	private final static Types.ValueType[] schemaStrings2 = {Types.ValueType.INT64, Types.ValueType.BOOLEAN, Types.ValueType.FP32, Types.ValueType.FP64, Types.ValueType.STRING, Types.ValueType.FP32};

	public enum TestType {
		GREATER, LESS, EQUALS, NOT_EQUALS, GREATER_EQUALS, LESS_EQUALS,
	}

	@BeforeClass
	public static void init() {
		TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
	}

	@AfterClass
	public static void cleanUp() {
		if (TEST_CACHE_ENABLED) {
			TestUtils.clearDirectory(TEST_DATA_DIR + TEST_CLASS_DIR);
		}
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"D"}));
		if (TEST_CACHE_ENABLED) {
			setOutAndExpectedDeletionDisabled(true);
		}
	}

	@Test
	public void testFrameEqualCP() {
		runComparisonTest(schemaStrings1, schemaStrings2, rows, schemaStrings1.length, TestType.EQUALS, ExecType.CP);
	}

	@Test
	public void testFrameEqualSpark() {
		runComparisonTest(schemaStrings1, schemaStrings2, rows, schemaStrings1.length, TestType.EQUALS, ExecType.SPARK);
	}

	@Test
	public void testFrameNotEqualCP() {
		runComparisonTest(schemaStrings1, schemaStrings2, rows, schemaStrings1.length, TestType.NOT_EQUALS, ExecType.CP);
	}

	@Test
	public void testFrameNotEqualSpark() {
		runComparisonTest(schemaStrings1, schemaStrings2, rows, schemaStrings1.length, TestType.NOT_EQUALS, ExecType.SPARK);
	}

	@Test
	public void testFrameLessThanCP() {
		runComparisonTest(schemaStrings1, schemaStrings2, rows, schemaStrings1.length, TestType.LESS, ExecType.CP);
	}

	@Test
	public void testFrameLessThanSpark() {
		runComparisonTest(schemaStrings1, schemaStrings2, rows, schemaStrings1.length, TestType.LESS, ExecType.SPARK);
	}

	@Test
	public void testFrameGreaterEqualsCP() {
		runComparisonTest(schemaStrings1, schemaStrings2, rows, schemaStrings1.length, TestType.GREATER_EQUALS, ExecType.CP);
	}

	@Test
	public void testFrameGreaterEqualsSpark() {
		runComparisonTest(schemaStrings1, schemaStrings2, rows, schemaStrings1.length, TestType.GREATER_EQUALS, ExecType.SPARK);
	}

	@Test 
	public void testFrameLessEqualsCP() {
		runComparisonTest(schemaStrings1, schemaStrings2, rows, schemaStrings1.length, TestType.LESS_EQUALS, ExecType.CP);
	}

	@Test 
	public void testFrameLessEqualsSpark() {
		runComparisonTest(schemaStrings1, schemaStrings2, rows, schemaStrings1.length, TestType.LESS_EQUALS, ExecType.SPARK);
	}

	@Test
	public void testFrameGreaterThanCP() {
		runComparisonTest(schemaStrings1, schemaStrings2, rows, schemaStrings1.length, TestType.GREATER, ExecType.CP);
	}

	@Test
	public void testFrameGreaterThanSpark() {
		runComparisonTest(schemaStrings1, schemaStrings2, rows, schemaStrings1.length, TestType.GREATER, ExecType.SPARK);
	}

	private void runComparisonTest(Types.ValueType[] schema1, Types.ValueType[] schema2, int rows, int cols,
			TestType type, ExecType et)
	{
		Types.ExecMode platformOld = setExecMode(et);
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		//setOutputBuffering(true);
		try {
			getAndLoadTestConfiguration(TEST_NAME);

			double[][] A = getRandomMatrix(rows, cols, 2, 3, 1, 2);
			double[][] B = getRandomMatrix(rows, cols, 10, 20, 1, 0);

			writeInputFrameWithMTD("A", A, true, schemaStrings1, FileFormat.BINARY);
			writeInputFrameWithMTD("B", B, true, schemaStrings2, FileFormat.BINARY);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-explain", "recompile_runtime", "-nvargs", "A=" + input("A"), "B=" + input("B"),
					"rows=" + String.valueOf(rows), "cols=" + Integer.toString(cols), "type=" + String.valueOf(type), "C=" + output("C")};

			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + String.valueOf(type) + " " + expectedDir();

			runTest(null);
			runRScript(true);

			//compare matrices
			HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("C");
			HashMap<MatrixValue.CellIndex, Double> rfile = readRMatrixFromExpectedDir("C");

			double eps = 0.0001;
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
		}
		catch (Exception ex) {
			throw new RuntimeException(ex);
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
			OptimizerUtils.ALLOW_AUTO_VECTORIZATION = true;
			OptimizerUtils.ALLOW_OPERATOR_FUSION = true;
		}
	}
}
