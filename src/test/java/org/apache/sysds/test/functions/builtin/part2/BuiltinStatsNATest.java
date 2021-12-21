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

package org.apache.sysds.test.functions.builtin.part2;

import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.util.HashMap;

public class BuiltinStatsNATest extends AutomatedTestBase {
	private final static String TEST_NAME = "statsNATest";
	private final static String TEST_DIR = "functions/builtin/";
	private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinStatsNATest.class.getSimpleName() + "/";
	private final static double eps = 1e-3;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"B",}));
	}

	@Test
	public void testStatsNA1() {
		runStatsNA(1, 100, ExecType.CP);
	}

	@Test
	public void testStatsNA2() {
		runStatsNA(4, 100, ExecType.CP);
	}

	@Test
	public void testStatsNA3() {
		runStatsNA(100, 1000, ExecType.CP);
	}

	@Test
	public void testStatsNA4() {
		runStatsNA(100, 10000, ExecType.CP);
	}


	private void runStatsNA(int bins, int size, ExecType instType) {
		Types.ExecMode platformOld = setExecMode(instType);
		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{ "-nvargs", "X=" + input("A"), "bins=" + bins, "Out=" + output("Out")};

			double[][] A = getRandomMatrix(size, 1, -10, 10, 0.6, 7);
			writeInputMatrixWithMTD("A", A, true);

			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = getRCmd(inputDir(), Integer.toString(bins), expectedDir());

			runTest(true, false, null, -1);
			runRScript(true);
			//compare matrices
			HashMap<MatrixValue.CellIndex, Double> dmlfileOut1 = readDMLMatrixFromOutputDir("Out");
			HashMap<MatrixValue.CellIndex, Double> rfileOut1 = readRMatrixFromExpectedDir("Out");
			TestUtils.compareMatrices(dmlfileOut1, rfileOut1, eps, "Stat-DML", "Stat-R");
		}
		catch(Exception e) {
			e.printStackTrace();
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
