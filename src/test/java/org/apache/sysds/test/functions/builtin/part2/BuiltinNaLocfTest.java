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

import org.apache.commons.lang.ArrayUtils;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.lineage.Lineage;
import org.apache.sysds.runtime.lineage.LineageCacheConfig.ReuseCacheType;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.util.HashMap;

public class BuiltinNaLocfTest extends AutomatedTestBase {
	private final static String TEST_NAME = "na_locfTest";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinNaLocfTest.class.getSimpleName() + "/";

	private final static double eps = 1e-10;
	private final static int rows = 25;
	private final static int cols = 25;

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"O"}));
	}

	@Test
	public void tesLocfNoLineageCP() {
		runLocfTest(false, "locf", ExecType.CP);
	}

	@Test
	public void tesLocfLineageCP() {
		runLocfTest(true, "locf", ExecType.CP);
	}

	@Test
	public void tesLocfNoLineageSPARK() {
		runLocfTest(false,"locf",  ExecType.SPARK);
	}

	@Test
	public void tesLocfLineageSPARK() {
		runLocfTest(true,"locf",  ExecType.SPARK);
	}

	@Test
	public void tesnocbNoLineageCP() {
		runLocfTest(false, "nocb", ExecType.CP);
	}

	@Test
	public void tesnocbLineageCP() {
		runLocfTest(true, "nocb", ExecType.CP);
	}

	@Test
	public void tesnocbNoLineageSPARK() {
		runLocfTest(false,"nocb",  ExecType.SPARK);
	}

	@Test
	public void tesnocbLineageSPARK() {
		runLocfTest(true,"nocb",  ExecType.SPARK);
	}

	private void runLocfTest(boolean lineage, String option, ExecType instType) {
		Types.ExecMode platformOld = setExecMode(instType);
		try {
			
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-nvargs", "X=" + input("A"), "option="+option, "O=" + output("O")};
			if(lineage) {
				String[] lin = new String[] {"-stats", "-lineage", ReuseCacheType.REUSE_HYBRID.name().toLowerCase()};
				programArgs = (String[]) ArrayUtils.addAll(programArgs, lin);
			}

			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = getRCmd(inputDir(), option, expectedDir());

			//generate actual dataset
			double[][] A = getRandomMatrix(rows, cols, -10, 10, 0.6, 7);
			writeInputMatrixWithMTD("A", A, true);

			Lineage.resetInternalState();
			runTest(true, false, null, -1);
			runRScript(true);
			//compare matrices
			HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("O");
			HashMap<MatrixValue.CellIndex, Double> rfile = readRMatrixFromExpectedDir("O");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
