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

package org.apache.sysds.test.functions.builtin.part1;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import java.util.HashMap;

public class BuiltinDbscanApplyTest extends AutomatedTestBase
{
	private final static String TEST_NAME = "dbscanApply";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinDbscanApplyTest.class.getSimpleName() + "/";

	private final static double eps = 1e-9;
	private final static int rows = 1700;
	private final static int cols = 3;
	private final static int min = -10;
	private final static int max = 10;

	private final static int minPts = 5;

	@Override
	public void setUp() { 
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"B"}));
	}

	@Test
	public void testDBSCANOutlierDefault0CP() {
		runOutlierByDBSCAN(true, 6, 18, 1, ExecType.CP);
	}

	@Test
	public void testDBSCANOutlierDefault0SP() {
		runOutlierByDBSCAN(true, 6, 18, 1, ExecType.SPARK);
	}

	@Test
	public void testDBSCANOutlierDefault1CP() {
		runOutlierByDBSCAN(true, 5, 15, 1, ExecType.CP);
	}

	@Test
	public void testDBSCANOutlierDefault1SP() {
		runOutlierByDBSCAN(true, 5, 15, 1, ExecType.SPARK);
	}

	@Test
	public void testDBSCANOutlierDefault2CP() {
		runOutlierByDBSCAN(true, 12, 77, 1, ExecType.CP);
	}

	@Test
	public void testDBSCANOutlierDefault2SP() {
		runOutlierByDBSCAN(true, 12, 77, 1, ExecType.SPARK);
	}

	private void runOutlierByDBSCAN(boolean defaultProb, int seedA, int seedB, double epsDB, ExecType instType)
	{
		ExecMode platformOld = setExecMode(instType);

		try
		{
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;

			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain","-nvargs",
				"X=" + input("A"), "Y=" + input("B"),"Z=" + output("C"), "eps=" + epsDB, "minPts=" + minPts};
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = getRCmd(inputDir(), inputDir(), Double.toString(epsDB), Integer.toString(minPts), expectedDir());

			//generate actual dataset
			double[][] A = getNonZeroRandomMatrix(rows, cols, min, max, seedA);
			writeInputMatrixWithMTD("A", A, true);
			double[][] B = getNonZeroRandomMatrix(rows, cols, min, max, seedB);
			writeInputMatrixWithMTD("B", B, true);
			
			runTest(true, false, null, -1);
			runRScript(true);

			//compare matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("C");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("C");

			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
