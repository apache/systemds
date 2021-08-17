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

package org.apache.sysds.test.functions.builtin;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import java.util.HashMap;

public class BuiltinDBSCANTest extends AutomatedTestBase
{
	private final static String TEST_NAME = "dbscan";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinDBSCANTest.class.getSimpleName() + "/";

	private final static double eps = 1e-3;
	private final static int rows = 1700;

	private final static double epsDBSCAN = 1;
	private final static int minPts = 5;

	@Override
	public void setUp() { 
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"B"}));
	}

	@Test
	public void testDBSCANDefaultCP() {
		runDBSCAN(true, ExecType.CP);
	}

	@Test
	public void testDBSCANDefaultSP() {
		runDBSCAN(true, ExecType.SPARK);
	}

	private void runDBSCAN(boolean defaultProb, ExecType instType)
	{
		ExecMode platformOld = setExecMode(instType);

		try
		{
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;

			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain","-nvargs",
				"X=" + input("A"), "Y=" + output("B"), "eps=" + epsDBSCAN, "minPts=" + minPts};
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = getRCmd(inputDir(), Double.toString(epsDBSCAN), Integer.toString(minPts), expectedDir());

			//generate actual dataset
			double[][] A = getNonZeroRandomMatrix(rows, 3, -10, 10, 7);
			writeInputMatrixWithMTD("A", A, true);

			runTest(true, false, null, -1);
			runRScript(true);

			//compare matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("B");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("B");

			//map cluster ids
			//NOTE: border points that are reachable from more than 1 cluster
			// are assigned to lowest point id, not cluster id -> can fail in this case, but it's still correct
			BiMap<Double, Double> merged = HashBiMap.create();
			rfile.forEach((key, value) -> merged.put(value, dmlfile.get(key)));
			dmlfile.replaceAll((k, v) -> merged.inverse().get(v));

			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
