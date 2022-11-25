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
package org.apache.sysds.test.functions.linearization;

import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public class DagLinearizationTest extends AutomatedTestBase {

	private static final Log LOG = LogFactory.getLog(DagLinearizationTest.class.getName());

	private final String testNames[] = {"matrixmult_dag_linearization", "csplineCG_dag_linearization",
		"linear_regression_dag_linearization"};

	private final String testConfigs[] = {"breadth-first", "depth-first", "min-intermediate", "max-parallelize"};

	private final String testDir = "functions/linearization/";

	@Override
	public void setUp() {
		setOutputBuffering(true);
		disableConfigFile = true;
		TestUtils.clearAssertionInformation();
		for(String testname : testNames) {
			addTestConfiguration(testname, new TestConfiguration(testDir, testname));
		}
	}

	private String getPath(String filename) {
		return SCRIPT_DIR + "/" + testDir + filename;
	}

	@Test
	public void testMatrixMultSameOutput() {
		try {
			fullDMLScriptName = getPath("MatrixMult.dml");
			loadTestConfiguration(getTestConfiguration(testNames[0]));

			// Default arguments
			programArgs = new String[] {"-config", "", "-args", output("totalResult")};

			run(0, "totalResult");
		}
		catch(Exception ex) {
			ex.printStackTrace();
			fail("Exception in execution: " + ex.getMessage());
		}
	}

	@Test
	public void testCSplineCGSameOutput() {
		try {
			int rows = 10;
			int cols = 1;
			int numIter = rows;

			loadTestConfiguration(getTestConfiguration(testNames[1]));

			List<String> proArgs = new ArrayList<>();
			proArgs.add("-config");
			proArgs.add("");
			proArgs.add("-nvargs");
			proArgs.add("X=" + input("X"));
			proArgs.add("Y=" + input("Y"));
			proArgs.add("K=" + output("K"));
			proArgs.add("O=" + output("pred_y"));
			proArgs.add("maxi=" + numIter);
			proArgs.add("inp_x=" + 4.5);

			fullDMLScriptName = SCRIPT_DIR + "applications/cspline/CsplineCG.dml";

			double[][] X = new double[rows][cols];

			// X axis is given in the increasing order
			for(int rid = 0; rid < rows; rid++)
				for(int cid = 0; cid < cols; cid++)
					X[rid][cid] = rid + 1;

			double[][] Y = getRandomMatrix(rows, cols, 0, 5, 1.0, -1);

			writeInputMatrixWithMTD("X", X, true);
			writeInputMatrixWithMTD("Y", Y, true);

			programArgs = proArgs.toArray(new String[proArgs.size()]);
			run(Math.pow(10, -5), "pred_y");

		}
		catch(Exception ex) {
			ex.printStackTrace();
			fail("Exception in execution: " + ex.getMessage());
		}
	}

	@Test
	public void testLinearRegressionSameOutput() {
		try {
			int rows = 100;
			int cols = 50;

			loadTestConfiguration(getTestConfiguration(testNames[2]));

			List<String> proArgs = new ArrayList<>();
			proArgs.add("-config");
			proArgs.add("");
			proArgs.add("-args");
			proArgs.add(input("v"));
			proArgs.add(input("y"));
			proArgs.add(Double.toString(Math.pow(10, -8)));
			proArgs.add(output("w"));

			fullDMLScriptName = SCRIPT_DIR + "applications/linear_regression/LinearRegression.dml";

			double[][] v = getRandomMatrix(rows, cols, 0, 1, 0.01, -1);
			double[][] y = getRandomMatrix(rows, 1, 1, 10, 1, -1);
			writeInputMatrixWithMTD("v", v, true);
			writeInputMatrixWithMTD("y", y, true);

			programArgs = proArgs.toArray(new String[proArgs.size()]);
			run(Math.pow(10, -10), "w");

		}
		catch(Exception ex) {
			ex.printStackTrace();
			fail("Exception in execution: " + ex.getMessage());
		}
	}

	private void run(double eps, String out) {
		programArgs[1] = getPath("SystemDS-config-default.xml");
		LOG.debug(runTest(null));
		HashMap<MatrixValue.CellIndex, Double> retDefault = readDMLMatrixFromOutputDir(out);

		for(String conf : testConfigs) {
			programArgs[1] = getPath("SystemDS-config-" + conf + ".xml");
			LOG.debug(runTest(null));
			HashMap<MatrixValue.CellIndex, Double> ret2 = readDMLMatrixFromOutputDir(out);
			TestUtils.compareMatrices(retDefault, ret2, eps, "default", conf);
		}
	}
}
