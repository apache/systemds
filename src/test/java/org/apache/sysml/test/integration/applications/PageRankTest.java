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

package org.apache.sysml.test.integration.applications;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import org.junit.runners.Parameterized.Parameters;

import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

public abstract class PageRankTest extends AutomatedTestBase {

	protected final static String TEST_DIR = "applications/page_rank/";
	protected final static String TEST_NAME = "PageRank";
	protected String TEST_CLASS_DIR = TEST_DIR + PageRankTest.class.getSimpleName() + "/";

	protected int numRows, numCols;

	public PageRankTest(int rows, int cols) {
		this.numRows = rows;
		this.numCols = cols;
	}

	@Parameters
	public static Collection<Object[]> data() {
		Object[][] data = new Object[][] { { 50, 50 }, { 1500, 1500 }, { 7500, 7500 } };
		return Arrays.asList(data);
	}

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "p" }));
	}

	protected void testPageRank(ScriptType scriptType) {
		System.out.println("------------ BEGIN " + TEST_NAME + " " + scriptType + " TEST {" + numRows + ", "
				+ numCols + "} ------------");
		this.scriptType = scriptType;

		int rows = numRows;
		int cols = numCols;
		int maxiter = 2;
		double alpha = 0.85;

		getAndLoadTestConfiguration(TEST_NAME);
		
		List<String> proArgs = new ArrayList<String>();
		if (scriptType == ScriptType.PYDML) {
			proArgs.add("-python");
		}
		proArgs.add("-args");
		proArgs.add(input("g"));
		proArgs.add(input("p"));
		proArgs.add(input("e"));
		proArgs.add(input("u"));
		proArgs.add(Double.toString(alpha));
		proArgs.add(Integer.toString(maxiter));
		proArgs.add(output("p"));
		programArgs = proArgs.toArray(new String[proArgs.size()]);
		
		fullDMLScriptName = getScript();

		double[][] g = getRandomMatrix(rows, cols, 1, 1, 0.000374962, -1);
		double[][] p = getRandomMatrix(rows, 1, 1, 1, 1, -1);
		double[][] e = getRandomMatrix(rows, 1, 1, 1, 1, -1);
		double[][] u = getRandomMatrix(1, cols, 1, 1, 1, -1);
		writeInputMatrixWithMTD("g", g, true);
		writeInputMatrixWithMTD("p", p, true);
		writeInputMatrixWithMTD("e", e, true);
		writeInputMatrixWithMTD("u", u, true);

		for (int i = 0; i < maxiter; i++) {
			double[][] gp = TestUtils.performMatrixMultiplication(g, p);
			double[][] eu = TestUtils.performMatrixMultiplication(e, u);
			double[][] eup = TestUtils.performMatrixMultiplication(eu, p);
			for (int j = 0; j < rows; j++) {
				p[j][0] = alpha * gp[j][0] + (1 - alpha) * eup[j][0];
			}
		}

		writeExpectedMatrix("p", p);

		/*
		 * Expected number of jobs: Reblock - 1 job While loop iteration - 6 jobs (can be reduced if MMCJ job can
		 * produce multiple outputs) Final output write - 1 job
		 */
		int expectedNumberOfJobs = 8;
		runTest(true, EXCEPTION_NOT_EXPECTED, null, expectedNumberOfJobs);

		compareResults(0.0000001);
	}

}
