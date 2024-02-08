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

package org.apache.sysds.test.applications;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class WelchTTest extends AutomatedTestBase {

	protected final static String TEST_DIR = "applications/welchTTest/";
	protected final static String TEST_NAME = "welchTTest";
	protected String TEST_CLASS_DIR = TEST_DIR + WelchTTest.class.getSimpleName() + "/";

	protected int numAttr, numPosSamples, numNegSamples;

	public WelchTTest(int numAttr, int numPosSamples, int numNegSamples) {
		this.numAttr = numAttr;
		this.numPosSamples = numPosSamples;
		this.numNegSamples = numNegSamples;
	}

	@Parameters
	public static Collection<Object[]> data() {
		Object[][] data = new Object[][] {{5, 100, 150}, {50, 2000, 1500}, {50, 7000, 1500}};
		return Arrays.asList(data);
	}

	@Override
	public void setUp() {
		addTestConfiguration(TEST_CLASS_DIR, TEST_NAME);
	}

	@Test
	public void testWelchTTest() {
		System.out.println("------------ BEGIN " + TEST_NAME + " TEST {" + numAttr + ", " + numPosSamples + ", "
			+ numNegSamples + "} ------------");

		getAndLoadTestConfiguration(TEST_NAME);

		List<String> proArgs = new ArrayList<>();
		proArgs.add("-explain");
		proArgs.add("-args");
		proArgs.add(input("posSamples"));
		proArgs.add(input("negSamples"));
		proArgs.add(output("t_statistics"));
		proArgs.add(output("degrees_of_freedom"));
		programArgs = proArgs.toArray(new String[proArgs.size()]);

		fullDMLScriptName = getScript();

		rCmd = getRCmd(inputDir(), expectedDir());

		double[][] posSamples = getRandomMatrix(numPosSamples, numAttr, 1, 5, 0.2, System.currentTimeMillis());
		double[][] negSamples = getRandomMatrix(numNegSamples, numAttr, 1, 5, 0.2, System.currentTimeMillis());

		MatrixCharacteristics mc1 = new MatrixCharacteristics(numPosSamples, numAttr, -1, -1);
		writeInputMatrixWithMTD("posSamples", posSamples, true, mc1);
		MatrixCharacteristics mc2 = new MatrixCharacteristics(numNegSamples, numAttr, -1, -1);
		writeInputMatrixWithMTD("negSamples", negSamples, true, mc2);

		int expectedNumberOfJobs = 2;

		runTest(true, EXCEPTION_NOT_EXPECTED, null, expectedNumberOfJobs);

		runRScript(true);

		double tol = Math.pow(10, -13);
		HashMap<CellIndex, Double> t_statistics_R = readRMatrixFromExpectedDir("t_statistics");
		HashMap<CellIndex, Double> t_statistics_SYSTEMDS = readDMLMatrixFromOutputDir("t_statistics");
		TestUtils.compareMatrices(t_statistics_R, t_statistics_SYSTEMDS, tol, "t_statistics_R", "t_statistics_SYSTEMDS");

		HashMap<CellIndex, Double> degrees_of_freedom_R = readRMatrixFromExpectedDir("degrees_of_freedom");
		HashMap<CellIndex, Double> degrees_of_freedom_SYSTEMDS = readDMLMatrixFromOutputDir("degrees_of_freedom");
		TestUtils.compareMatrices(degrees_of_freedom_R, degrees_of_freedom_SYSTEMDS, tol, "degrees_of_freedom_R",
			"degrees_of_freedom_SYSTEMDS");
	}
}
