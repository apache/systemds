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
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestUtils;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class L2SVMTest extends AutomatedTestBase 
{
	protected final static String TEST_DIR = "applications/l2svm/";
	protected final static String TEST_NAME = "L2SVM";
	protected String TEST_CLASS_DIR = TEST_DIR + L2SVMTest.class.getSimpleName() + "/";

	protected int numRecords, numFeatures;
	protected double sparsity;
	protected boolean intercept;

	public L2SVMTest(int rows, int cols, double sp, boolean intercept) {
		numRecords = rows;
		numFeatures = cols;
		sparsity = sp;
		this.intercept = intercept;
	}
	
	@Parameters
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] {
			//sparse tests (sparsity=0.01)
			{100, 50, 0.01, false}, {1000, 500, 0.01, false}, {10000, 750, 0.01, false}, {10000, 750, 0.01, true}, {100000, 1000, 0.01, false},
			//dense tests (sparsity=0.7)
			{100, 50, 0.7, false}, {1000, 500, 0.7, false}, {1000, 500, 0.7, true}, {10000, 750, 0.7, false} });
	}
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_CLASS_DIR, TEST_NAME);
	}
	
	@Test
	public void testL2SVM1() {
		testL2SVM(true);
	}

	@Test
	public void testL2SVM2() {
		testL2SVM(false);
	}

	private void testL2SVM(boolean ngrams)
	{
		System.out.println("------------ BEGIN " + TEST_NAME 
			+ " TEST WITH {" + numRecords + ", " + numFeatures
			+ ", " + sparsity + ", " + intercept + "} ------------");
		int rows = numRecords;
		int cols = numFeatures;
		double epsilon = 1e-10;
		double lambda = 1.0;
		int maxiterations = 3;
		int maxNumberOfMRJobs = 21;

		getAndLoadTestConfiguration(TEST_NAME);

		List<String> proArgs = new ArrayList<>();
		proArgs.add("-stats");
		if (ngrams) {
			proArgs.add("-ngrams");
			proArgs.add("3,2");
			proArgs.add("10");
		}
		proArgs.add("-nvargs");
		proArgs.add("X=" + input("X"));
		proArgs.add("Y=" + input("Y"));
		proArgs.add("icpt=" + (intercept ? 1 : 0));
		proArgs.add("tol=" + epsilon);
		proArgs.add("reg=" + lambda);
		proArgs.add("maxiter=" + maxiterations);
		proArgs.add("model=" + output("w"));
		proArgs.add("Log=" + output("Log"));
		programArgs = proArgs.toArray(new String[proArgs.size()]);
		fullDMLScriptName = getScript();
		rCmd = getRCmd(inputDir(), (intercept ? Integer.toString(1) : Integer.toString(0)), Double.toString(epsilon), 
				Double.toString(lambda), Integer.toString(maxiterations), expectedDir());

		double[][] X = getRandomMatrix(rows, cols, 0, 1, sparsity, -1);
		double[][] Y = getRandomMatrix(rows, 1, -1, 1, 1, -1);
		for(int i=0; i<rows; i++)
			Y[i][0] = (Y[i][0] > 0) ? 1 : -1;

		writeInputMatrixWithMTD("X", X, true);
		writeInputMatrixWithMTD("Y", Y, true);
		runTest(true, EXCEPTION_NOT_EXPECTED, null, maxNumberOfMRJobs);

		runRScript(true);

		HashMap<CellIndex, Double> wR = readRMatrixFromExpectedDir("w");
		HashMap<CellIndex, Double> wSYSTEMDS= readDMLMatrixFromOutputDir("w");
		TestUtils.compareMatrices(wR, wSYSTEMDS, epsilon, "wR", "wSYSTEMDS");
	}
}
