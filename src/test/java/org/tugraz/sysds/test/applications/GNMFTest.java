/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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
 
package org.tugraz.sysds.test.applications;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;
import org.tugraz.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestUtils;

@RunWith(value = Parameterized.class)
public class GNMFTest extends AutomatedTestBase 
{

	protected final static String TEST_DIR = "applications/gnmf/";
	protected final static String TEST_NAME = "GNMF";
	protected String TEST_CLASS_DIR = TEST_DIR + GNMFTest.class.getSimpleName() + "/";
	
	protected int m, n, k;
	
	public GNMFTest(int m, int n, int k) {
		this.m = m;
		this.n = n;
		this.k = k;
	}
	
	@Parameters
	 public static Collection<Object[]> data() {
	   Object[][] data = new Object[][] { { 100, 50, 5 }, { 2000, 1500, 50 }, { 7000, 1500, 50 }};
	   return Arrays.asList(data);
	 }
	 
	@Override
	public void setUp() {
		addTestConfiguration(TEST_CLASS_DIR, TEST_NAME);
	}
	
	@Test
	public void testGNMF() {
		System.out.println("------------ BEGIN " + TEST_NAME + " TEST {" + m + ", "
				+ n + ", " + k + "} ------------");
		
		int maxiter = 2;
		double Eps = Math.pow(10, -8);
				
		getAndLoadTestConfiguration(TEST_NAME);

		List<String> proArgs = new ArrayList<String>();
		
		proArgs.add("-args");
		proArgs.add(input("v"));
		proArgs.add(input("w"));
		proArgs.add(input("h"));
		proArgs.add(Integer.toString(maxiter));
		proArgs.add(output("w"));
		proArgs.add(output("h"));
		programArgs = proArgs.toArray(new String[proArgs.size()]);
		
		fullDMLScriptName = getScript();
		
		rCmd = getRCmd(inputDir(), Integer.toString(maxiter), expectedDir());

		double[][] v = getRandomMatrix(m, n, 1, 5, 0.2, System.currentTimeMillis());
		double[][] w = getRandomMatrix(m, k, 0, 1, 1, System.currentTimeMillis());
		double[][] h = getRandomMatrix(k, n, 0, 1, 1, System.currentTimeMillis());

		writeInputMatrixWithMTD("v", v, true);
		writeInputMatrixWithMTD("w", w, true);
		writeInputMatrixWithMTD("h", h, true);

		for (int i = 0; i < maxiter; i++) {
			double[][] tW = TestUtils.performTranspose(w);
			double[][] tWV = TestUtils.performMatrixMultiplication(tW, v);
			double[][] tWW = TestUtils.performMatrixMultiplication(tW, w);
			double[][] tWWH = TestUtils.performMatrixMultiplication(tWW, h);
			for (int j = 0; j < k; j++) {
				for (int l = 0; l < n; l++) {
					h[j][l] = h[j][l] * (tWV[j][l] / (tWWH[j][l] + Eps));
				}
			}

			double[][] tH = TestUtils.performTranspose(h);
			double[][] vTH = TestUtils.performMatrixMultiplication(v, tH);
			double[][] hTH = TestUtils.performMatrixMultiplication(h, tH);
			double[][] wHTH = TestUtils.performMatrixMultiplication(w, hTH);
			for (int j = 0; j < m; j++) {
				for (int l = 0; l < k; l++) {
					w[j][l] = w[j][l] * (vTH[j][l] / (wHTH[j][l] + Eps));
				}
			}
		}

		/*
		 * Expected number of jobs:
		 * Reblock - 1 job 
		 * While loop iteration - 10 jobs
		 * Final output write - 1 job
		 */
		int expectedNumberOfJobs = 12;
		runTest(true, EXCEPTION_NOT_EXPECTED, null, expectedNumberOfJobs); 
		
		runRScript(true);

		HashMap<CellIndex, Double> hmWSYSTEMDS = readDMLMatrixFromHDFS("w");
		HashMap<CellIndex, Double> hmHSYSTEMDS = readDMLMatrixFromHDFS("h");
		HashMap<CellIndex, Double> hmWR = readRMatrixFromFS("w");
		HashMap<CellIndex, Double> hmHR = readRMatrixFromFS("h");

		TestUtils.compareMatrices(hmWSYSTEMDS, hmWR, 0.000001, "hmWSYSTEMDS", "hmWR");
		TestUtils.compareMatrices(hmHSYSTEMDS, hmHR, 0.000001, "hmHSYSTEMDS", "hmHR");
	}
}
