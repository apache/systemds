/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.test.integration.applications;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;

import org.junit.runners.Parameterized.Parameters;

import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

public abstract class GNMFTest extends AutomatedTestBase 
{

	protected final static String TEST_DIR = "applications/gnmf/";
	protected final static String TEST_GNMF = "GNMF";
	protected final static String GNMF_HOME = SCRIPT_DIR + TEST_DIR;
	
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
		addTestConfiguration(TEST_GNMF, new TestConfiguration(TEST_DIR, TEST_GNMF, new String[] { "w", "h" }));
	}
	
	protected void testGNMF(ScriptType scriptType) {
		System.out.println("------------ BEGIN " + TEST_GNMF + " " + scriptType + " TEST {" + m + ", "
				+ n + ", " + k + "} ------------");
		this.scriptType = scriptType;
		
		int maxiter = 2;
		
		/* This is for running the junit test the old way, i.e., replace $$x$$ in DML script with its value */
		TestConfiguration config = getTestConfiguration(TEST_GNMF);
		config.addVariable("m", m);
		config.addVariable("n", n);
		config.addVariable("k", k);
		config.addVariable("maxiter", maxiter);
		loadTestConfiguration(config);
		
		double Eps = Math.pow(10, -8);

		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		List<String> proArgs = new ArrayList<String>();
		if (scriptType == ScriptType.PYDML) {
			proArgs.add("-python");
		}
		proArgs.add("-args");
		proArgs.add(input("v"));
		proArgs.add(input("w"));
		proArgs.add(input("h"));
		proArgs.add(Integer.toString(m));
		proArgs.add(Integer.toString(n));
		proArgs.add(Integer.toString(k));
		proArgs.add(Integer.toString(maxiter));
		proArgs.add(output("w"));
		proArgs.add(output("h"));
		programArgs = proArgs.toArray(new String[proArgs.size()]);
		System.out.println("arguments from test case: " + Arrays.toString(programArgs));
		
		fullDMLScriptName = getScript();
		
		rCmd = getRCmd(inputDir(), Integer.toString(maxiter), expectedDir());

		double[][] v = getRandomMatrix(m, n, 1, 5, 0.2, System.currentTimeMillis());
		double[][] w = getRandomMatrix(m, k, 0, 1, 1, System.currentTimeMillis());
		double[][] h = getRandomMatrix(k, n, 0, 1, 1, System.currentTimeMillis());

		writeInputMatrix("v", v, true);
		writeInputMatrix("w", w, true);
		writeInputMatrix("h", h, true);

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
		
		/* GNMF must be run in the new way as GNMF.dml will be shipped */
		runTest(true, EXCEPTION_NOT_EXPECTED, null, expectedNumberOfJobs); 
		
		runRScript(true);
		disableOutAndExpectedDeletion();

		HashMap<CellIndex, Double> hmWSYSTEMML = readDMLMatrixFromHDFS("w");
		HashMap<CellIndex, Double> hmHSYSTEMML = readDMLMatrixFromHDFS("h");
		HashMap<CellIndex, Double> hmWR = readRMatrixFromFS("w");
		HashMap<CellIndex, Double> hmHR = readRMatrixFromFS("h");
		//HashMap<CellIndex, Double> hmWJava = TestUtils.convert2DDoubleArrayToHashMap(w);
		//HashMap<CellIndex, Double> hmHJava = TestUtils.convert2DDoubleArrayToHashMap(h);

		TestUtils.compareMatrices(hmWSYSTEMML, hmWR, 0.000001, "hmWSYSTEMML", "hmWR");
		TestUtils.compareMatrices(hmHSYSTEMML, hmHR, 0.000001, "hmHSYSTEMML", "hmHR");
		//TestUtils.compareMatrices(hmWDML, hmWJava, 0.000001, "hmWDML", "hmWJava");
		//TestUtils.compareMatrices(hmWR, hmWJava, 0.000001, "hmRDML", "hmWJava");
	}
}
