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
import java.util.HashMap;
import java.util.List;

import org.junit.Test;

import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;


public class HITSTest extends AutomatedTestBase 
{
	private final static String TEST_DIR = "applications/hits/";
	private final static String TEST_HITS = "HITS";
	private final static String HITS_HOME = SCRIPT_DIR + TEST_DIR;

	@Override
	public void setUp() {
		addTestConfiguration(TEST_HITS, new TestConfiguration(TEST_DIR, "HITS", new String[] { "hubs", "authorities" }));
	}

	@Test
	public void testHitsDml() {
		System.out.println("------------ BEGIN " + TEST_HITS + " DML TEST ------------");
		testHits(ScriptType.DML);
	}

	@Test
	public void testHitsPyDml() {
		System.out.println("------------ BEGIN " + TEST_HITS + " PYDML TEST ------------");
		testHits(ScriptType.PYDML);
	}
	
	public void testHits(ScriptType scriptType) {
		this.scriptType = scriptType;
		
		int rows = 1000;
		int cols = 1000;
		int maxiter = 2;

		/* This is for running the junit test by constructing the arguments directly */
		List<String> proArgs = new ArrayList<String>();
		proArgs.add("-args");
		proArgs.add(HITS_HOME + INPUT_DIR + "G");
		proArgs.add(Integer.toString(maxiter));
		proArgs.add(Integer.toString(rows));
		proArgs.add(Integer.toString(cols));
		proArgs.add(Double.toString(Math.pow(10, -6)));
		proArgs.add(HITS_HOME + OUTPUT_DIR + "hubs");
		proArgs.add(HITS_HOME + OUTPUT_DIR + "authorities");
		
		switch (scriptType) {
		case DML:
			fullDMLScriptName = HITS_HOME + TEST_HITS + ".dml";
			break;
		case PYDML:
			fullPYDMLScriptName = HITS_HOME + TEST_HITS + ".pydml";
			proArgs.add(0, "-python");
			break;
		}
		programArgs = proArgs.toArray(new String[proArgs.size()]);
		System.out.println("arguments from test case: " + Arrays.toString(programArgs));
		
		fullRScriptName = HITS_HOME + TEST_HITS + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + 
		       HITS_HOME + INPUT_DIR + " " + Integer.toString(maxiter) + " " + Double.toString(Math.pow(10, -6))+ " " + HITS_HOME + EXPECTED_DIR;

		TestConfiguration config = getTestConfiguration(TEST_HITS);
		/* This is for running the junit test the old way */
		config.addVariable("maxiter", maxiter);
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		loadTestConfiguration(config);
		
		double[][] G = getRandomMatrix(rows, cols, 0, 1, 1.0, -1);
		writeInputMatrix("G", G, true);
		
		/*
		 * Expected number of jobs:
		 * Reblock - 1 job 
		 * While loop iteration - 9 jobs (Optimal = 8)
		 * Final output write - 1 job
		 */
		int expectedNumberOfJobs = 11;
		runTest(true, EXCEPTION_NOT_EXPECTED, null, expectedNumberOfJobs);
		
		runRScript(true);
		disableOutAndExpectedDeletion();

		HashMap<CellIndex, Double> hubsDML = readDMLMatrixFromHDFS("hubs");
		HashMap<CellIndex, Double> authDML = readDMLMatrixFromHDFS("authorities");
		HashMap<CellIndex, Double> hubsR = readRMatrixFromFS("hubs");
		HashMap<CellIndex, Double> authR = readRMatrixFromFS("authorities");

		TestUtils.compareMatrices(hubsDML, hubsR, 0.001, "hubsDML", "hubsR");
		TestUtils.compareMatrices(authDML, authR, 0.001, "authDML", "authR");
		
	}
}
