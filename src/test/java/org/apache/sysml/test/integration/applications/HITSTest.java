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

package org.apache.sysml.test.integration.applications;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.utils.TestUtils;


public abstract class HITSTest extends AutomatedTestBase 
{
	protected final static String TEST_DIR = "applications/hits/";
	protected final static String TEST_NAME = "HITS";
	protected String TEST_CLASS_DIR = TEST_DIR + HITSTest.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		addTestConfiguration(TEST_CLASS_DIR, TEST_NAME);
	}
	
	protected void testHits(ScriptType scriptType) {
		System.out.println("------------ BEGIN " + TEST_NAME + " " + scriptType + " TEST ------------");
		this.scriptType = scriptType;
		
		int rows = 1000;
		int cols = 1000;
		int maxiter = 2;

		getAndLoadTestConfiguration(TEST_NAME);
		
		List<String> proArgs = new ArrayList<String>();
		if (scriptType == ScriptType.PYDML) {
			proArgs.add("-python");
		}
		proArgs.add("-args");
		proArgs.add(input("G"));
		proArgs.add(Integer.toString(maxiter));
		proArgs.add(Double.toString(Math.pow(10, -6)));
		proArgs.add(output("hubs"));
		proArgs.add(output("authorities"));
		programArgs = proArgs.toArray(new String[proArgs.size()]);
		
		fullDMLScriptName = getScript();
		
		rCmd = getRCmd(inputDir(), Integer.toString(maxiter), Double.toString(Math.pow(10, -6)), expectedDir());
		
		double[][] G = getRandomMatrix(rows, cols, 0, 1, 1.0, -1);
		writeInputMatrixWithMTD("G", G, true);
		
		/*
		 * Expected number of jobs:
		 * Reblock - 1 job 
		 * While loop iteration - 9 jobs (Optimal = 8)
		 * Final output write - 1 job
		 */
		int expectedNumberOfJobs = 11;
		runTest(true, EXCEPTION_NOT_EXPECTED, null, expectedNumberOfJobs);
		
		runRScript(true);

		HashMap<CellIndex, Double> hubsSYSTEMML = readDMLMatrixFromHDFS("hubs");
		HashMap<CellIndex, Double> authSYSTEMML = readDMLMatrixFromHDFS("authorities");
		HashMap<CellIndex, Double> hubsR = readRMatrixFromFS("hubs");
		HashMap<CellIndex, Double> authR = readRMatrixFromFS("authorities");

		TestUtils.compareMatrices(hubsSYSTEMML, hubsR, 0.001, "hubsSYSTEMML", "hubsR");
		TestUtils.compareMatrices(authSYSTEMML, authR, 0.001, "authSYSTEMML", "authR");
		
	}
}
