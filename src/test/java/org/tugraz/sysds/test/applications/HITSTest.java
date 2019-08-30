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
import java.util.HashMap;
import java.util.List;

import org.junit.Test;
import org.tugraz.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestUtils;

public class HITSTest extends AutomatedTestBase 
{
	protected final static String TEST_DIR = "applications/hits/";
	protected final static String TEST_NAME = "HITS";
	protected String TEST_CLASS_DIR = TEST_DIR + HITSTest.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		addTestConfiguration(TEST_CLASS_DIR, TEST_NAME);
	}
	
	@Test
	public void testHits() {
		System.out.println("------------ BEGIN " + TEST_NAME + " TEST ------------");
		
		int rows = 1000;
		int cols = 1000;
		int maxiter = 2;

		getAndLoadTestConfiguration(TEST_NAME);
		
		List<String> proArgs = new ArrayList<String>();
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

		HashMap<CellIndex, Double> hubsSYSTEMDS = readDMLMatrixFromHDFS("hubs");
		HashMap<CellIndex, Double> authSYSTEMDS = readDMLMatrixFromHDFS("authorities");
		HashMap<CellIndex, Double> hubsR = readRMatrixFromFS("hubs");
		HashMap<CellIndex, Double> authR = readRMatrixFromFS("authorities");

		TestUtils.compareMatrices(hubsSYSTEMDS, hubsR, 0.001, "hubsSYSTEMDS", "hubsR");
		TestUtils.compareMatrices(authSYSTEMDS, authR, 0.001, "authSYSTEMDS", "authR");
		
	}
}
