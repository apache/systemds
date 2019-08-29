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

package org.tugraz.sysds.test.applications;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;
import org.tugraz.sysds.common.Types.ExecMode;
import org.tugraz.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestUtils;
import org.tugraz.sysds.utils.Statistics;

@RunWith(value = Parameterized.class)
public class ID3Test extends AutomatedTestBase
{
	protected final static String TEST_DIR = "applications/id3/";
	protected final static String TEST_NAME = "id3";
	protected String TEST_CLASS_DIR = TEST_DIR + ID3Test.class.getSimpleName() + "/";

	protected int numRecords, numFeatures;
	
	public ID3Test(int numRecords, int numFeatures) {
		this.numRecords = numRecords;
		this.numFeatures = numFeatures;
	}
	
	@Parameters
	public static Collection<Object[]> data() {
		Object[][] data = new Object[][] { {100, 50}, {1000, 50} };
		return Arrays.asList(data);
	}

	@Override
	public void setUp()
	{
		addTestConfiguration(TEST_CLASS_DIR, TEST_NAME);
	}
	
	@Test
	public void testID3() 
	{
		System.out.println("------------ BEGIN " + TEST_NAME + " TEST {" + numRecords + ", "
				+ numFeatures + "} ------------");
		
		int rows = numRecords; // # of rows in the training data 
		int cols = numFeatures;
		
		getAndLoadTestConfiguration(TEST_NAME);

		List<String> proArgs = new ArrayList<String>();
		proArgs.add("-explain");
		proArgs.add("-args");
		proArgs.add(input("X"));
		proArgs.add(input("y"));
		proArgs.add(output("nodes"));
		proArgs.add(output("edges"));
		programArgs = proArgs.toArray(new String[proArgs.size()]);
		
		fullDMLScriptName = getScript();
		
		rCmd = getRCmd(inputDir(), expectedDir());

		// prepare training data set
		double[][] X = TestUtils.round(getRandomMatrix(rows, cols, 1, 10, 1.0, 3));
		double[][] y = TestUtils.round(getRandomMatrix(rows, 1, 1, 10, 1.0, 7));
		writeInputMatrixWithMTD("X", X, true);
		writeInputMatrixWithMTD("y", y, true);
		
		//run tests
		runTest(true, EXCEPTION_NOT_EXPECTED, null, 129); //max 68 compiled jobs
		runRScript(true);

		//check also num actually executed jobs
		if(AutomatedTestBase.rtplatform != ExecMode.SPARK) {
			long actualSP = Statistics.getNoOfExecutedSPInst();
			Assert.assertEquals("Wrong number of executed jobs: expected 0 but executed "+actualSP+".", 0, actualSP);
		}
		
		//compare results
		HashMap<CellIndex, Double> nR = readRMatrixFromFS("nodes");
		HashMap<CellIndex, Double> nSYSTEMDS= readDMLMatrixFromHDFS("nodes");
		HashMap<CellIndex, Double> eR = readRMatrixFromFS("edges");
		HashMap<CellIndex, Double> eSYSTEMDS= readDMLMatrixFromHDFS("edges");
		TestUtils.compareMatrices(nR, nSYSTEMDS, Math.pow(10, -14), "nR", "nSYSTEMDS");
		TestUtils.compareMatrices(eR, eSYSTEMDS, Math.pow(10, -14), "eR", "eSYSTEMDS");
	}
}
