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

package org.apache.sysds.test.functions.builtin;


import org.junit.Assert;
import org.junit.Test;

import java.util.HashMap;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class BuiltinSliceFinderTest extends AutomatedTestBase
{
	private static final String PREP_NAME = "slicefinderPrep";
	private static final String TEST_NAME = "slicefinder";
	private static final String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinSliceFinderTest.class.getSimpleName() + "/";
	private static final boolean VERBOSE = true;
	
	private static final double[][] EXPECTED_TOPK = new double[][]{
		{1.042, 69210699988.477, 11078019685.642, 18.000},
		{0.478, 92957580467.849, 11078019685.642, 39.000},
		{0.316, 40425449547.480, 11078019685.642, 10.000},
		{0.262, 67630559163.266, 7261504482.540, 29.000},
		{0.224, 202448990843.317, 11119010986.000, 125.000},
		{0.218, 68860581248.568, 7261504482.540, 31.000},
		{0.164, 206527445340.279, 11119010986.000, 135.000},
		{0.122, 68961886413.866, 7261504482.540, 34.000},
		{0.098, 360278523220.479, 11119010986.000, 266.000},
		{0.092, 73954209826.485, 11078019685.642, 39.000}
	};
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"R"}));
	}

	@Test
	public void testTop4HybridDP() {
		runSliceFinderTest(4, true, ExecMode.HYBRID);
	}
	
	@Test
	public void testTop4SinglenodeDP() {
		runSliceFinderTest(4, true, ExecMode.SINGLE_NODE);
	}
	
	@Test
	public void testTop4HybridTP() {
		runSliceFinderTest(4, false, ExecMode.HYBRID);
	}
	
	@Test
	public void testTop4SinglenodeTP() {
		runSliceFinderTest(4, false, ExecMode.SINGLE_NODE);
	}

	@Test
	public void testTop10HybridDP() {
		runSliceFinderTest(10, true, ExecMode.HYBRID);
	}
	
	@Test
	public void testTop10SinglenodeDP() {
		runSliceFinderTest(10, true, ExecMode.SINGLE_NODE);
	}
	
	@Test
	public void testTop10HybridTP() {
		runSliceFinderTest(10, false, ExecMode.HYBRID);
	}
	
	@Test
	public void testTop10SinglenodeTP() {
		runSliceFinderTest(10, false, ExecMode.SINGLE_NODE);
	}
	
	private void runSliceFinderTest(int K, boolean dp, ExecMode mode) {
		ExecMode platformOld = setExecMode(ExecMode.HYBRID);
		loadTestConfiguration(getTestConfiguration(TEST_NAME));
		String HOME = SCRIPT_DIR + TEST_DIR;
		String data = HOME + "/data/Salaries.csv";
		
		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			
			//run data preparation
			fullDMLScriptName = HOME + PREP_NAME + ".dml";
			programArgs = new String[]{"-args", data, output("X"), output("e")};
			runTest(true, false, null, -1);
			
			//read output and store for dml and R
			double[][] X = TestUtils.convertHashMapToDoubleArray(readDMLMatrixFromHDFS("X"));
			double[][] e = TestUtils.convertHashMapToDoubleArray(readDMLMatrixFromHDFS("e"));
			writeInputMatrixWithMTD("X", X, true);
			writeInputMatrixWithMTD("e", e, true);
			
			//execute main test
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", input("X"), input("e"),
				String.valueOf(K),String.valueOf(!dp).toUpperCase(),
				String.valueOf(VERBOSE).toUpperCase(), output("R")};
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + String.valueOf(K) 
				+ " " + String.valueOf(!dp).toUpperCase() + " " + expectedDir();
			
			runTest(true, false, null, -1);
			runRScript(true); 
			
			//compare dml and R
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("R");
			TestUtils.compareMatrices(dmlfile, rfile, 1e-2, "Stat-DML", "Stat-R");
			
			//compare expected results
			double[][] ret = TestUtils.convertHashMapToDoubleArray(dmlfile);
			for(int i=0; i<K; i++)
				TestUtils.compareMatrices(EXPECTED_TOPK[i], ret[i], 1e-2);
		
			//ensure proper inlining, despite initially multiple calls and large function
			Assert.assertFalse(heavyHittersContainsSubString("evalSlice"));
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
