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

package org.apache.sysds.test.functions.builtin.part2;


import org.junit.Assert;
import org.junit.Test;

import java.util.HashMap;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class BuiltinSliceLineTest extends AutomatedTestBase
{
	private static final String PREP_NAME = "slicefinderPrep";
	private static final String TEST_NAME = "slicefinder";
	private static final String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinSliceLineTest.class.getSimpleName() + "/";
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
		runSliceFinderTest(4, "e", true, false, ExecMode.HYBRID);
	}
	
	@Test
	public void testTop4SinglenodeDP() {
		runSliceFinderTest(4, "e", true, false, ExecMode.SINGLE_NODE);
	}
	
	@Test
	public void testTop4HybridTP() {
		runSliceFinderTest(4, "e", false, false, ExecMode.HYBRID);
	}
	
	@Test
	public void testTop4SinglenodeTP() {
		runSliceFinderTest(4, "e", false, false, ExecMode.SINGLE_NODE);
	}

	@Test
	public void testTop10HybridDP() {
		runSliceFinderTest(10, "e", true, false, ExecMode.HYBRID);
	}
	
	@Test
	public void testTop10SinglenodeDP() {
		runSliceFinderTest(10, "e", true, false, ExecMode.SINGLE_NODE);
	}
	
	@Test
	public void testTop10HybridTP() {
		runSliceFinderTest(10, "e", false, false, ExecMode.HYBRID);
	}
	
	@Test
	public void testTop10SinglenodeTP() {
		runSliceFinderTest(10, "e", false, false, ExecMode.SINGLE_NODE);
	}

	@Test
	public void testTop4HybridDPSel() {
		runSliceFinderTest(4, "e", true, true, ExecMode.HYBRID);
	}
	
	@Test
	public void testTop4SinglenodeDPSel() {
		runSliceFinderTest(4, "e", true, true, ExecMode.SINGLE_NODE);
	}
	
	@Test
	public void testTop4HybridTPSel() {
		runSliceFinderTest(4, "e", false, true, ExecMode.HYBRID);
	}
	
	@Test
	public void testTop4SinglenodeTPSel() {
		runSliceFinderTest(4, "e", false, true, ExecMode.SINGLE_NODE);
	}

	@Test
	public void testTop10HybridDPSel() {
		runSliceFinderTest(10, "e", true, true, ExecMode.HYBRID);
	}
	
	@Test
	public void testTop10SinglenodeDPSel() {
		runSliceFinderTest(10, "e", true, true, ExecMode.SINGLE_NODE);
	}
	
	@Test
	public void testTop10HybridTPSel() {
		runSliceFinderTest(10, "e", false, true, ExecMode.HYBRID);
	}
	
	@Test
	public void testTop10SinglenodeTPSel() {
		runSliceFinderTest(10, "e", false, true, ExecMode.SINGLE_NODE);
	}
	
	@Test
	public void testTop10HybridTPSelE2() {
		runSliceFinderTest(10, "oe", false, true, ExecMode.HYBRID);
	}
	
	@Test
	public void testTop10SinglenodeTPSelE2() {
		runSliceFinderTest(10, "oe", false, true, ExecMode.SINGLE_NODE);
	}

	@Test
	public void testSlicefinderCustomInputs1() {
		double[][] X = {
			{2, 1, 1, 2, 3, 2, 3, 3, 1, 2}, 
			{2, 2, 2, 3, 4, 1, 2, 1, 3, 2}, 
			{2, 1, 3, 3, 2, 2, 3, 1, 1, 4}, 
			{1, 2, 2, 1, 3, 2, 3, 2, 2, 3}, 
			{3, 2, 3, 4, 3, 3, 4, 1, 1, 3}, 
			{4, 3, 2, 3, 4, 4, 3, 4, 1, 1}, 
			{2, 2, 2, 4, 3, 3, 2, 2, 1, 2}, 
			{1, 1, 2, 2, 3, 3, 2, 1, 1, 2}, 
			{4, 3, 2, 1, 3, 2, 4, 2, 4, 3}, 
			{1, 3, 1, 4, 1, 3, 3, 2, 3, 2},
			{2, 4, 3, 1, 2, 4, 1, 3, 2, 4},
			{3, 2, 4, 3, 1, 4, 2, 3, 4, 1},
			{4, 1, 2, 4, 3, 1, 4, 2, 1, 3},
			{1, 3, 4, 2, 4, 3, 1, 4, 2, 3},
			{2, 4, 1, 3, 2, 4, 3, 1, 4, 2},
			{3, 2, 4, 1, 3, 4, 2, 3, 1, 4},
			{4, 1, 3, 2, 4, 1, 4, 2, 3, 1},
			{1, 3, 2, 4, 1, 3, 4, 2, 4, 3},
			{2, 4, 1, 3, 2, 4, 3, 1, 2, 4},
			{2, 3, 3, 2, 1, 4, 2, 3, 2, 3}
			};
		double[][] e = {
			{0.159}, {0.588}, {0.414}, {0.305}, {0.193}, {0.195}, {0.878}, {0.149}, {0.835}, {0.344},
			{0.123}, {0.456}, {0.789}, {0.987}, {0.654}, {0.321}, {0.246}, {0.135}, {0.579}, {0.802}
		};
		int K = 10; 
		double[][] correctRes = {
			{0.307, 2.807, 0.878, 4.000},
			{0.307, 2.807, 0.878, 4.000},
			{0.282, 2.759, 0.987, 4.000},
			{0.157, 4.046, 0.987, 7.000},
			{0.127, 2.956, 0.878, 5.000},
			{0.122, 2.942, 0.878, 5.000},
			{0.074, 3.298, 0.987, 6.000},
			{0.064, 4.197, 0.878, 8.000},
			{0.061, 2.796, 0.987, 5.000},
			{0.038, 3.194, 0.878, 6.000}
		};
		testSlicefinderCustomInputs(X, e, K, correctRes);
	}

	@Test
	public void testSlicefinderCustomInputs2() {
		double[][] X = {
			{2, 1, 1, 1, 3, 4, 2, 2, 1, 2},
			{3, 3, 3, 2, 1, 2, 3, 1, 4, 2},
			{3, 2, 3, 1, 1, 1, 4, 3, 4, 2},
			{1, 3, 2, 3, 2, 3, 2, 1, 2, 1},
			{4, 3, 1, 1, 1, 1, 1, 1, 3, 2},
			{2, 2, 3, 3, 2, 2, 2, 3, 4, 1},
			{3, 2, 2, 2, 4, 4, 2, 4, 1, 1},
			{1, 3, 3, 2, 1, 3, 1, 2, 4, 4},
			{2, 1, 2, 2, 3, 1, 2, 3, 2, 1},
			{4, 1, 3, 4, 1, 4, 2, 3, 4, 4},
			{4, 2, 4, 4, 2, 1, 2, 1, 1, 4},
			{4, 1, 1, 4, 1, 4, 3, 2, 4, 2},
			{2, 1, 2, 2, 3, 1, 4, 3, 3, 4},
			{4, 1, 3, 1, 3, 1, 2, 1, 3, 3},
			{2, 1, 3, 1, 1, 3, 1, 2, 1, 2},
			{1, 3, 4, 3, 1, 2, 2, 2, 1, 1},
			{2, 4, 4, 3, 4, 1, 2, 1, 2, 4},
			{3, 3, 3, 3, 3, 1, 2, 3, 4, 4},
			{3, 2, 2, 2, 4, 1, 4, 2, 3, 1},
			{1, 2, 3, 2, 4, 3, 2, 3, 2, 3}		
			};
		
		double[][] e = {
			{0.591}, {0.858}, {0.144}, {0.350}, {0.931}, {0.951}, {0.788}, {0.491}, {0.358}, {0.443},
			{0.231}, {0.564}, {0.897}, {0.879}, {0.546}, {0.132}, {0.462}, {0.153}, {0.759}, {0.028}
		};
		int K = 10; 
		double[][] correctRes = {
			{0.410, 3.466, 0.931, 4.000},
			{0.410, 3.466, 0.931, 4.000},
			{0.111, 2.802, 0.897, 4.000},
			{0.075, 3.805, 0.951, 6.000},
			{0.057, 4.278, 0.897, 7.000},
			{0.047, 3.711, 0.931, 6.000},
			{0.035, 3.152, 0.897, 5.000},
			{0.032, 4.179, 0.897, 7.000},
			{0.023, 3.634, 0.931, 6.000},
			{0.013, 3.091, 0.931, 5.000}
			};
		
		testSlicefinderCustomInputs(X, e, K, correctRes);
	}

	
	@Test
	public void testSlicefinderCustomInputs3() {
		double[][] X = {
			{2, 1, 1, 2, 3, 2, 3, 3, 1, 2}, 
			{2, 2, 2, 3, 4, 1, 2, 1, 3, 2}, 
			{2, 1, 3, 3, 2, 2, 3, 1, 1, 4}, 
			{1, 2, 2, 1, 3, 2, 3, 2, 2, 3}, 
			{3, 2, 3, 4, 3, 3, 4, 1, 1, 3}, 
			{4, 3, 2, 3, 4, 4, 3, 4, 1, 1}, 
			{2, 2, 2, 4, 3, 3, 2, 2, 1, 2}, 
			{1, 1, 2, 2, 3, 3, 2, 1, 1, 2}, 
			{4, 3, 2, 1, 3, 2, 4, 2, 4, 3}, 
			{1, 3, 1, 4, 1, 3, 3, 2, 3, 2},
			{2, 4, 3, 1, 2, 4, 1, 3, 2, 4},
			{3, 2, 4, 3, 1, 4, 2, 3, 4, 1},
			{4, 1, 2, 4, 3, 1, 4, 2, 1, 3},
			{1, 3, 4, 2, 4, 3, 1, 4, 2, 3},
			{2, 4, 1, 3, 2, 4, 3, 1, 4, 2},
			{3, 2, 4, 1, 3, 4, 2, 3, 1, 4},
			{4, 1, 3, 2, 4, 1, 4, 2, 3, 1},
			{1, 3, 2, 4, 1, 3, 4, 2, 4, 3},
			{2, 4, 1, 3, 2, 4, 3, 1, 2, 4},
			{2, 3, 3, 2, 1, 4, 2, 3, 2, 3},
			{2, 1, 1, 1, 3, 4, 2, 2, 1, 2},
			{3, 3, 3, 2, 1, 2, 3, 1, 4, 2},
			{3, 2, 3, 1, 1, 1, 4, 3, 4, 2},
			{1, 3, 2, 3, 2, 3, 2, 1, 2, 1},
			{4, 3, 1, 1, 1, 1, 1, 1, 3, 2},
			{2, 2, 3, 3, 2, 2, 2, 3, 4, 1},
			{3, 2, 2, 2, 4, 4, 2, 4, 1, 1},
			{1, 3, 3, 2, 1, 3, 1, 2, 4, 4},
			{2, 1, 2, 2, 3, 1, 2, 3, 2, 1},
			{4, 1, 3, 4, 1, 4, 2, 3, 4, 4},
			{4, 2, 4, 4, 2, 1, 2, 1, 1, 4},
			{4, 1, 1, 4, 1, 4, 3, 2, 4, 2},
			{2, 1, 2, 2, 3, 1, 4, 3, 3, 4},
			{4, 1, 3, 1, 3, 1, 2, 1, 3, 3},
			{2, 1, 3, 1, 1, 3, 1, 2, 1, 2},
			{1, 3, 4, 3, 1, 2, 2, 2, 1, 1},
			{2, 4, 4, 3, 4, 1, 2, 1, 2, 4},
			{3, 3, 3, 3, 3, 1, 2, 3, 4, 4},
			{3, 2, 2, 2, 4, 1, 4, 2, 3, 1},
			{1, 2, 3, 2, 4, 3, 2, 3, 2, 3}
			};
		double[][] e = {
			{0.159}, {0.588}, {0.414}, {0.305}, {0.193}, {0.195}, {0.878}, {0.149}, {0.835}, {0.344},
			{0.123}, {0.456}, {0.789}, {0.987}, {0.654}, {0.321}, {0.246}, {0.135}, {0.579}, {0.802},
			{0.591}, {0.858}, {0.144}, {0.350}, {0.931}, {0.951}, {0.788}, {0.491}, {0.358}, {0.443},
			{0.231}, {0.564}, {0.897}, {0.879}, {0.546}, {0.132}, {0.462}, {0.153}, {0.759}, {0.028}
		};
		int K = 10; 
		double[][] correctRes = {
			{0.149, 4.300, 0.931, 6.000},
			{0.113, 3.138, 0.987, 4.000},
			{0.093, 4.644, 0.931, 7.000},
			{0.090, 4.630, 0.951, 7.000},
			{0.059, 8.002, 0.951, 14.000},
			{0.024, 2.954, 0.951, 4.000},
			{0.017, 3.415, 0.897, 5.000},
			{0.010, 3.398, 0.878, 5.000},
			{0.009, 2.923, 0.897, 4.000},
			{0.008, 3.391, 0.897, 5.000}
		};
		testSlicefinderCustomInputs(X, e, K, correctRes);
	}
	
//	@Test
//	public void testTop10SparkTP() {
//		runSliceFinderTest(10, false, ExecMode.SPARK);
//	}
	
	private void runSliceFinderTest(int K, String err, boolean dp, boolean selCols, ExecMode mode) {
		ExecMode platformOld = setExecMode(mode);
		loadTestConfiguration(getTestConfiguration(TEST_NAME));
		String HOME = SCRIPT_DIR + TEST_DIR;
		String data = DATASET_DIR+ "Salaries.csv";
		
		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			
			//run data preparation
			fullDMLScriptName = HOME + PREP_NAME + ".dml";
			programArgs = new String[]{"-args", data, err, output("X"), output("e")};
			runTest(true, false, null, -1);
			
			//read output and store for dml and R
			double[][] X = TestUtils.convertHashMapToDoubleArray(readDMLMatrixFromOutputDir("X"));
			double[][] e = TestUtils.convertHashMapToDoubleArray(readDMLMatrixFromOutputDir("e"));
			writeInputMatrixWithMTD("X", X, true);
			writeInputMatrixWithMTD("e", e, true);
			
			//execute main test
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", input("X"), input("e"), String.valueOf(K),
				String.valueOf(!dp).toUpperCase(), String.valueOf(selCols).toUpperCase(),
				String.valueOf(VERBOSE).toUpperCase(), output("R")};
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + String.valueOf(K) 
				+ " " + String.valueOf(!dp).toUpperCase() + " " + expectedDir();
			
			runTest(true, false, null, -1);
			runRScript(true); 
			
			//compare dml and R
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("R");
			TestUtils.compareMatrices(dmlfile, rfile, 1e-2, "Stat-DML", "Stat-R");
			
			//compare expected results
			if( err.equals("e") ) {
				double[][] ret = TestUtils.convertHashMapToDoubleArray(dmlfile);
				if( mode != ExecMode.SPARK ) //TODO why only CP correct, but R always matches? test framework?
					for(int i=0; i<K; i++)
						TestUtils.compareMatrices(EXPECTED_TOPK[i], ret[i], 1e-2);
			}
			
			//ensure proper inlining, despite initially multiple calls and large function
			Assert.assertFalse(heavyHittersContainsSubString("evalSlice"));
		}
		finally {
			rtplatform = platformOld;
		}
	}

	public void testSlicefinderCustomInputs(double[][] X, double[][] e, int K, double[][] correctRes) {
		boolean dp = true, selCols = false;
		ExecMode mode = ExecMode.SINGLE_NODE; 
		ExecMode platformOld = setExecMode(mode);
		loadTestConfiguration(getTestConfiguration(TEST_NAME));
		String HOME = SCRIPT_DIR + TEST_DIR;
		
		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));

			writeInputMatrixWithMTD("X", X, false);
			writeInputMatrixWithMTD("e", e, false);
			
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", input("X"), input("e"), String.valueOf(K),
				String.valueOf(!dp).toUpperCase(), String.valueOf(selCols).toUpperCase(),
				String.valueOf(VERBOSE).toUpperCase(), output("R")};
			
			runTest(true, false, null, -1);
			
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");
			double[][] ret = TestUtils.convertHashMapToDoubleArray(dmlfile);
			TestUtils.compareMatrices(correctRes, ret, 1e-2);
		
			Assert.assertFalse(heavyHittersContainsSubString("evalSlice"));
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
