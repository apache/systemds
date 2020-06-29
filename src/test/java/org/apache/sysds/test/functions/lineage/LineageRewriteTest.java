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

package org.apache.sysds.test.functions.lineage;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.junit.Test;
import org.apache.sysds.hops.recompile.Recompiler;
import org.apache.sysds.runtime.lineage.Lineage;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class LineageRewriteTest extends AutomatedTestBase {
	protected static final String TEST_DIR = "functions/lineage/";
	protected static final String TEST_NAME1 = "RewriteTest3";
	protected static final String TEST_NAME2 = "RewriteTest2";
	protected static final String TEST_NAME3 = "RewriteTest7";
	protected static final String TEST_NAME4 = "RewriteTest8";
	protected static final String TEST_NAME5 = "RewriteTest9";
	protected static final String TEST_NAME6 = "RewriteTest10";
	protected static final String TEST_NAME7 = "RewriteTest11";
	protected static final String TEST_NAME8 = "RewriteTest12";
	protected static final String TEST_NAME9 = "RewriteTest13";
	
	protected String TEST_CLASS_DIR = TEST_DIR + LineageRewriteTest.class.getSimpleName() + "/";
	
	protected static final int numRecords = 100;
	protected static final int numFeatures = 30;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2));
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3));
		addTestConfiguration(TEST_NAME4, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4));
		addTestConfiguration(TEST_NAME5, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME5));
		addTestConfiguration(TEST_NAME6, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME6));
		addTestConfiguration(TEST_NAME7, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME7));
		addTestConfiguration(TEST_NAME8, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME8));
		addTestConfiguration(TEST_NAME9, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME9));
	}
	
	@Test
	public void testTsmm2Cbind() {
		testRewrite(TEST_NAME1, false, 0);
	}

	@Test
	public void testTsmmCbind() {
		testRewrite(TEST_NAME2, false, 0);
	}

	@Test
	public void testTsmmRbind() {
		testRewrite(TEST_NAME3, false, 0);
	}

	@Test
	public void testMatmulRbindLeft() {
		testRewrite(TEST_NAME4, false, 0);
	}

	@Test
	public void testMatmulCbindRight() {
		testRewrite(TEST_NAME5, false, 0);
	}

	@Test
	public void testElementMulRbind() {
		testRewrite(TEST_NAME6, true, 0);
	}

	@Test
	public void testElementMulCbind() {
		testRewrite(TEST_NAME7, true, 0);
	}

	@Test
	public void testaggregatecbind() {
		testRewrite(TEST_NAME8, false, 2);
	}

	@Test
	public void testTsmmCbindOnes() {
		// This also tests testMatmulCbindRightOnes.
		testRewrite(TEST_NAME9, false, 0);
	}

	private void testRewrite(String testname, boolean elementwise, int classes) {
		try {
			getAndLoadTestConfiguration(testname);
			List<String> proArgs = new ArrayList<>();
			
			proArgs.add("-stats");
			proArgs.add("-lineage");
			proArgs.add("-args");
			proArgs.add(input("X"));
			proArgs.add(input("Y"));
			proArgs.add(output("Res"));
			programArgs = proArgs.toArray(new String[proArgs.size()]);
			fullDMLScriptName = getScript();
			double[][] X = getRandomMatrix(numRecords, numFeatures, 0, 1, 0.8, -1);
			double[][] Y = !elementwise ? getRandomMatrix(numFeatures, numRecords, 0, 1, 0.8, -1)
				: getRandomMatrix(numRecords, numFeatures, 0, 1, 0.8, -1);
			if (classes > 0) {
				 Y = getRandomMatrix(numRecords, 1, 0, 1, 1, -1);
				 for(int i=0; i<numRecords; i++){
					 Y[i][0] = (int)(Y[i][0]*classes) + 1;
					 Y[i][0] = (Y[i][0] > classes) ? classes : Y[i][0];
				}
			}
			writeInputMatrixWithMTD("X", X, true);
			writeInputMatrixWithMTD("Y", Y, true);
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
			HashMap<MatrixValue.CellIndex, Double> R_orig = readDMLMatrixFromHDFS("Res");

			proArgs.clear();
			proArgs.add("-stats");
			proArgs.add("-lineage");
			proArgs.add("reuse_hybrid");
			proArgs.add("-args");
			proArgs.add(input("X"));
			proArgs.add(input("Y"));
			proArgs.add(output("Res"));
			programArgs = proArgs.toArray(new String[proArgs.size()]);
			fullDMLScriptName = getScript();
			writeInputMatrixWithMTD("X", X, true);
			Lineage.resetInternalState();
			Lineage.setLinReusePartial();
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
			Lineage.setLinReuseNone();
			HashMap<MatrixValue.CellIndex, Double> R_reused = readDMLMatrixFromHDFS("Res");
			TestUtils.compareMatrices(R_orig, R_reused, 1e-6, "Origin", "Reused");
		}
		finally {
			Recompiler.reinitRecompiler();
		}
	}
}
