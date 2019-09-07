/*
 * Copyright 2019 Graz University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tugraz.sysds.test.functions.lineage;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.junit.Test;
import org.tugraz.sysds.hops.recompile.Recompiler;
import org.tugraz.sysds.runtime.lineage.Lineage;
import org.tugraz.sysds.runtime.matrix.data.MatrixValue;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestConfiguration;
import org.tugraz.sysds.test.TestUtils;

public class LineageRewriteTest extends AutomatedTestBase {
	protected static final String TEST_DIR = "functions/lineage/";
	protected static final String TEST_NAME1 = "RewriteTest3";
	protected static final String TEST_NAME2 = "RewriteTest2";
	
	protected String TEST_CLASS_DIR = TEST_DIR + LineageRewriteTest.class.getSimpleName() + "/";
	
	protected static final int numRecords = 1000;
	protected static final int numFeatures = 100;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2));
	}
	
	@Test
	public void testRewrite1() {
		testRewrite(TEST_NAME1);
	}

	@Test
	public void testRewrite2() {
		testRewrite(TEST_NAME2);
	}

	private void testRewrite(String testname) {
		try {
			getAndLoadTestConfiguration(testname);
			List<String> proArgs = new ArrayList<String>();
			
			proArgs.add("-explain");
			proArgs.add("-stats");
			proArgs.add("-lineage");
			proArgs.add("-args");
			proArgs.add(input("X"));
			proArgs.add(output("Res"));
			programArgs = proArgs.toArray(new String[proArgs.size()]);
			fullDMLScriptName = getScript();
			double[][] X = getRandomMatrix(numRecords, numFeatures, 0, 1, 0.8, -1);
			writeInputMatrixWithMTD("X", X, true);
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
			HashMap<MatrixValue.CellIndex, Double> R_orig = readDMLMatrixFromHDFS("Res");

			proArgs.clear();
			proArgs.add("-explain");
			proArgs.add("recompile_hops");
			proArgs.add("-stats");
			proArgs.add("-lineage");
			proArgs.add("reuse");
			proArgs.add("-args");
			proArgs.add(input("X"));
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
