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

import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.util.HashMap;

public class BuiltinUniqueTest extends AutomatedTestBase {
	private final static String TEST_NAME = "unique";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinUniqueTest.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"R"}));
	}

	@Test
	public void testUnique1CP() {
		double[][] X = {{1},{1},{6},{9},{4},{2},{0},{9},{0},{0},{4},{4}};
		double[][] expected = {{0}, {1}, {2}, {4}, {6}, {9}};
		runUniqueTest(X, expected, true, ExecType.CP);
	}

	@Test
	public void testUnique1SP() {
		double[][] X = {{1},{1},{6},{9},{4},{2},{0},{9},{0},{0},{4},{4}};
		double[][] expected = {{0}, {1}, {2}, {4}, {6}, {9}};
		runUniqueTest(X, expected,true, ExecType.SPARK);
	}

	private void runUniqueTest(double[][] X, double[][] expected, boolean defaultProb, ExecType instType) {
		Types.ExecMode platformOld = setExecMode(instType);
		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{ "-args", input("X"), output("R")};

			writeInputMatrixWithMTD("X", X, true);

			runTest(true, false, null, -1);

			HashMap<MatrixValue.CellIndex, Double> R = new HashMap<>();
			for(int i=0; i<expected.length; i++)
				R.put(new MatrixValue.CellIndex(i+1,1), expected[i][0]);
			HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");
			TestUtils.compareMatrices(dmlfile, R, 1e-10, "dml", "expected");
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
