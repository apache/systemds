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

import java.util.HashMap;

import org.junit.Test;

import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class BuiltinShortestPathTest extends AutomatedTestBase {
	private final static String TEST_NAME = "shortestPathTest";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinShortestPathTest.class.getSimpleName() + "/";
	private final static double eps = 1e-10;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"r"}));
	}

	@Test
	public void testShortestPathNode1CP() {
		runShortestPathNodeTest(1, new double[][] {{0}, {2}, {5}, {5}});
	}
	
	@Test
	public void testShortestPathNode2CP() {
		runShortestPathNodeTest(2, new double[][] {{1}, {0}, {4}, {5}});
	}
	
	@Test
	public void testShortestPathNode3CP() {
		runShortestPathNodeTest(3, new double[][] {{4}, {3}, {0}, {1}});
	}
	
	

	private void runShortestPathNodeTest(int node, double [][] Res) {
		loadTestConfiguration(getTestConfiguration(TEST_NAME));
	
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[]{ "-args",
			input("X"), String.valueOf(node), output("R")};
	
		double[][] X = {{0, 2, 5, 5 }, 
						{1, 0, 4, 10}, 
						{0, 3, 0, 1 },
						{3, 2, 0, 0 }};
		writeInputMatrixWithMTD("X", X, true);
	
		runTest(true, false, null, -1);
	
		HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");
		double[][] Y = TestUtils.convertHashMapToDoubleArray(dmlfile);
		TestUtils.compareMatrices(Res, Y, eps);
	}
}
