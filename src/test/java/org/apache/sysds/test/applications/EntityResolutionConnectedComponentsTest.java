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
package org.apache.sysds.test.applications;

import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.junit.Test;

public class EntityResolutionConnectedComponentsTest extends AutomatedTestBase {
	private final static String TEST_NAME = "EntityResolutionConnectedComponents";
	private final static String TEST_DIR = "applications/entity_resolution/connected_components/";

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_DIR, "cluster_by_connected_components", new String[]{"B"}));
	}

	@Test
	public void testConnectedComponents1() {
		testClusterByConnectedComponent(
			new double[][]{{0,},},
			new double[][]{{0,}}
		);
	}
	@Test
	public void testConnectedComponents2() {
		testClusterByConnectedComponent(
			new double[][]{
				{0, 0},
				{0, 0},
			},
			new double[][]{
				{0, 0},
				{0, 0},
			}
		);
	}
	@Test
	public void testConnectedComponents3() {
		testClusterByConnectedComponent(
			new double[][]{
				{0, 1},
				{1, 0},
			},
			new double[][]{
				{0, 1},
				{1, 0},
			}
		);
	}
	@Test
	public void testConnectedComponents4() {
		testClusterByConnectedComponent(
			new double[][]{
				{0, 1, 0},
				{1, 0, 1},
				{0, 1, 0},
			},
			new double[][]{
				{0, 1, 1},
				{1, 0, 1},
				{1, 1, 0},
			}
		);
	}
	@Test
	public void testConnectedComponents5() {
		testClusterByConnectedComponent(
			new double[][]{
				{0, 0, 1, 0, 0, 0},
				{0, 0, 0, 1, 0, 0},
				{1, 0, 0, 0, 1, 0},
				{0, 1, 0, 0, 0, 0},
				{0, 0, 1, 0, 0, 0},
				{0, 0, 0, 0, 0, 0},
			},
			new double[][]{
				{0, 0, 1, 0, 1, 0},
				{0, 0, 0, 1, 0, 0},
				{1, 0, 0, 0, 1, 0},
				{0, 1, 0, 0, 0, 0},
				{1, 0, 1, 0, 0, 0},
				{0, 0, 0, 0, 0, 0},
			}
		);
	}
	@Test
	public void testConnectedComponents6() {
		testClusterByConnectedComponent(
			new double[][]{
				{0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0},
			},
			new double[][]{
				{0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0},
			}
		);
	}
	@Test
	public void testConnectedComponents7() {
		testClusterByConnectedComponent(
			new double[][]{
				{0, 1, 0, 1, 0, 0, 0},
				{1, 0, 1, 1, 0, 0, 0},
				{0, 1, 0, 0, 0, 0, 0},
				{1, 1, 0, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 1, 1},
				{0, 0, 0, 0, 1, 0, 1},
				{0, 0, 0, 0, 1, 1, 0},
			},
			new double[][]{
				{0, 1, 1, 1, 0, 0, 0},
				{1, 0, 1, 1, 0, 0, 0},
				{1, 1, 0, 1, 0, 0, 0},
				{1, 1, 1, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 1, 1},
				{0, 0, 0, 0, 1, 0, 1},
				{0, 0, 0, 0, 1, 1, 0},
			}
		);
	}
	@Test
	public void testConnectedComponents8() {
		testClusterByConnectedComponent(
			new double[][]{
				{0, 1, 1, 1, 1, 1, 1},
				{1, 0, 1, 1, 1, 1, 1},
				{1, 1, 0, 1, 1, 1, 1},
				{1, 1, 1, 0, 1, 1, 1},
				{1, 1, 1, 1, 0, 1, 1},
				{1, 1, 1, 1, 1, 0, 1},
				{1, 1, 1, 1, 1, 1, 0},
			},
			new double[][]{
				{0, 1, 1, 1, 1, 1, 1},
				{1, 0, 1, 1, 1, 1, 1},
				{1, 1, 0, 1, 1, 1, 1},
				{1, 1, 1, 0, 1, 1, 1},
				{1, 1, 1, 1, 0, 1, 1},
				{1, 1, 1, 1, 1, 0, 1},
				{1, 1, 1, 1, 1, 1, 0},
			}
		);
	}

	public void testClusterByConnectedComponent(double[][] adjacencyMatrix, double[][] expectedMatrix) {
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		loadTestConfiguration(config);
		fullDMLScriptName = SCRIPT_DIR + TEST_DIR + config.getTestScript() + ".dml";
		programArgs = new String[]{"-nvargs",
			"inFile=" + input("A"), "outFile=" + output("B")};
		writeInputMatrixWithMTD("A", adjacencyMatrix, false);
		writeExpectedMatrix("B", expectedMatrix);
		runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
		compareResults(0.01);
	}
}
