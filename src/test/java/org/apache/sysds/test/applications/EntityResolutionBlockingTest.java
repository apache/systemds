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

public class EntityResolutionBlockingTest extends AutomatedTestBase {
	private final static String TEST_NAME = "EntityResolutionBlocking";
	private final static String TEST_DIR = "applications/entity_resolution/blocking/";

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_DIR, "blocking_naive", new String[]{"B"}));
	}

	@Test
	public void testNaive1() {
		testNaiveBlocking(
			new double[][]{{0,},}, 
			10,
			new double[][]{{1,},{2,},}
		);
	}
	@Test
	public void testNaive2() {
		testNaiveBlocking(
			new double[][]{{0,},{1,},},
			1,
			new double[][]{{1,},{3,},}
		);
	}
	@Test
	public void testNaive3() {
		testNaiveBlocking(
			new double[][]{{0,},{1,},},
			2,
			new double[][]{{1,},{2,},{3,},});
	}
	@Test
	public void testNaive4() {
		testNaiveBlocking(
			new double[][]{{0,},{1,},{2,},},
			2,
			new double[][]{{1,}, {3,},{4,},});
	}

	public void testNaiveBlocking(double[][] dataset, int targetNumBlocks, double[][] expectedBlockingIndices) {
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		loadTestConfiguration(config);
		fullDMLScriptName = SCRIPT_DIR + TEST_DIR + config.getTestScript() + ".dml";

		programArgs = new String[]{
				"-nvargs", //
				"inFile=" + input("A"), //
				"outFile=" + output("B"), //
				"targetNumBlocks=" + targetNumBlocks
		};
		writeInputMatrixWithMTD("A", dataset, false);
		writeExpectedMatrix("B", expectedBlockingIndices);
		runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
		compareResults();
	}
}
