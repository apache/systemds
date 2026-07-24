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

package org.apache.sysds.test.functions.rewrite;

import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.util.HashMap;

public class RewriteRaJoinTest extends AutomatedTestBase {
	private final static String TEST_NAME = "raJoin";
	private final static String TEST_DIR = "functions/rewrite/";
	private final static String TEST_CLASS_DIR = TEST_DIR + RewriteRaJoinTest.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "OUT" }));
	}

	@Test
	public void testRaJoin() {
		// Load test configuration (sets up temp input/output folders)
		getAndLoadTestConfiguration(TEST_NAME);

		// Create inputs
		double[][] A = {{1,1},{1,1}}; 
		double[][] B = {{3,2,1},{1,2,3},{3,1,2}}; 
		double[][] C = {	
			{1,1,1,1},
			{2,2,2,2},
			{3,3,3,3},
			{4,4,4,4}
		};

		MatrixCharacteristics mcA = new MatrixCharacteristics(2,2,-1,-1);
		writeInputMatrixWithMTD("A", A, true, mcA);

		MatrixCharacteristics mcB = new MatrixCharacteristics(3,3,-1,-1);
		writeInputMatrixWithMTD("B", B, true, mcB);

		MatrixCharacteristics mcC = new MatrixCharacteristics(4,4,-1,-1);
		writeInputMatrixWithMTD("C", C, true, mcC);

		programArgs = new String[] {
			"-explain", "hops",
			"-stats",
			"-args",
			input("A"),
			input("B"),
			input("C"),
			output("OUT")
		};

		fullDMLScriptName = SCRIPT_DIR + TEST_DIR + TEST_NAME + ".dml";

		// Execute single threaded
		ExecMode oldPlatform = setExecMode(ExecMode.SINGLE_NODE);
		try {
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);

			HashMap<CellIndex, Double> out = readDMLMatrixFromOutputDir("OUT");

			System.out.println("Result matrix:");
			for (CellIndex idx : out.keySet()) {
				System.out.println(idx + " -> " + out.get(idx));
			}

			double[][] expected = {
				{1,1,3,2,1,2,2,2,2},
				{1,1,3,2,1,2,2,2,2}
			};
			HashMap<CellIndex, Double> expectedMap = TestUtils.convert2DDoubleArrayToHashMap(expected);
			TestUtils.compareMatrices(expectedMap, out, 1e-10, "expected", "actual");

		} finally {
			rtplatform = oldPlatform;
		}
	}
}
