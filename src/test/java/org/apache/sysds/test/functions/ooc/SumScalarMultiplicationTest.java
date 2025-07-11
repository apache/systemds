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

package org.apache.sysds.test.functions.ooc;

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;
import org.junit.Assert;
import org.junit.Test;

import java.util.HashMap;

public class SumScalarMultiplicationTest extends AutomatedTestBase {

	private static final String TEST_NAME = "SumScalarMultiplication";
	private static final String TEST_DIR = "functions/ooc/";
	private static final String TEST_CLASS_DIR = TEST_DIR + SumScalarMultiplicationTest.class.getSimpleName() + "/";
	private static final String INPUT_NAME = "X";
	private static final String OUTPUT_NAME = "res";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		TestConfiguration config = new TestConfiguration(TEST_CLASS_DIR, TEST_NAME);
		addTestConfiguration(TEST_NAME, config);
	}

	/**
	 * Test the sum of scalar multiplication, "sum(X*7)", with OOC backend.
	 */
	@Test
	public void testSumScalarMult() {

		Types.ExecMode platformOld = rtplatform;
		rtplatform = Types.ExecMode.SINGLE_NODE;

		try {
			getAndLoadTestConfiguration(TEST_NAME);
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-explain", "-stats", "-ooc", "-args", input(INPUT_NAME), output(OUTPUT_NAME)};

			int rows = 3;
			int cols = 4;
			double sparsity = 0.8;

			double[][] X = getRandomMatrix(rows, cols, -1, 1, sparsity, 7);
			writeInputMatrixWithMTD(INPUT_NAME, X, true);

			runTest(true, false, null, -1);

			HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir(OUTPUT_NAME);
			// only one entry
			Double result = dmlfile.get(new MatrixValue.CellIndex(1, 1));

			double expected = 0.0;
			for(int i = 0; i < rows; i++) {
				for(int j = 0; j < cols; j++) {
					expected += X[i][j] * 7;
				}
			}

			Assert.assertEquals(expected, result, 1e-10);

			String prefix = Instruction.OOC_INST_PREFIX;

			boolean usedOOCMult = Statistics.getCPHeavyHitterOpCodes().contains(prefix + Opcodes.MULT);
			Assert.assertTrue("OOC wasn't used for MULT", usedOOCMult);

			boolean usedOOCSum = Statistics.getCPHeavyHitterOpCodes().contains(prefix + Opcodes.UAKP);
			Assert.assertTrue("OOC wasn't used for SUM", usedOOCSum);

		}
		finally {
			// reset
			rtplatform = platformOld;
		}
	}
}
