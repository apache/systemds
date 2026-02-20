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

import java.io.IOException;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class PNMFTest extends AutomatedTestBase {
	private static final String TEST_NAME = "PNMF";
	private static final String TEST_DIR = "functions/ooc/";
	private static final String TEST_CLASS_DIR = TEST_DIR + PNMFTest.class.getSimpleName() + "/";

	private static final String INPUT_X = "X";
	private static final String OUTPUT_W_OOC = "W";
	private static final String OUTPUT_H_OOC = "H";
	private static final String OUTPUT_W_CP = "W_cp";
	private static final String OUTPUT_H_CP = "H_cp";

	private static final int ROWS = 1468;
	private static final int COLS = 1207;
	private static final int RANK = 20;
	private static final int MAX_ITER = 10;
	private static final int BLOCK_SIZE = 1000;

	private static final double SPARSITY = 0.7;
	private static final double EPS = 1e-6;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME));
	}

	//@Test
	public void testPNMFOOCVsCP() {
		runPNMFTest();
	}

	private void runPNMFTest() {
		Types.ExecMode platformOld = setExecMode(Types.ExecMode.SINGLE_NODE);

		try {
			getAndLoadTestConfiguration(TEST_NAME);

			String home = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = home + TEST_NAME + ".dml";

			double[][] xData = getRandomMatrix(ROWS, COLS, 1, 10, SPARSITY, 7);
			writeBinaryWithMTD(INPUT_X, DataConverter.convertToMatrixBlock(xData));

			programArgs = new String[] {"-explain", "-stats", "-seed", "7", "-ooc", "-args",
				input(INPUT_X), String.valueOf(RANK), String.valueOf(MAX_ITER),
				output(OUTPUT_W_OOC), output(OUTPUT_H_OOC)};
			runTest(true, false, null, -1);

			programArgs = new String[] {"-explain", "-stats", "-seed", "7", "-args",
				input(INPUT_X), String.valueOf(RANK), String.valueOf(MAX_ITER),
				output(OUTPUT_W_CP), output(OUTPUT_H_CP)};
			runTest(true, false, null, -1);

			MatrixBlock wOOC = DataConverter.readMatrixFromHDFS(output(OUTPUT_W_OOC),
				Types.FileFormat.BINARY, ROWS, RANK, BLOCK_SIZE);
			MatrixBlock hOOC = DataConverter.readMatrixFromHDFS(output(OUTPUT_H_OOC),
				Types.FileFormat.BINARY, RANK, COLS, BLOCK_SIZE);

			MatrixBlock wCP = DataConverter.readMatrixFromHDFS(output(OUTPUT_W_CP),
				Types.FileFormat.BINARY, ROWS, RANK, BLOCK_SIZE);
			MatrixBlock hCP = DataConverter.readMatrixFromHDFS(output(OUTPUT_H_CP),
				Types.FileFormat.BINARY, RANK, COLS, BLOCK_SIZE);

			TestUtils.compareMatrices(wOOC, wCP, EPS);
			TestUtils.compareMatrices(hOOC, hCP, EPS);
		}
		catch(IOException e) {
			throw new RuntimeException(e);
		}
		finally {
			resetExecMode(platformOld);
		}
	}
}
