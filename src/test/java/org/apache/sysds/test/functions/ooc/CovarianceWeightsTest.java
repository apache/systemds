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

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.io.MatrixWriter;
import org.apache.sysds.runtime.io.MatrixWriterFactory;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

public class CovarianceWeightsTest extends AutomatedTestBase {
	private final static String TEST_NAME = "CovarianceWeights";
	private final static String TEST_DIR = "functions/ooc/";
	private final static String TEST_CLASS_DIR = TEST_DIR + CovarianceWeightsTest.class.getSimpleName() + "/";
	private final static double eps = 1e-10;

	private final static String INPUT_A = "A";
	private final static String INPUT_B = "B";
	private final static String INPUT_W = "W";
	private final static String OUTPUT_CP = "R_CP";
	private final static String OUTPUT_OOC = "R_OOC";

	private final static int rows = 1871;
	private final static int cols = 1;
	private final static int blocksize = 1000;
	private final static int maxVal = 7;

	private final static double denseSparsity = 0.65;
	private final static double sparseSparsity = 0.05;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME,
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { OUTPUT_CP, OUTPUT_OOC }));
	}

	@Test
	public void testWeightedCovarianceDenseOOC() {
		runWeightedCovarianceOOCCompareTest(false);
	}

	@Test
	public void testWeightedCovarianceSparseOOC() {
		runWeightedCovarianceOOCCompareTest(true);
	}

	private void runWeightedCovarianceOOCCompareTest(boolean sparse) {
		Types.ExecMode platformOld = setExecMode(Types.ExecMode.SINGLE_NODE);

		try {
			getAndLoadTestConfiguration(TEST_NAME);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";

			double sparsity = sparse ? sparseSparsity : denseSparsity;

			double[][] A = getRandomMatrix(rows, cols, 1, maxVal, sparsity, 7);
			double[][] B = getRandomMatrix(rows, cols, 1, maxVal, sparsity, 823);

			// Weights should be positive. Avoid zero/negative weights.
			double[][] W = getRandomMatrix(rows, cols, 1, maxVal, sparsity, 1234);

			MatrixBlock ABlock = DataConverter.convertToMatrixBlock(A);
			MatrixBlock BBlock = DataConverter.convertToMatrixBlock(B);
			MatrixBlock WBlock = DataConverter.convertToMatrixBlock(W);

			writeBinaryMatrix(INPUT_A, ABlock, rows, cols, blocksize);
			writeBinaryMatrix(INPUT_B, BBlock, rows, cols, blocksize);
			writeBinaryMatrix(INPUT_W, WBlock, rows, cols, blocksize);

			// Reference run: normal single-node CP execution.
			programArgs = new String[] {
				"-args",
				input(INPUT_A),
				input(INPUT_B),
				input(INPUT_W),
				output(OUTPUT_CP)
			};
			runTest(true, false, null, -1);

			// OOC run: compare the out-of-core weighted covariance path against CP.
			programArgs = new String[] {
				"-explain", "-stats", "-ooc",
				"-args",
				input(INPUT_A),
				input(INPUT_B),
				input(INPUT_W),
				output(OUTPUT_OOC)
			};
			runTest(true, false, null, -1);

			Assert.assertTrue("OOC wasn't used for weighted covariance",
				heavyHittersContainsString(Instruction.OOC_INST_PREFIX + Opcodes.COV));

			MatrixBlock cpResult = DataConverter.readMatrixFromHDFS(
				output(OUTPUT_CP), Types.FileFormat.BINARY, 1, 1, blocksize, 1);

			MatrixBlock oocResult = DataConverter.readMatrixFromHDFS(
				output(OUTPUT_OOC), Types.FileFormat.BINARY, 1, 1, blocksize, 1);

			TestUtils.compareMatrices(cpResult, oocResult, eps);
		}
		catch(IOException ex) {
			throw new RuntimeException(ex);
		}
		finally {
			resetExecMode(platformOld);
		}
	}

	private void writeBinaryMatrix(String name, MatrixBlock mb, int rows, int cols, int blocksize) throws IOException {
		MatrixWriter writer = MatrixWriterFactory.createMatrixWriter(Types.FileFormat.BINARY);
		writer.writeMatrixToHDFS(mb, input(name), rows, cols, blocksize, mb.getNonZeros());

		HDFSTool.writeMetaDataFile(input(name + ".mtd"),
			Types.ValueType.FP64,
			new MatrixCharacteristics(rows, cols, blocksize, mb.getNonZeros()),
			Types.FileFormat.BINARY);
	}
}
