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
import org.apache.sysds.lops.MMTSJ.MMTSJType;
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

import java.io.IOException;

public class TransposeSelfMMTest extends AutomatedTestBase {
	private static final String TEST_NAME_LEFT = "TSMM";
	private static final String TEST_NAME_RIGHT = "TSMMRight";
	private final static String TEST_DIR = "functions/ooc/";
	private final static String TEST_CLASS_DIR = TEST_DIR + TransposeSelfMMTest.class.getSimpleName() + "/";
	private final static double eps = 1e-8;
	private static final String INPUT_NAME = "X";
	private static final String OUTPUT_NAME_CP = "res_cp";
	private static final String OUTPUT_NAME_OOC = "res_ooc";

	private static final int SINGLE_TILE_ROWS = 2143;
	private static final int SINGLE_TILE_COLS = 123;
	private static final int SINGLE_TILE_BLOCK_SIZE = 1000;
	private static final int MULTI_TILE_ROWS = 1003;
	private static final int MULTI_TILE_COLS = 1007;
	private static final int MULTI_TILE_BLOCK_SIZE = 1000;
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.1;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME_LEFT, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_LEFT));
		addTestConfiguration(TEST_NAME_RIGHT, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_RIGHT));
	}

	@Test
	public void testTsmmLeftDenseSingleTile() {
		runTSMMTest(MMTSJType.LEFT, SINGLE_TILE_ROWS, SINGLE_TILE_COLS, SINGLE_TILE_BLOCK_SIZE, false);
	}

	@Test
	public void testTsmmLeftSparseSingleTile() {
		runTSMMTest(MMTSJType.LEFT, SINGLE_TILE_ROWS, SINGLE_TILE_COLS, SINGLE_TILE_BLOCK_SIZE, true);
	}

	@Test
	public void testTsmmLeftDenseMultiTile() {
		runTSMMTest(MMTSJType.LEFT, MULTI_TILE_ROWS, MULTI_TILE_COLS, MULTI_TILE_BLOCK_SIZE, false);
	}

	@Test
	public void testTsmmRightDenseMultiTile() {
		runTSMMTest(MMTSJType.RIGHT, MULTI_TILE_ROWS, MULTI_TILE_COLS, MULTI_TILE_BLOCK_SIZE, false);
	}

	private void runTSMMTest(MMTSJType type, int rows, int cols, int blockSize, boolean sparse) {
		Types.ExecMode platformOld = setExecMode(Types.ExecMode.SINGLE_NODE);

		try {
			String testName = type.isLeft() ? TEST_NAME_LEFT : TEST_NAME_RIGHT;
			getAndLoadTestConfiguration(testName);
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testName + ".dml";

			double[][] A_data = getRandomMatrix(rows, cols, 0, 1, sparse?sparsity2:sparsity1, 10);
			MatrixBlock A_mb = DataConverter.convertToMatrixBlock(A_data);
			MatrixWriter writer = MatrixWriterFactory.createMatrixWriter(Types.FileFormat.BINARY);
			writer.writeMatrixToHDFS(A_mb, input(INPUT_NAME), rows, cols, blockSize, A_mb.getNonZeros());
			HDFSTool.writeMetaDataFile(input(INPUT_NAME + ".mtd"), Types.ValueType.FP64,
				new MatrixCharacteristics(rows, cols, blockSize, A_mb.getNonZeros()), Types.FileFormat.BINARY);

			programArgs = new String[] {"-stats", "-args", input(INPUT_NAME), output(OUTPUT_NAME_CP)};
			runTest(true, false, null, -1);

			programArgs = new String[] {"-explain", "-stats", "-ooc",
				"-args", input(INPUT_NAME), output(OUTPUT_NAME_OOC)};
			runTest(true, false, null, -1);

			Assert.assertTrue("OOC wasn't used for TSMM",
				heavyHittersContainsString(Instruction.OOC_INST_PREFIX + Opcodes.TSMM));

			int outputDim = type.isLeft() ? cols : rows;
			MatrixBlock actual = DataConverter.readMatrixFromHDFS(output(OUTPUT_NAME_OOC),
				Types.FileFormat.BINARY, outputDim, outputDim, blockSize);
			MatrixBlock expected = DataConverter.readMatrixFromHDFS(output(OUTPUT_NAME_CP),
				Types.FileFormat.BINARY, outputDim, outputDim, blockSize);
			TestUtils.compareMatrices(actual, expected, eps);
		}
		catch (IOException e) {
			throw new RuntimeException(e);
		}
		finally {
			resetExecMode(platformOld);
		}
	}
}
