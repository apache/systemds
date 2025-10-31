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

public class BinaryMatrixMatrixTest extends AutomatedTestBase {
	private final static String TEST_NAME1 = "BinaryMatrixMatrix";
	private final static String TEST_DIR = "functions/ooc/";
	private final static String TEST_CLASS_DIR = TEST_DIR + BinaryMatrixMatrixTest.class.getSimpleName() + "/";
	private final static double eps = 1e-8;
	private static final String INPUT_NAME_1 = "X";
	private static final String INPUT_NAME_2 = "Y";
	private static final String OUTPUT_NAME = "res";

	private final static int rows = 1500;
	private final static int cols = 1200;
	private final static int maxVal = 7;
	private final static double sparsity1 = 1;
	private final static double sparsity2 = 0.05;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		TestConfiguration config = new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1);
		addTestConfiguration(TEST_NAME1, config);
	}

	@Test
	public void testBinaryMatrixMatrixDenseDense() {
		runBinaryMatrixMatrixTest(false, false);
	}

	@Test
	public void testBinaryMatrixMatrixDenseSparse() {
		runBinaryMatrixMatrixTest(false, true);
	}

	@Test
	public void testBinaryMatrixMatrixSparseDense() {
		runBinaryMatrixMatrixTest(true, false);
	}

	@Test
	public void testBinaryMatrixMatrixSparseSparse() {
		runBinaryMatrixMatrixTest(true, true);
	}

	private void runBinaryMatrixMatrixTest(boolean sparse1, boolean sparse2) {
		Types.ExecMode platformOld = setExecMode(Types.ExecMode.SINGLE_NODE);

		try {
			getAndLoadTestConfiguration(TEST_NAME1);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME1 + ".dml";
			programArgs = new String[] {"-explain", "-stats", "-ooc", "-args", input(INPUT_NAME_1), input(INPUT_NAME_2), output(OUTPUT_NAME)};

			// 1. Generate the data in-memory as MatrixBlock objects
			double[][] X_data = getRandomMatrix(rows, 1, 1, maxVal, sparse1 ? sparsity2 : sparsity1, 7);
			double[][] Y_data = getRandomMatrix(rows, 1, 0, 1, sparse2 ? sparsity2 : sparsity1, 8);

			// 2. Convert the double arrays to MatrixBlock objects
			MatrixBlock X_mb = DataConverter.convertToMatrixBlock(X_data);
			MatrixBlock Y_mb = DataConverter.convertToMatrixBlock(Y_data);

			// 3. Create a binary matrix writer
			MatrixWriter writer = MatrixWriterFactory.createMatrixWriter(Types.FileFormat.BINARY);

			// 4. Write matrix A to a binary SequenceFile
			writer.writeMatrixToHDFS(X_mb, input(INPUT_NAME_1), rows, cols, 1000, X_mb.getNonZeros());
			writer.writeMatrixToHDFS(Y_mb, input(INPUT_NAME_2), rows, cols, 1000, Y_mb.getNonZeros());
			HDFSTool.writeMetaDataFile(input(INPUT_NAME_1 + ".mtd"), Types.ValueType.FP64,
				new MatrixCharacteristics(rows, cols, 1000, X_mb.getNonZeros()), Types.FileFormat.BINARY);
			HDFSTool.writeMetaDataFile(input(INPUT_NAME_2 + ".mtd"), Types.ValueType.FP64,
				new MatrixCharacteristics(rows, cols, 1000, Y_mb.getNonZeros()), Types.FileFormat.BINARY);

			runTest(true, false, null, -1);

			//check tsmm OOC
			Assert.assertTrue("OOC wasn't used for multiplication",
				heavyHittersContainsString(Instruction.OOC_INST_PREFIX + Opcodes.MULT));

			//compare results

			// rerun without ooc flag
			programArgs = new String[] {"-explain", "-stats", "-args", input(INPUT_NAME_1), input(INPUT_NAME_2), output(OUTPUT_NAME + "_target")};
			runTest(true, false, null, -1);

			// compare matrices
			MatrixBlock ret1 = DataConverter.readMatrixFromHDFS(output(OUTPUT_NAME),
				Types.FileFormat.BINARY, rows, cols, 1000);
			MatrixBlock ret2 = DataConverter.readMatrixFromHDFS(output(OUTPUT_NAME + "_target"),
				Types.FileFormat.BINARY, rows, cols, 1000);
			TestUtils.compareMatrices(ret1, ret2, eps);
		}
		catch(IOException e) {
			throw new RuntimeException(e);
		}
		finally {
			resetExecMode(platformOld);
		}
	}
}
