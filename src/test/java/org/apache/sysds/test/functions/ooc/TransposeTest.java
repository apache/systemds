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

public class TransposeTest extends AutomatedTestBase {
	private final static String TEST_NAME1 = "Transpose";
	private final static String TEST_DIR = "functions/ooc/";
	private final static String TEST_CLASS_DIR = TEST_DIR + TransposeTest.class.getSimpleName() + "/";
	private final static double eps = 1e-10;
	private static final String INPUT_NAME = "X";
	private static final String OUTPUT_NAME = "res";

	private final static int rows = 1500;
	private final static int cols_wide = 2500;
	private final static int cols_skinny = 500;

	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.1;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		TestConfiguration config = new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1);
		addTestConfiguration(TEST_NAME1, config);
	}

	@Test
	public void testTranspose1() {
		runTransposeTest(cols_wide, false);
	}

	@Test
	public void testTranspose2() {
		runTransposeTest(cols_skinny, false);
	}

	@Test
	public void testTransposeSparse1() {
		runTransposeTest(cols_wide, true);
	}

	@Test
	public void testTransposeSparse2() {
		runTransposeTest(cols_skinny, true);
	}

	private void runTransposeTest(int cols, boolean sparse) {
		Types.ExecMode platformOld = setExecMode(Types.ExecMode.SINGLE_NODE);

		try {
			getAndLoadTestConfiguration(TEST_NAME1);
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME1 + ".dml";
			programArgs = new String[] {"-explain", "-stats", "-ooc", "-args", input(INPUT_NAME), output(OUTPUT_NAME)};

			// 1. Generate the data as MatrixBlock object
			double[][] A_data = getRandomMatrix(rows, cols, 0, 1, sparse ? sparsity2 : sparsity1, 10);

			// 2. Convert the double arrays to MatrixBlock object
			MatrixBlock A_mb = DataConverter.convertToMatrixBlock(A_data);

			// 3. Create a binary matrix writer
			MatrixWriter writer = MatrixWriterFactory.createMatrixWriter(Types.FileFormat.BINARY);

			// 4. Write matrix A to a binary SequenceFile
			writer.writeMatrixToHDFS(A_mb, input(INPUT_NAME), rows, cols, 1000, A_mb.getNonZeros());
			HDFSTool.writeMetaDataFile(input(INPUT_NAME + ".mtd"), Types.ValueType.FP64,
				new MatrixCharacteristics(rows, cols, 1000, A_mb.getNonZeros()), Types.FileFormat.BINARY);

			boolean exceptionExpected = false;
			runTest(true, exceptionExpected, null, -1);

			double[][] C1 = readMatrix(output(OUTPUT_NAME), Types.FileFormat.BINARY, cols, rows, 1000);
			double result = 0.0;
			for(int i = 0; i < rows; i++) { // verify the results with Java
				double expected = 0.0;
				for(int j = 0; j < cols; j++) {
					expected = A_mb.get(i, j);
					result = C1[j][i];
					Assert.assertEquals(expected, result, eps);
				}

			}

			String prefix = Instruction.OOC_INST_PREFIX;
			Assert.assertTrue("OOC wasn't used for RBLK", heavyHittersContainsString(prefix + Opcodes.RBLK));
			Assert.assertTrue("OOC wasn't used for TRANSPOSE", heavyHittersContainsString(prefix + Opcodes.TRANSPOSE));
		}
		catch(IOException e) {
			throw new RuntimeException(e);
		}
		finally {
			resetExecMode(platformOld);
		}
	}

	private static double[][] readMatrix(String fname, Types.FileFormat fmt, long rows, long cols, int blen)
		throws IOException {
		MatrixBlock mb = DataConverter.readMatrixFromHDFS(fname, fmt, rows, cols, blen);
		double[][] C = DataConverter.convertToDoubleMatrix(mb);
		return C;
	}
}
