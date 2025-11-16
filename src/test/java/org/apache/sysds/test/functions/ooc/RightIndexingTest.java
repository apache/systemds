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

public class RightIndexingTest extends AutomatedTestBase {
	private final static String TEST_NAME1 = "RightIndexing";
	private final static String TEST_DIR = "functions/ooc/";
	private final static String TEST_CLASS_DIR = TEST_DIR + RightIndexingTest.class.getSimpleName() + "/";
	private final static double eps = 1e-8;
	private static final String INPUT_NAME = "X";
	private static final String OUTPUT_NAME = "res";

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
	public void testRightIndexingDense1() {
		runRightIndexingTest(2, 2002, 100, 1150, 2100, 1200, false);
	}

	@Test
	public void testRightIndexingSparse1() {
		runRightIndexingTest(2, 2002, 100, 1150, 2100, 1200, true);
	}

	@Test
	public void testRightIndexingAlignedDense() {
		runRightIndexingTest(1, 2002, 1, 1150, 2100, 1200, false);
	}

	@Test
	public void testRightIndexingAlignedSparse() {
		runRightIndexingTest(1, 2002, 1, 1150, 2100, 1200, true);
	}

	@Test
	public void testRightIndexingRowAlignedDense() {
		runRightIndexingTest(1, 2002, 100, 1150, 2100, 1200, false);
	}

	@Test
	public void testRightIndexingRowAlignedSparse() {
		runRightIndexingTest(1, 2002, 100, 1150, 2100, 1200, true);
	}

	@Test
	public void testRightIndexingSmallDense1() {
		runRightIndexingTest(1, 700, 150, 1020, 3000, 3000, false);
	}

	@Test
	public void testRightIndexingSmallSparse1() {
		runRightIndexingTest(1, 700, 150, 1020, 3000, 3000, true);
	}

	@Test
	public void testRightIndexingSmallDense2() {
		runRightIndexingTest(150, 1020, 1, 700, 3000, 3000, false);
	}

	@Test
	public void testRightIndexingSmallSparse2() {
		runRightIndexingTest(150, 1020, 1, 700, 3000, 3000, true);
	}

	@Test
	public void testRightIndexingSingleElementDense() {
		runRightIndexingTest(1111, 1111, 2222, 2222, 3000, 3000, false);
	}

	@Test
	public void testRightIndexingSingleElementSparse() {
		runRightIndexingTest(1111, 1111, 2222, 2222, 3000, 3000, true);
	}

	@Test
	public void testRightIndexingCrossBlockBothDense() {
		runRightIndexingTest(950, 1050, 995, 1005, 3000, 3000, false);
	}

	@Test
	public void testRightIndexingCrossBlockBothSparse() {
		runRightIndexingTest(950, 1050, 995, 1005, 3000, 3000, true);
	}

	@Test
	public void testRightIndexingSingleRowMultiBlockDense() {
		runRightIndexingTest(1001, 1001, 800, 1205, 3000, 3000, false);
	}

	@Test
	public void testRightIndexingSingleRowMultiBlockSparse() {
		runRightIndexingTest(1001, 1001, 800, 1205, 3000, 3000, true);
	}

	@Test
	public void testRightIndexingSingleColumnMultiBlockDense() {
		runRightIndexingTest(800, 1205, 1001, 1001, 3000, 3000, false);
	}

	@Test
	public void testRightIndexingSingleColumnMultiBlockSparse() {
		runRightIndexingTest(800, 1205, 1001, 1001, 3000, 3000, true);
	}

	@Test
	public void testRightIndexingTrailingBlocksDense() {
		runRightIndexingTest(2501, 3000, 1500, 2100, 3000, 3000, false);
	}

	@Test
	public void testRightIndexingTrailingBlocksSparse() {
		runRightIndexingTest(2501, 3000, 1500, 2100, 3000, 3000, true);
	}

	private void runRightIndexingTest(int rs, int re, int cs, int ce, int nrows, int ncols, boolean sparse) {
		Types.ExecMode platformOld = setExecMode(Types.ExecMode.SINGLE_NODE);

		try {
			getAndLoadTestConfiguration(TEST_NAME1);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME1 + ".dml";
			programArgs = new String[] {"-explain", "-stats", "-ooc", "-args", input(INPUT_NAME), "" + rs, "" + re, "" + cs, "" + ce, output(OUTPUT_NAME)};

			// 1. Generate the data in-memory as MatrixBlock objects
			double[][] X_data = getRandomMatrix(nrows, ncols, 1, maxVal, sparse ? sparsity2 : sparsity1, 7);

			// 2. Convert the double arrays to MatrixBlock objects
			MatrixBlock X_mb = DataConverter.convertToMatrixBlock(X_data);

			// 3. Create a binary matrix writer
			MatrixWriter writer = MatrixWriterFactory.createMatrixWriter(Types.FileFormat.BINARY);

			// 4. Write matrix A to a binary SequenceFile
			writer.writeMatrixToHDFS(X_mb, input(INPUT_NAME), nrows, ncols, 1000, X_mb.getNonZeros());
			HDFSTool.writeMetaDataFile(input(INPUT_NAME + ".mtd"), Types.ValueType.FP64,
				new MatrixCharacteristics(nrows, ncols, 1000, X_mb.getNonZeros()), Types.FileFormat.BINARY);

			runTest(true, false, null, -1);

			//check tsmm OOC
			Assert.assertTrue("OOC wasn't used for multiplication",
				heavyHittersContainsString(Instruction.OOC_INST_PREFIX + Opcodes.RIGHT_INDEX));

			//compare results

			// rerun without ooc flag
			programArgs = new String[] {"-explain", "-stats", "-args", input(INPUT_NAME), "" + rs, "" + re, "" + cs, "" + ce, output(OUTPUT_NAME + "_target")};
			runTest(true, false, null, -1);

			// compare matrices
			int outNRows = re-rs+1;
			int outNCols = ce-cs+1;

			MatrixBlock ret1 = DataConverter.readMatrixFromHDFS(output(OUTPUT_NAME),
				Types.FileFormat.BINARY, outNRows, outNCols, 1000);
			MatrixBlock ret2 = DataConverter.readMatrixFromHDFS(output(OUTPUT_NAME + "_target"),
				Types.FileFormat.BINARY, outNRows, outNCols, 1000);

			//System.out.println(ret1.getNumRows() + "x" + ret1.getNumColumns() + " <=> " + ret2.getNumRows() + "x" + ret2.getNumColumns());
			/*System.out.println(ret1.slice(998, 1000, 901, 910));
			System.out.println(ret2.slice(998, 1000, 901, 910));*/
			TestUtils.compareMatrices(ret2, ret1, eps);
		}
		catch(IOException e) {
			throw new RuntimeException(e);
		}
		finally {
			resetExecMode(platformOld);
		}
	}
}
