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
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

import java.io.IOException;
import java.util.HashMap;

public class CentralMomentTest extends AutomatedTestBase {
	private final static String TEST_NAME1 = "CentralMoment";
	private final static String TEST_DIR = "functions/ooc/";
	private final static String TEST_CLASS_DIR = TEST_DIR + CentralMomentTest.class.getSimpleName() + "/";
	private final static double eps = 1e-8;
	private static final String INPUT_NAME = "X";
	private static final String OUTPUT_NAME = "res";

	private final static int rows = 1871;
	private final static int maxVal = 7;
	private final static double sparsity1 = 0.65;
	private final static double sparsity2 = 0.05;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		TestConfiguration config = new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1);
		addTestConfiguration(TEST_NAME1, config);
	}

	@Test
	public void testCentralMoment2Dense() {
		runCentralMomentTest(2, false);
	}

	@Test
	public void testCentralMoment3Dense() {
		runCentralMomentTest(3, false);
	}

	@Test
	public void testCentralMoment4Dense() {
		runCentralMomentTest(4, false);
	}

	@Test
	public void testCentralMoment2Sparse() {
		runCentralMomentTest(2, true);
	}

	@Test
	public void testCentralMoment3Sparse() {
		runCentralMomentTest(3, true);
	}

	@Test
	public void testCentralMoment4Sparse() {
		runCentralMomentTest(4, true);
	}

	private void runCentralMomentTest(int order, boolean sparse) {
		Types.ExecMode platformOld = setExecMode(Types.ExecMode.SINGLE_NODE);

		try {
			getAndLoadTestConfiguration(TEST_NAME1);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME1 + ".dml";
			programArgs = new String[] {"-explain", "-stats", "-ooc", "-args", input(INPUT_NAME),
				Integer.toString(order), output(OUTPUT_NAME)};

			// 1. Generate the data in-memory as MatrixBlock objects
			double[][] A_data = getRandomMatrix(rows, 1, 1, maxVal, sparse ? sparsity2 : sparsity1, 7);

			// 2. Convert the double arrays to MatrixBlock objects
			MatrixBlock A_mb = DataConverter.convertToMatrixBlock(A_data);

			// 3. Create a binary matrix writer
			MatrixWriter writer = MatrixWriterFactory.createMatrixWriter(Types.FileFormat.BINARY);

			// 4. Write matrix A to a binary SequenceFile
			writer.writeMatrixToHDFS(A_mb, input(INPUT_NAME), rows, 1, 1000, A_mb.getNonZeros());
			HDFSTool.writeMetaDataFile(input(INPUT_NAME + ".mtd"), Types.ValueType.FP64,
				new MatrixCharacteristics(rows, 1, 1000, A_mb.getNonZeros()), Types.FileFormat.BINARY);

			runTest(true, false, null, -1);

			//check Central Moment OOC
			Assert.assertTrue("OOC wasn't used for CentralMoment",
				heavyHittersContainsString(Instruction.OOC_INST_PREFIX + Opcodes.CM));

			//compare results

			// rerun without ooc flag
			programArgs = new String[] {"-explain", "-stats", "-args", input(INPUT_NAME), Integer.toString(order),
				output(OUTPUT_NAME + "_target")};
			runTest(true, false, null, -1);

			// compare matrices
			HashMap<MatrixValue.CellIndex, Double> ret1 = readDMLMatrixFromOutputDir(OUTPUT_NAME);
			HashMap<MatrixValue.CellIndex, Double> ret2 = readDMLMatrixFromOutputDir(OUTPUT_NAME + "_target");
			TestUtils.compareMatrices(ret1, ret2, eps, "Ret-1", "Ret-2");
		}
		catch(IOException e) {
			throw new RuntimeException(e);
		}
		finally {
			resetExecMode(platformOld);
		}
	}
}
