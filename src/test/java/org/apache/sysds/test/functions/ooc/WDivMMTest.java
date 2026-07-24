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
import java.util.Arrays;
import java.util.Collection;

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
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class WDivMMTest extends AutomatedTestBase {
	private final static String INPUT_NAME_1 = "W";
	private final static String INPUT_NAME_2 = "U";
	private final static String INPUT_NAME_3 = "V";
	private final static String OUTPUT_NAME = "R";
	private static final String TEST_DIR = "functions/ooc/";
	private static final String TEST_CLASS_DIR = TEST_DIR + WDivMMTest.class.getSimpleName() + "/";

	private static final int rows = 2201;
	private static final int cols = 1103;
	private static final int rank = 20;
	private static final int blen = 1000;
	private static final double eps = 1e-6;
	private static final double div_eps = 0.1;

	private final static String TEST_NAME_1 = "WeightedDivMMLeft";
	private final static String TEST_NAME_2 = "WeightedDivMMRight";
	private final static String TEST_NAME_3 = "WeightedDivMMMultBasic";
	private final static String TEST_NAME_4 = "WeightedDivMMMultLeft";
	private final static String TEST_NAME_5 = "WeightedDivMMMultRight";
	private final static String TEST_NAME_6 = "WeightedDivMMMultMinusLeft";
	private final static String TEST_NAME_7 = "WeightedDivMMMultMinusRight";
	private final static String TEST_NAME_8 = "WeightedDivMM4MultMinusLeft";
	private final static String TEST_NAME_9 = "WeightedDivMM4MultMinusRight";
	private final static String TEST_NAME_10 = "WeightedDivMMLeftEps";
	private final static String TEST_NAME_11 = "WeightedDivMMRightEps";
	private String TEST_NAME;

	public WDivMMTest(String testName) {
		this.TEST_NAME = testName;
	}

	@Parameterized.Parameters(name = "{0}")
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] {{TEST_NAME_1}, {TEST_NAME_2}, {TEST_NAME_3}, {TEST_NAME_4}, {TEST_NAME_5},
			{TEST_NAME_6}, {TEST_NAME_7}, {TEST_NAME_8}, {TEST_NAME_9}, {TEST_NAME_10}, {TEST_NAME_11}});
	}

	@Before
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {OUTPUT_NAME}));
	}

	@Test
	public void testWeightedDivMM() {
		runWeightedDivMMTest(TEST_NAME);
	}

	private void runWeightedDivMMTest(String TEST_NAME) {
		Types.ExecMode platformOld = setExecMode(Types.ExecMode.SINGLE_NODE);

		try {
			boolean basic = TEST_NAME.equals(TEST_NAME_3);
			boolean left = TEST_NAME.equals(TEST_NAME_1) || TEST_NAME.equals(TEST_NAME_4) ||
				TEST_NAME.equals(TEST_NAME_6) || TEST_NAME.equals(TEST_NAME_8) || TEST_NAME.equals(TEST_NAME_10);

			getAndLoadTestConfiguration(TEST_NAME);
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";

			double[][] W = getRandomMatrix(rows, cols, 0, 1, 0.7, 7);
			double[][] U = getRandomMatrix(rows, rank, 0, 1, 1.0, 713);
			double[][] V = getRandomMatrix(cols, rank, 0, 1, 1.0, 812);

			MatrixWriter writer = MatrixWriterFactory.createMatrixWriter(Types.FileFormat.BINARY);
			writer.writeMatrixToHDFS(DataConverter.convertToMatrixBlock(W), input(INPUT_NAME_1), rows,
				cols, blen, rows * cols);
			writer.writeMatrixToHDFS(DataConverter.convertToMatrixBlock(U), input(INPUT_NAME_2), rows,
				rank, blen, rows * rank);
			writer.writeMatrixToHDFS(DataConverter.convertToMatrixBlock(V), input(INPUT_NAME_3), cols,
				rank, blen, cols * rank);

			HDFSTool.writeMetaDataFile(input(INPUT_NAME_1 + ".mtd"), Types.ValueType.FP64,
				new MatrixCharacteristics(rows, cols, blen, rows * cols), Types.FileFormat.BINARY);
			HDFSTool.writeMetaDataFile(input(INPUT_NAME_2 + ".mtd"), Types.ValueType.FP64,
				new MatrixCharacteristics(rows, rank, blen, rows * rank), Types.FileFormat.BINARY);
			HDFSTool.writeMetaDataFile(input(INPUT_NAME_3 + ".mtd"), Types.ValueType.FP64,
				new MatrixCharacteristics(cols, rank, blen, cols * rank), Types.FileFormat.BINARY);

			programArgs = new String[] {"-ooc", "-stats", "-explain", "runtime", "-args", input(INPUT_NAME_1),
				input(INPUT_NAME_2), input(INPUT_NAME_3), output(OUTPUT_NAME), Double.toString(div_eps)};

			runTest(true, false, null, -1);

			Assert.assertTrue("OOC wasn't used for wdivmm",
				heavyHittersContainsString(Instruction.OOC_INST_PREFIX + Opcodes.WEIGHTEDDIVMM));

			programArgs = new String[] {"-stats", "-explain", "runtime", "-args", input(INPUT_NAME_1),
				input(INPUT_NAME_2), input(INPUT_NAME_3), output(OUTPUT_NAME + "_target"), Double.toString(div_eps)};

			runTest(true, false, null, -1);

			int rows2 = left ? cols : rows;
			int cols2 = basic ? cols : rank;
			checkDMLMetaDataFile("R", new MatrixCharacteristics(rows2, cols2));

			MatrixBlock actual = DataConverter.readMatrixFromHDFS(output(OUTPUT_NAME),
				Types.FileFormat.BINARY, rows2, cols2, blen);
			MatrixBlock expected = DataConverter.readMatrixFromHDFS(output(OUTPUT_NAME + "_target"),
				Types.FileFormat.BINARY, rows2, cols2, blen);
			TestUtils.compareMatrices(expected, actual, eps);
		}
		catch(IOException e) {
			throw new RuntimeException(e);
		}
		finally {
			resetExecMode(platformOld);
		}
	}
}
