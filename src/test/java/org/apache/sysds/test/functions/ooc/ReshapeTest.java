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
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.DMLRuntimeException;
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
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.ArrayList;

@RunWith(Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class ReshapeTest extends AutomatedTestBase {
	private final static String TEST_NAME1 = "MatrixReshapeRowWise";
	private final static String TEST_NAME2 = "MatrixReshapeColWise";
	private final static String TEST_DIR = "functions/ooc/";
	private static final String TEST_CLASS_DIR = TEST_DIR + ReshapeTest.class.getSimpleName() + "/";
	private static final String INPUT_NAME = "X";
	private static final String OUTPUT_NAME = "Y";
	private static final double eps = 1e-8;
	private static final int blen = 1000;

	private final int rlen;
	private final int clen;
	private final int rows;
	private final int cols;
	private final boolean rowWise;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2));
	}

	public ReshapeTest(int rlen, int clen, int rows, int cols, boolean rowWise) {
		this.rlen = rlen;
		this.clen = clen;
		this.rows = rows;
		this.cols = cols;
		this.rowWise = rowWise;
	}

	@Parameterized.Parameters(name = "{0}x{1} {2}x{3} rowWise {4}")
	public static Iterable<Object[]> getParams() {

		int[][][] dims = {
			{{1000, 1000}, {1, 1000000}},	// single row/col
			{{3000, 4000}, {1500, 8000}},	// partialBlocks
			{{2400, 1400}, {800, 4200}}		// fullBlocks
		};

		ArrayList<Object[]> params = new ArrayList<>();

		for(int[][] d : dims) {
			params.add(new Object[] {d[0][0], d[0][1], d[1][0], d[1][1], true});
			params.add(new Object[] {d[1][0], d[1][1], d[0][0], d[0][1], true});

			params.add(new Object[] {d[0][1], d[0][0], d[1][1], d[1][0], false});
			params.add(new Object[] {d[1][1], d[1][0], d[0][1], d[0][0], false});
		}

		for(boolean rowWise : new boolean[] {true, false}) {
			// single block
			params.add(new Object[] {400, 300, 300, 400, rowWise});
			// non matching dims
			params.add(new Object[] {1400, 1000, 5000, 1, rowWise});
			// no change
			params.add(new Object[] {300, 400, 300, 400, rowWise});
		}

		return params;
	}

	@Test
	public void runTestMatrixReshapeOOC() {
		ExecMode platformOld = setExecMode(ExecMode.SINGLE_NODE);

		try {
			String TEST_NAME = (rowWise) ? TEST_NAME1 : TEST_NAME2;
			getAndLoadTestConfiguration(TEST_NAME);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";

			double[][] X = getRandomMatrix(rlen, clen, 0, 1, 1, 7);
			MatrixWriter writer = MatrixWriterFactory.createMatrixWriter(Types.FileFormat.BINARY);
			writer.writeMatrixToHDFS(DataConverter.convertToMatrixBlock(X), input(INPUT_NAME), rlen, clen, 1000, rlen * clen);
			HDFSTool.writeMetaDataFile(input(INPUT_NAME + ".mtd"), Types.ValueType.FP64,
				new MatrixCharacteristics(rlen, clen, blen, rlen * clen), Types.FileFormat.BINARY);

			programArgs = new String[] {"-explain", "-stats", "-ooc", "-args", input(INPUT_NAME), String.valueOf(rlen),
				String.valueOf(clen), String.valueOf(rows), String.valueOf(cols), output(OUTPUT_NAME)};

			if(rlen * clen != rows * cols) {
				runTest(true, true, DMLRuntimeException.class, -1);
				return;
			}

			runTest(true, false, null, -1);
			if(rlen != rows)
				Assert.assertTrue("OOC wasn't used for reshape",
					heavyHittersContainsString(Instruction.OOC_INST_PREFIX + Opcodes.RESHAPE));
			else
				Assert.assertTrue("OOC RBLK wasn't used for unchanged dimensions",
					heavyHittersContainsString(Instruction.OOC_INST_PREFIX + Opcodes.RBLK));

			// rerun without ooc flag
			programArgs = new String[] {"-explain", "-stats", "-args", input(INPUT_NAME), String.valueOf(rlen),
				String.valueOf(clen), String.valueOf(rows), String.valueOf(cols), output(OUTPUT_NAME + "_target")};
			runTest(true, false, null, -1);

			// compare results
			MatrixBlock actual = DataConverter.readMatrixFromHDFS(output(OUTPUT_NAME),
				Types.FileFormat.BINARY, rows, cols, blen);
			MatrixBlock expected = DataConverter.readMatrixFromHDFS(output(OUTPUT_NAME + "_target"),
				Types.FileFormat.BINARY, rows, cols, blen);

			TestUtils.compareMatrices(expected, actual, eps);
		}
		catch(Exception ex) {
			Assert.fail(ex.getMessage());
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
