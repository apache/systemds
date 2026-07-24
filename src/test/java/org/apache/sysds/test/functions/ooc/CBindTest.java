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
import org.apache.sysds.parser.LanguageException;
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
public class CBindTest extends AutomatedTestBase {

	private static final String TEST_NAME = "CBindTest";
	private static final String TEST_DIR = "functions/ooc/";
	private static final String TEST_CLASS_DIR = TEST_DIR + CBindTest.class.getSimpleName() + "/";

	private final static double eps = 1e-8;
	private static final String INPUT_NAME_1 = "A";
	private static final String INPUT_NAME_2 = "B";
	private static final String OUTPUT_NAME = "res";

	private final int r1;
	private final int c1;
	private final int r2;
	private final int c2;
	private final int bsize;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME));
	}

	public CBindTest(int r1, int c1, int r2, int c2, int bsize) {
		this.r1 = r1;
		this.c1 = c1;
		this.r2 = r2;
		this.c2 = c2;
		this.bsize = bsize;
	}

	@Parameterized.Parameters(name = "{0}x{1} {2}x{3} bsize {4}")
	public static Iterable<Object[]> getParams() {
		int[] rows = new int[]{1000, 2000};
		int[] cols = new int[]{300, 700, 2300, 2700, 3000, 3300};
		int[] bsizes = new int[]{1000};

		ArrayList<Object[]> params = new ArrayList<>();

		for(int row : rows) {
			for(int col : cols) {
				for(int col2 : cols) {
					for(int bsize : bsizes) {
						params.add(new Object[] {row, col, row, col2, bsize});
					}
				}
			}
		}

		params.add(new Object[] {10, 1000, 20, 1000, 1000});
		params.add(new Object[] {0, 1000, 0, 1000, 1000});
		params.add(new Object[] {1000, 0, 1000, 1000, 1000});
		params.add(new Object[] {1000, 1000, 1000, 0, 1000});
		params.add(new Object[] {1000, 0, 1000, 0, 1000});

		return params;
	}

	@Test
	public void runCBindTest() {
		Types.ExecMode platformOld = rtplatform;
		rtplatform = Types.ExecMode.SINGLE_NODE;

		try {
			getAndLoadTestConfiguration(TEST_NAME);
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";

			double[][] A = TestUtils.floor(getRandomMatrix(r1, c1, -1, 1, 1.0, 7));
			double[][] B = TestUtils.floor(getRandomMatrix(r2, c2, -1, 1, 1.0, 13));

			MatrixWriter writer = MatrixWriterFactory.createMatrixWriter(Types.FileFormat.BINARY);
			writer.writeMatrixToHDFS(DataConverter.convertToMatrixBlock(A), input(INPUT_NAME_1), r1, c1, bsize, r1*c1);
			writer.writeMatrixToHDFS(DataConverter.convertToMatrixBlock(B), input(INPUT_NAME_2), r2, c2, bsize, r2*c2);

			HDFSTool.writeMetaDataFile(input(INPUT_NAME_1 + ".mtd"), Types.ValueType.FP64,
				new MatrixCharacteristics(r1, c1, bsize, r1*c1), Types.FileFormat.BINARY);
			HDFSTool.writeMetaDataFile(input(INPUT_NAME_2 + ".mtd"), Types.ValueType.FP64,
				new MatrixCharacteristics(r2, c2, bsize, r2*c2), Types.FileFormat.BINARY);


			programArgs = new String[] {"-explain", "-stats", "-ooc", "-args",
				input(INPUT_NAME_1), input(INPUT_NAME_2), output(OUTPUT_NAME)};

			if(r1 != r2){
				runTest(true,true, LanguageException.class,-1);
				return;
			}

			runTest(true, false, null, -1);
			Assert.assertTrue("OOC wasn't used for cbind",
				heavyHittersContainsString(Instruction.OOC_INST_PREFIX + Opcodes.APPEND));

			// rerun without ooc flag
			programArgs = new String[] {"-explain", "-stats", "-args",
				input(INPUT_NAME_1), input(INPUT_NAME_2), output(OUTPUT_NAME + "_target")};
			runTest(true, false, null, -1);

			// compare results
			MatrixBlock ret1 = DataConverter.readMatrixFromHDFS(output(OUTPUT_NAME),
				Types.FileFormat.BINARY, r1, c1+c2, bsize);
			MatrixBlock ret2 = DataConverter.readMatrixFromHDFS(output(OUTPUT_NAME + "_target"),
				Types.FileFormat.BINARY, r1, c1+c2, bsize);
			TestUtils.compareMatrices(ret1, ret2, eps);
		}
		catch(Exception ex) {
			Assert.fail(ex.getMessage());
		}
		finally {
			resetExecMode(platformOld);
		}
	}
}
