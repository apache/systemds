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
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.DataConverter;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

import java.io.IOException;

public class RandTest extends AutomatedTestBase {
	private final static String TEST_NAME_1 = "Rand1";
	private final static String TEST_NAME_2 = "Rand2";
	private final static String TEST_DIR = "functions/ooc/";
	private final static String TEST_CLASS_DIR = TEST_DIR + RandTest.class.getSimpleName() + "/";
	private final static double eps = 1e-8;
	private static final String INPUT_NAME_1 = "X";
	private static final String OUTPUT_NAME = "res";

	private final static int rows = 1500;
	private final static int cols = 1200;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		TestConfiguration config = new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_1);
		addTestConfiguration(TEST_NAME_1, config);
		TestConfiguration config2 = new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_2);
		addTestConfiguration(TEST_NAME_2, config2);
	}

	// Actual rand operation not yet supported
	/*@Test
	public void testRand() {
		runRandTest(TEST_NAME_1);
	}*/

	@Test
	public void testConstInit() {
		runRandTest(TEST_NAME_2);
	}

	private void runRandTest(String TEST_NAME) {
		Types.ExecMode platformOld = setExecMode(Types.ExecMode.SINGLE_NODE);

		try {
			getAndLoadTestConfiguration(TEST_NAME);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-explain", "-stats", "-ooc", "-args", input(INPUT_NAME_1), output(OUTPUT_NAME)};

			runTest(true, false, null, -1);

			//check replace OOC op
			Assert.assertTrue("OOC wasn't used for rand",
				heavyHittersContainsString(Instruction.OOC_INST_PREFIX + Opcodes.RANDOM));

			//compare results

			// rerun without ooc flag
			programArgs = new String[] {"-explain", "-stats", "-args", input(INPUT_NAME_1), output(OUTPUT_NAME + "_target")};
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
