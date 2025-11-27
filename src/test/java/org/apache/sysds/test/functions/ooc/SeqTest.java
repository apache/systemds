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
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

public class SeqTest extends AutomatedTestBase {
	private final static String TEST_NAME1 = "Seq";
	private final static String TEST_DIR = "functions/ooc/";
	private final static String TEST_CLASS_DIR = TEST_DIR + SeqTest.class.getSimpleName() + "/";
	private final static double eps = 1e-8;
	private static final String OUTPUT_NAME = "res";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		TestConfiguration config = new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1);
		addTestConfiguration(TEST_NAME1, config);
	}

	@Test
	public void testSeq1() {
		runSeqTest(0, 10, 0.1);
	}

	@Test
	public void testSeq2() {
		runSeqTest(0, 15.9, 0.01);
	}

	private void runSeqTest(double from, double to, double incr) {
		Types.ExecMode platformOld = setExecMode(Types.ExecMode.SINGLE_NODE);

		try {
			getAndLoadTestConfiguration(TEST_NAME1);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME1 + ".dml";
			programArgs = new String[] {"-explain", "-stats", "-ooc", "-args", Double.toString(from), Double.toString(to), Double.toString(incr), output(OUTPUT_NAME)};

			runTest(true, false, null, -1);

			//check seq OOC
			Assert.assertTrue("OOC wasn't used for seq",
				heavyHittersContainsString(Instruction.OOC_INST_PREFIX + Opcodes.SEQUENCE));
			//compare results

			// rerun without ooc flag
			programArgs = new String[] {"-explain", "-stats", "-args", Double.toString(from), Double.toString(to), Double.toString(incr), output(OUTPUT_NAME + "_target")};
			runTest(true, false, null, -1);

			// compare matrices
			MatrixBlock ret1 = TestUtils.readBinary(output(OUTPUT_NAME));
			MatrixBlock ret2 = TestUtils.readBinary(output(OUTPUT_NAME + "_target"));

			TestUtils.compareMatrices(ret1, ret2, eps);
		}
		finally {
			resetExecMode(platformOld);
		}
	}
}
