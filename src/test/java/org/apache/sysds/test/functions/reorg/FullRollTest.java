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

package org.apache.sysds.test.functions.reorg;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.instructions.Instruction;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;
import org.junit.Assert;
import org.junit.Test;

import java.util.HashMap;


public class FullRollTest extends AutomatedTestBase {
	private final static String TEST_NAME1 = "Roll1";
	//private final static String TEST_NAME2 = "Roll2";

	private final static String TEST_DIR = "functions/reorg/";
	private static final String TEST_CLASS_DIR = TEST_DIR + FullRollTest.class.getSimpleName() + "/";

	private final static int rows1 = 2017;
	private final static int cols1 = 1001;
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.1;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[]{"B", "C"}));
	}

	@Test
	public void testRollVectorDense() {
		runRollTest(TEST_NAME1, false, false);
	}

	@Test
	public void testRollVectorSparse() {
		runRollTest(TEST_NAME1, false, true);
	}

	@Test
	public void testRollMatrixDense() {
		runRollTest(TEST_NAME1, true, false);
	}

	@Test
	public void testRollMatrixSparse() {
		runRollTest(TEST_NAME1, true, true);
	}

	private void runRollTest(String testname, boolean matrix, boolean sparse) {
		//rtplatform for MR
		ExecMode platformOld = rtplatform;
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;

		String TEST_NAME = testname;

		try {
			int cols = matrix ? cols1 : 1;
			double sparsity = sparse ? sparsity2 : sparsity1;
			getAndLoadTestConfiguration(TEST_NAME);

			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";

			//generate actual dataset
			double[][] A = getRandomMatrix(rows1, cols, -1, 1, sparsity, 7);
			writeInputMatrixWithMTD("A", A, true);

			// Run test CP
			rtplatform = ExecMode.HYBRID;
			DMLScript.USE_LOCAL_SPARK_CONFIG = false;
			programArgs = new String[]{"-stats", "-explain", "-args", input("A"), output("B")};
			runTest(true, false, null, -1);
			boolean opcodeCP = Statistics.getCPHeavyHitterOpCodes().contains(Opcodes.ROLL.toString());
			
			// Run test SP
			rtplatform = ExecMode.SPARK;
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
			programArgs = new String[]{"-stats", "-explain", "-args", input("A"), output("C")};
			runTest(true, false, null, -1);
			boolean opcodeSP = Statistics.getCPHeavyHitterOpCodes().contains(Instruction.SP_INST_PREFIX + Opcodes.ROLL.toString());
			
			//compare matrices
			HashMap<CellIndex, Double> dmlfileCP = readDMLMatrixFromOutputDir("B");
			HashMap<CellIndex, Double> dmlfileSP = readDMLMatrixFromOutputDir("C");
			TestUtils.compareMatrices(dmlfileCP, dmlfileSP, 0, "Stat-DML-CP", "Stat-DML-SP");

			Assert.assertTrue("Missing opcode: roll", opcodeCP);
			Assert.assertTrue("Missing opcode: sp_roll", opcodeSP);
		} finally {
			//reset flags
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}
