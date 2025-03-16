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

package org.apache.sysds.test.functions.lineage;

import static org.junit.Assert.assertTrue;

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.lineage.Lineage;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.transform.encode.ColumnEncoderRecode;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;
import org.junit.Test;

@net.jcip.annotations.NotThreadSafe
public class LineageFedReuseAlg extends AutomatedTestBase {

	private final static String TEST_DIR = "functions/lineage/";
	private final static String TEST_NAME1 = "FedLmPipelineReuse";
	private final static String TEST_CLASS_DIR = TEST_DIR + LineageFedReuseAlg.class.getSimpleName() + "/";

	public int rows = 10000;
	public int cols = 100;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"Z"}));
	}

	@Test
	public void federatedLmPipelineContinguous() {
		federatedLmPipeline(Types.ExecMode.SINGLE_NODE, true, TEST_NAME1);
	}

	@Test
	public void federatedLmPipelineSampled() {
		federatedLmPipeline(Types.ExecMode.SINGLE_NODE, false, TEST_NAME1);
	}

	public void federatedLmPipeline(ExecMode execMode, boolean contSplits, String TEST_NAME) {
		ExecMode oldExec = setExecMode(execMode);
		boolean oldSort = ColumnEncoderRecode.SORT_RECODE_MAP;
		ColumnEncoderRecode.SORT_RECODE_MAP = true;

		getAndLoadTestConfiguration(TEST_NAME);
		String HOME = SCRIPT_DIR + TEST_DIR;

		try {
			// generated lm data
			MatrixBlock X = MatrixBlock.randOperations(rows, cols, 1.0, 0, 1, "uniform", 7);
			MatrixBlock w = MatrixBlock.randOperations(cols, 1, 1.0, 0, 1, "uniform", 3);
			MatrixBlock y = new MatrixBlock(rows, 1, false).allocateBlock();
			LibMatrixMult.matrixMult(X, w, y);
			MatrixBlock c = MatrixBlock.randOperations(rows, 1, 1.0, 1, 50, "uniform", 23);
			MatrixBlock rc = c.unaryOperations(InstructionUtils.parseUnaryOperator("round"), new MatrixBlock());
			X = rc.append(X, new MatrixBlock(), true);

			// We have two matrices handled by a single federated worker
			int quarterRows = rows / 2;
			int[] k = new int[] {quarterRows - 1, quarterRows, rows - 1, 0, 0, 0, 0};
			writeInputMatrixWithMTD("X1", X.slice(0, k[0]), false);
			writeInputMatrixWithMTD("X2", X.slice(k[1], k[2]), false);
			writeInputMatrixWithMTD("X3", X.slice(k[3], k[4]), false);
			writeInputMatrixWithMTD("X4", X.slice(k[5], k[6]), false);
			writeInputMatrixWithMTD("Y", y, false);

			int port1 = getRandomAvailablePort();
			int port2 = getRandomAvailablePort();
			int port3 = getRandomAvailablePort();
			int port4 = getRandomAvailablePort();
			String[] otherargs = new String[] {"-lineage", "reuse_full"};
			Thread t1 = startLocalFedWorkerThread(port1, otherargs, FED_WORKER_WAIT_S);
			Thread t2 = startLocalFedWorkerThread(port2, otherargs);

			TestConfiguration config = availableTestConfigurations.get(TEST_NAME);
			loadTestConfiguration(config);

			// Run with federated matrix and with reuse
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-stats", "20", "-lineage", "reuse_full",
				"-nvargs", "in_X1=" + TestUtils.federatedAddress(port1, input("X1")),
				"in_X2=" + TestUtils.federatedAddress(port2, input("X2")),
				"in_X3=" + TestUtils.federatedAddress(port3, input("X3")),
				"in_X4=" + TestUtils.federatedAddress(port4, input("X4")), "rows=" + rows, "cols=" + (cols + 1),
				"in_Y=" + input("Y"), "cont=" + String.valueOf(contSplits).toUpperCase(), "out=" + output("Z")};
			Lineage.resetInternalState();
			runTest(true, false, null, -1);
			long tsmmCount_reuse = Statistics.getCPHeavyHitterCount(Opcodes.TSMM.toString());
			long fed_tsmmCount_reuse = Statistics.getCPHeavyHitterCount("fed_tsmm");
			long mmCount_reuse = Statistics.getCPHeavyHitterCount(Opcodes.MMULT.toString());
			long fed_mmCount_reuse = Statistics.getCPHeavyHitterCount("fed_ba+*");

			// Run with federated matrix and without reuse
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-stats", "20",
				"-nvargs", "in_X1=" + TestUtils.federatedAddress(port1, input("X1")),
				"in_X2=" + TestUtils.federatedAddress(port2, input("X2")),
				"in_X3=" + TestUtils.federatedAddress(port3, input("X3")),
				"in_X4=" + TestUtils.federatedAddress(port4, input("X4")), "rows=" + rows, "cols=" + (cols + 1),
				"in_Y=" + input("Y"), "cont=" + String.valueOf(contSplits).toUpperCase(), "out=" + expected("Z")};
			runTest(true, false, null, -1);
			long tsmmCount = Statistics.getCPHeavyHitterCount(Opcodes.TSMM.toString());
			long fed_tsmmCount = Statistics.getCPHeavyHitterCount("fed_tsmm");
			long mmCount = Statistics.getCPHeavyHitterCount(Opcodes.MMULT.toString());
			long fed_mmCount = Statistics.getCPHeavyHitterCount("fed_ba+*");

			// compare results 
			compareResults(1e-2);
			// compare potentially reused instruction counts
			assertTrue(tsmmCount > tsmmCount_reuse);
			assertTrue(fed_tsmmCount > fed_tsmmCount_reuse);
			assertTrue(mmCount > mmCount_reuse);
			assertTrue(fed_mmCount > fed_mmCount_reuse);

			TestUtils.shutdownThreads(t1, t2);
		}
		finally {
			resetExecMode(oldExec);
			ColumnEncoderRecode.SORT_RECODE_MAP = oldSort;
		}
	}
}