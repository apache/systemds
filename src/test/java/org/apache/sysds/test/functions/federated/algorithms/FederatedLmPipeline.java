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

package org.apache.sysds.test.functions.federated.algorithms;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.instructions.InstructionUtils;
import org.apache.sysds.runtime.matrix.data.LibMatrixMult;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.transform.encode.ColumnEncoderRecode;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;
import org.junit.Assert;
import org.junit.Test;

@net.jcip.annotations.NotThreadSafe
public class FederatedLmPipeline extends AutomatedTestBase {

	protected static final Log LOG = LogFactory.getLog(FederatedLmPipeline.class.getName());

	private final static String TEST_DIR = "functions/federated/";
	private final static String TEST_NAME1 = "FederatedLmPipeline";
	private final static String TEST_NAME2 = "FederatedLmPipeline4Workers";
	private final static String TEST_CLASS_DIR = TEST_DIR + FederatedLmPipeline.class.getSimpleName() + "/";

	public int rows = 10000;
	public int cols = 1000;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"Z"}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] {"Z"}));
	}

	@Test
	public void federatedLmPipelineContinguous() {
		federatedLmPipeline(Types.ExecMode.SINGLE_NODE, true, TEST_NAME1);
	}

	@Test
	public void federatedLmPipelineContinguous4Workers() {
		federatedLmPipeline(Types.ExecMode.SINGLE_NODE, true, TEST_NAME2);
	}

	@Test
	public void federatedLmPipelineSampled() {
		federatedLmPipeline(Types.ExecMode.SINGLE_NODE, false, TEST_NAME1);
	}

	@Test
	public void federatedLmPipelineSampled4Workers() {
		federatedLmPipeline(Types.ExecMode.SINGLE_NODE, false, TEST_NAME2);
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
			int quarterRows = TEST_NAME.equals(TEST_NAME2) ? rows / 4 : rows / 2;
			int[] k = TEST_NAME.equals(TEST_NAME2) ? new int[] {quarterRows - 1, quarterRows, 2 * quarterRows - 1,
				2 * quarterRows, 3 * quarterRows - 1, 3 * quarterRows,
				rows - 1} : new int[] {quarterRows - 1, quarterRows, rows - 1, 0, 0, 0, 0};
			writeInputMatrixWithMTD("X1", X.slice(0, k[0]), false);
			writeInputMatrixWithMTD("X2", X.slice(k[1], k[2]), false);
			writeInputMatrixWithMTD("X3", X.slice(k[3], k[4]), false);
			writeInputMatrixWithMTD("X4", X.slice(k[5], k[6]), false);
			writeInputMatrixWithMTD("Y", y, false);

			// empty script name because we don't execute any script, just start the worker
			fullDMLScriptName = "";
			int port1 = getRandomAvailablePort();
			int port2 = getRandomAvailablePort();
			int port3 = getRandomAvailablePort();
			int port4 = getRandomAvailablePort();
			Thread t1 = startLocalFedWorkerThread(port1, FED_WORKER_WAIT_S);
			Thread t2 = startLocalFedWorkerThread(port2, FED_WORKER_WAIT_S);
			Thread t3 = startLocalFedWorkerThread(port3, FED_WORKER_WAIT_S);
			Thread t4 = startLocalFedWorkerThread(port4);

			TestConfiguration config = availableTestConfigurations.get(TEST_NAME);
			loadTestConfiguration(config);
			
			// Run reference dml script with normal matrix
			fullDMLScriptName = HOME + TEST_NAME + "Reference.dml";
			programArgs = new String[] {"-args", input("X1"), input("X2"), input("X3"), input("X4"), input("Y"),
				String.valueOf(contSplits).toUpperCase(), expected("Z")};
			runTest(true, false, null, -1);

			// Run actual dml script with federated matrix
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-stats", "100", "-nvargs", "in_X1=" + TestUtils.federatedAddress(port1, input("X1")),
				"in_X2=" + TestUtils.federatedAddress(port2, input("X2")),
				"in_X3=" + TestUtils.federatedAddress(port3, input("X3")),
				"in_X4=" + TestUtils.federatedAddress(port4, input("X4")), "rows=" + rows, "cols=" + (cols + 1),
				"in_Y=" + input("Y"), "cont=" + String.valueOf(contSplits).toUpperCase(), "out=" + output("Z")};
			runTest(true, false, null, -1);

			// compare via files
			compareResults(1e-2);
			TestUtils.shutdownThreads(t1, t2, t3, t4);
			
			
			// check correct federated operations
			Assert.assertTrue(Statistics.getCPHeavyHitterCount("fed_mmchain")>10);
			Assert.assertTrue(Statistics.getCPHeavyHitterCount("fed_ba+*") == 2);
		}
		finally {
			resetExecMode(oldExec);
			ColumnEncoderRecode.SORT_RECODE_MAP = oldSort;
		}
	}
}
