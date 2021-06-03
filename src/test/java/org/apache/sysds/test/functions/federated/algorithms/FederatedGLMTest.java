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

import java.util.Arrays;
import java.util.Collection;

import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class FederatedGLMTest extends AutomatedTestBase {

	private final static String TEST_DIR = "functions/federated/";
	private final static String TEST_NAME = "FederatedGLMTest";
	private final static String TEST_CLASS_DIR = TEST_DIR + FederatedGLMTest.class.getSimpleName() + "/";

	private final static int blocksize = 1024;
	@Parameterized.Parameter()
	public int rows;
	@Parameterized.Parameter(1)
	public int cols;
	@Parameterized.Parameter(2)
	public boolean rowPartitioned;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"Z"}));
	}

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		// rows have to be even and > 1
		return Arrays.asList(new Object[][] {
			// {10000, 10, true}, {1000, 100, false},
			{2000, 44, true},
			{2000, 44, false}});
	}

	@Test
	public void federatedSinglenodeGLM() {
		federatedGLM(Types.ExecMode.SINGLE_NODE);
	}

	@Test
	public void federatedHybridGLM() {
		federatedGLM(Types.ExecMode.HYBRID);
	}

	public void federatedGLM(Types.ExecMode execMode) {
		ExecMode platformOld = setExecMode(execMode);

		getAndLoadTestConfiguration(TEST_NAME);
		String HOME = SCRIPT_DIR + TEST_DIR;

		// write input matrices
		int r = rowPartitioned ? rows / 2 : rows;
		int c = rowPartitioned ? cols : cols / 2;

		// We have two matrices handled by a single federated worker
		double[][] X1 = getRandomMatrix(r, c, 0, 1, 1, 42);
		double[][] X2 = getRandomMatrix(r, c, 0, 1, 1, 1340);
		double[][] Y = getRandomMatrix(rows, 1, -1, 1, 1, 1233);
		for(int i = 0; i < rows; i++)
			Y[i][0] = (Y[i][0] > 0) ? 1 : -1;

		writeInputMatrixWithMTD("X1", X1, false, new MatrixCharacteristics(r, c, blocksize, r * c));
		writeInputMatrixWithMTD("X2", X2, false, new MatrixCharacteristics(r, c, blocksize, r * c));
		writeInputMatrixWithMTD("Y", Y, false, new MatrixCharacteristics(rows, 1, blocksize, rows));

		// empty script name because we don't execute any script, just start the worker
		fullDMLScriptName = "";
		int port1 = getRandomAvailablePort();
		int port2 = getRandomAvailablePort();
		Thread t1 = startLocalFedWorkerThread(port1, FED_WORKER_WAIT_S);
		Thread t2 = startLocalFedWorkerThread(port2);

		TestConfiguration config = availableTestConfigurations.get(TEST_NAME);
		loadTestConfiguration(config);
		//

		// Run reference dml script with normal matrix
		fullDMLScriptName = HOME + TEST_NAME + "Reference.dml";
		programArgs = new String[] {"-args", input("X1"), input("X2"), input("Y"), Boolean.toString(rowPartitioned).toUpperCase(), expected("Z")};
		runTest(true, false, null, -1);

		// Run actual dml script with federated matrix
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[] {"-stats", "-nvargs", "in_X1=" + TestUtils.federatedAddress(port1, input("X1")),
			"in_X2=" + TestUtils.federatedAddress(port2, input("X2")), "rows=" + rows, "cols=" + cols,
			"in_Y=" + input("Y"), "rP=" + Boolean.toString(rowPartitioned).toUpperCase(), "out=" + output("Z")};
		runTest(true, false, null, -1);

		// compare via files
		compareResults(1e-2);

		TestUtils.shutdownThreads(t1, t2);

		// check for federated operations
		Assert.assertTrue(heavyHittersContainsString("fed_ba+*"));
		Assert.assertTrue(heavyHittersContainsString("fed_uark+", "fed_uarsqk+"));
		Assert.assertTrue(heavyHittersContainsString("fed_uack+"));
		// Assert.assertTrue(heavyHittersContainsString("fed_uak+"));
		Assert.assertTrue(!rowPartitioned || heavyHittersContainsString("fed_mmchain"));

		// check that federated input files are still existing
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X1")));
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X2")));

		resetExecMode(platformOld);
	}
}
