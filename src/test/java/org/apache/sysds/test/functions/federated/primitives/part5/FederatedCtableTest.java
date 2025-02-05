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

package org.apache.sysds.test.functions.federated.primitives.part5;

import java.util.Arrays;
import java.util.Collection;

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class FederatedCtableTest extends AutomatedTestBase {
	private final static String TEST_DIR = "functions/federated/";
	private final static String TEST_NAME1 = "FederatedCtableTest";
	private final static String TEST_NAME2 = "FederatedCtableFedOutput";
	private final static String TEST_NAME3 = "FederatedCtableSeqVecFedOut";
	private final static String TEST_CLASS_DIR = TEST_DIR + FederatedCtableTest.class.getSimpleName() + "/";

	private final static double TOLERANCE = 1e-12;

	private final static int blocksize = 1024;
	@Parameterized.Parameter()
	public int rows;
	@Parameterized.Parameter(1)
	public int cols;
	@Parameterized.Parameter(2)
	public int maxVal1;
	@Parameterized.Parameter(3)
	public int maxVal2;
	@Parameterized.Parameter(4)
	public boolean reversedInputs;
	@Parameterized.Parameter(5)
	public boolean weighted;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"F"}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] {"F"}));
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] {"F"}));
	}

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] {
			// {12, 4, 4, 7, true, true}, 
			// {12, 4, 4, 7, true, false},
			// {12, 4, 4, 7, false, true}, 
			// {12, 4, 4, 7, false, false},

			{100, 14, 4, 7, true, true}, //
			{100, 14, 4, 7, true, false}, //
			{100, 14, 4, 7, false, true}, //
			{100, 14, 4, 7, false, false}, //

			// {1000, 14, 4, 7, true, true}, {1000, 14, 4, 7, true, false},
			// {1000, 14, 4, 7, false, true}, {1000, 14, 4, 7, false, false}
		});
	}

	@Test
	public void federatedCtableSinglenode() {
		runCtable(Types.ExecMode.SINGLE_NODE, false, false);
	}

	@Test
	public void federatedCtableFedOutputSinglenode() {
		runCtable(Types.ExecMode.SINGLE_NODE, true, false);
	}

	@Test
	public void federatedCtableMatrixInputSinglenode() {
		runCtable(Types.ExecMode.SINGLE_NODE, false, true);
	}

	@Test
	public void federatedCtableMatrixInputFedOutputSingleNode() {
		runCtable(Types.ExecMode.SINGLE_NODE, true, true);
	}

	@Test
	@Ignore
	public void federatedCtableSeqVecFedOutputSingleNode() {
		runCtable(Types.ExecMode.SINGLE_NODE, true, false, true);
	}

	@Test
	public void federatedCtableSeqVecSliceFedOutputSingleNode() {
		runCtable(Types.ExecMode.SINGLE_NODE, true, true, true);
	}

	public void runCtable(Types.ExecMode execMode, boolean fedOutput, boolean matrixInput) {
		runCtable(execMode, fedOutput, matrixInput, false);
	}

	public void runCtable(Types.ExecMode execMode, boolean fedOutput, boolean matrixInput, boolean seqVec) {
		String TEST_NAME = fedOutput ? (seqVec ? TEST_NAME3 : TEST_NAME2) : TEST_NAME1;
		Types.ExecMode platformOld = setExecMode(execMode);

		getAndLoadTestConfiguration(TEST_NAME);
		String HOME = SCRIPT_DIR + TEST_DIR;

		// empty script name because we don't execute any script, just start the worker
		fullDMLScriptName = "";
		int port1 = getRandomAvailablePort();
		int port2 = getRandomAvailablePort();
		int port3 = getRandomAvailablePort();
		int port4 = getRandomAvailablePort();
		Process t1 = startLocalFedWorker(port1, FED_WORKER_WAIT_S);
		Process t2 = startLocalFedWorker(port2, FED_WORKER_WAIT_S);
		Process t3 = startLocalFedWorker(port3, FED_WORKER_WAIT_S);
		Process t4 = startLocalFedWorker(port4);

		
		try {
			if(!isAlive(t1, t2, t3, t4))
				throw new RuntimeException("Failed starting federated worker");

			TestConfiguration config = availableTestConfigurations.get(TEST_NAME);
			loadTestConfiguration(config);

			if(fedOutput)
				runFedCtable(HOME, TEST_NAME, matrixInput, port1, port2, port3, port4);
			else
				runNonFedCtable(HOME, TEST_NAME, matrixInput, port1, port2, port3, port4);
			checkResults(fedOutput);

		}
		finally {
			TestUtils.shutdownThreads(t1, t2, t3, t4);
			resetExecMode(platformOld);
		}
	}

	private void runNonFedCtable(String HOME, String TEST_NAME, boolean matrixInput, int port1, int port2, int port3,
		int port4) {
		int r = rows / 4;
		cols = matrixInput ? cols : 1;
		double[][] X1 = TestUtils.floor(getRandomMatrix(r, cols, 1, maxVal1, 1, 3));
		double[][] X2 = TestUtils.floor(getRandomMatrix(r, cols, 1, maxVal1, 1, 7));
		double[][] X3 = TestUtils.floor(getRandomMatrix(r, cols, 1, maxVal1, 1, 8));
		double[][] X4 = TestUtils.floor(getRandomMatrix(r, cols, 1, maxVal1, 1, 9));

		MatrixCharacteristics mc = new MatrixCharacteristics(r, cols, blocksize, r);
		writeInputMatrixWithMTD("X1", X1, false, mc);
		writeInputMatrixWithMTD("X2", X2, false, mc);
		writeInputMatrixWithMTD("X3", X3, false, mc);
		writeInputMatrixWithMTD("X4", X4, false, mc);

		double[][] Y = TestUtils.floor(getRandomMatrix(rows, cols, 1, maxVal2, 1, 9));
		writeInputMatrixWithMTD("Y", Y, false, new MatrixCharacteristics(rows, cols, blocksize, r));

		// empty script name because we don't execute any script, just start the worker
		fullDMLScriptName = "";

		// Run reference dml script with normal matrix
		fullDMLScriptName = HOME + TEST_NAME + "Reference.dml";
		programArgs = new String[] {"-stats", "100", "-args", input("X1"), input("X2"), input("X3"), input("X4"),
			input("Y"), Boolean.toString(reversedInputs).toUpperCase(), Boolean.toString(weighted).toUpperCase(),
			expected("F")};
		runTest(true, false, null, -1);

		// Run actual dml script with federated matrix
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[] {"-stats", "100", "-nvargs", "in_X1=" + TestUtils.federatedAddress(port1, input("X1")),
			"in_X2=" + TestUtils.federatedAddress(port2, input("X2")),
			"in_X3=" + TestUtils.federatedAddress(port3, input("X3")),
			"in_X4=" + TestUtils.federatedAddress(port4, input("X4")), "in_Y=" + input("Y"), "rows=" + rows,
			"cols=" + cols, "revIn=" + Boolean.toString(reversedInputs).toUpperCase(),
			"weighted=" + Boolean.toString(weighted).toUpperCase(), "out=" + output("F")};
		runTest(true, false, null, -1);
	}

	private void runFedCtable(String HOME, String TEST_NAME, boolean matrixInput, int port1, int port2, int port3,
		int port4) {
		int r = rows / 4;
		int c = cols;

		double[][] X1 = getRandomMatrix(r, c, 1, 3, 1, 3);
		double[][] X2 = getRandomMatrix(r, c, 1, 3, 1, 7);
		double[][] X3 = getRandomMatrix(r, c, 1, 3, 1, 8);
		double[][] X4 = getRandomMatrix(r, c, 1, 3, 1, 9);

		MatrixCharacteristics mc = new MatrixCharacteristics(r, c, blocksize, r * c);
		writeInputMatrixWithMTD("X1", X1, false, mc);
		writeInputMatrixWithMTD("X2", X2, false, mc);
		writeInputMatrixWithMTD("X3", X3, false, mc);
		writeInputMatrixWithMTD("X4", X4, false, mc);

		// execute main test
		fullDMLScriptName = HOME + TEST_NAME + "Reference.dml";
		programArgs = new String[] {"-stats", "100", "-args", input("X1"), input("X2"), input("X3"), input("X4"),
			Boolean.toString(reversedInputs).toUpperCase(), Boolean.toString(weighted).toUpperCase(),
			Boolean.toString(matrixInput).toUpperCase(), expected("F")};
		runTest(true, false, null, -1);

		// Run actual dml script with federated matrix
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[] {"-stats", "100", "-nvargs", "in_X1=" + TestUtils.federatedAddress(port1, input("X1")),
			"in_X2=" + TestUtils.federatedAddress(port2, input("X2")),
			"in_X3=" + TestUtils.federatedAddress(port3, input("X3")),
			"in_X4=" + TestUtils.federatedAddress(port4, input("X4")), "rows=" + rows, "cols=" + cols,
			"revIn=" + Boolean.toString(reversedInputs).toUpperCase(),
			"matrixInput=" + Boolean.toString(matrixInput).toUpperCase(),
			"weighted=" + Boolean.toString(weighted).toUpperCase(), "out=" + output("F")};
		runTest(true, false, null, -1);
	}

	void checkResults(boolean fedOutput) {
		// compare via files
		compareResults(TOLERANCE);

		// check for federated operations
		// TODO: add support for ctableexpand back when rewrite change first parameter to string seq
		if(heavyHittersContainsString("ctableexpand"))
			return; 

		Assert.assertTrue(heavyHittersContainsString("fed_ctable") || heavyHittersContainsString("ctableexpand"));
		if(fedOutput) { // verify output is federated
			Assert.assertTrue(heavyHittersContainsString("fed_uak+"));
			Assert.assertTrue(heavyHittersContainsString("fed_*"));
		}

		// check that federated input files are still existing
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X1")));
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X2")));
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X3")));
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X4")));
	}

}
