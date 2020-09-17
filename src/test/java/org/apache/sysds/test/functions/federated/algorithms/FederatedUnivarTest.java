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

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.Collection;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class FederatedUnivarTest extends AutomatedTestBase {
	private final static String TEST_DIR = "functions/federated/";
	private final static String TEST_NAME = "FederatedUnivarTest";
	private final static String TEST_CLASS_DIR = TEST_DIR + FederatedUnivarTest.class.getSimpleName() + "/";

	private final static int blocksize = 1024;
	@Parameterized.Parameter()
	public int rows;
	@Parameterized.Parameter(1)
	public int cols;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"B"}));
	}

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] {
				{10000, 16},
				{2000, 32}, {1000, 64}, {10000, 128}
		});
	}

	@Test
	public void federatedUnivarSinglenode() {
		federatedL2SVM(Types.ExecMode.SINGLE_NODE);
	}

	@Test
	public void federatedUnivarHybrid() {
		federatedL2SVM(Types.ExecMode.HYBRID);
	}

	public void federatedL2SVM(Types.ExecMode execMode) {
		Types.ExecMode platformOld = setExecMode(execMode);

		getAndLoadTestConfiguration(TEST_NAME);
		String HOME = SCRIPT_DIR + TEST_DIR;

		// write input matrices
		int quarterCols = cols / 4;
		// We have two matrices handled by a single federated worker
		double[][] X1 = getRandomMatrix(rows, quarterCols, 1, 5, 1, 3);
		double[][] X2 = getRandomMatrix(rows, quarterCols, 1, 5, 1, 7);
		double[][] X3 = getRandomMatrix(rows, quarterCols, 1, 5, 1, 8);
		double[][] X4 = getRandomMatrix(rows, quarterCols, 1, 5, 1, 9);

		// write types matrix shape of (1, D)
		double [][] Y = getRandomMatrix(1, cols, 0, 3, 1, 9);
		Arrays.stream(Y[0]).forEach(Math::ceil);

		MatrixCharacteristics mc= new MatrixCharacteristics(rows, quarterCols, blocksize, rows * quarterCols);
		writeInputMatrixWithMTD("X1", X1, false, mc);
		writeInputMatrixWithMTD("X2", X2, false, mc);
		writeInputMatrixWithMTD("X3", X3, false, mc);
		writeInputMatrixWithMTD("X4", X4, false, mc);
		writeInputMatrixWithMTD("Y", Y, false);

		// empty script name because we don't execute any script, just start the worker
		fullDMLScriptName = "";
		int port1 = getRandomAvailablePort();
		int port2 = getRandomAvailablePort();
		int port3 = getRandomAvailablePort();
		int port4 = getRandomAvailablePort();
		Thread t1 = startLocalFedWorkerThread(port1);
		Thread t2 = startLocalFedWorkerThread(port2);
		Thread t3 = startLocalFedWorkerThread(port3);
		Thread t4 = startLocalFedWorkerThread(port4);

		TestConfiguration config = availableTestConfigurations.get(TEST_NAME);
		loadTestConfiguration(config);
		

		// Run reference dml script with normal matrix
		fullDMLScriptName = HOME + TEST_NAME + "Reference.dml";
		programArgs = new String[] {"-stats", "100", "-args", input("X1"), input("X2"), input("X3"), input("X4"), input("Y"), expected("B")};
		runTest(true, false, null, -1);

		// Run actual dml script with federated matrix
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[] {"-stats",  "100", "-nvargs",
			"in_X1=" + TestUtils.federatedAddress(port1, input("X1")),
			"in_X2=" + TestUtils.federatedAddress(port2, input("X2")),
			"in_X3=" + TestUtils.federatedAddress(port3, input("X3")),
			"in_X4=" + TestUtils.federatedAddress(port4, input("X4")),
			"in_Y=" + input("Y"), // types
			"rows=" + rows, "cols=" + cols,
			"out=" + output("B")};
		runTest(true, false, null, -1);

		// compare via files
		compareResults(1e-9);
		TestUtils.shutdownThreads(t1, t2, t3, t4);

		// check for federated operations
		Assert.assertTrue(heavyHittersContainsString("fed_uacmax"));

		//check that federated input files are still existing
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X1")));
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X2")));
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X3")));
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X4")));
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("Y")));

		resetExecMode(platformOld);
	}
}
