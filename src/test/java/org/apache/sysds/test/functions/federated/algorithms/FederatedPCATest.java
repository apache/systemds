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

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

import java.util.Arrays;
import java.util.Collection;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class FederatedPCATest extends AutomatedTestBase {

	private final static String TEST_DIR = "functions/federated/";
	private final static String TEST_NAME = "FederatedPCATest";
	private final static String TEST_CLASS_DIR = TEST_DIR + FederatedPCATest.class.getSimpleName() + "/";

	private final static int blocksize = 1024;
	@Parameterized.Parameter()
	public int rows;
	@Parameterized.Parameter(1)
	public int cols;
	@Parameterized.Parameter(2)
	public boolean scaleAndShift;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"Z"}));
	}

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		// rows have to be even and > 1
		return Arrays.asList(new Object[][] {
			{10000, 10, false}, {2000, 50, false}, {1000, 100, false},
			{10000, 10, true}, {2000, 50, true}, {1000, 100, true}
		});
	}

	@Test
	public void federatedPCASinglenode() {
		federatedL2SVM(Types.ExecMode.SINGLE_NODE);
	}
	
	@Test
	public void federatedPCAHybrid() {
		federatedL2SVM(Types.ExecMode.HYBRID);
	}

	public void federatedL2SVM(Types.ExecMode execMode) {
		ExecMode platformOld = setExecMode(execMode);

		getAndLoadTestConfiguration(TEST_NAME);
		String HOME = SCRIPT_DIR + TEST_DIR;

		// write input matrices
		int quarterRows = rows / 4;
		// We have two matrices handled by a single federated worker
		double[][] X1 = getRandomMatrix(quarterRows, cols, 0, 1, 1, 3);
		double[][] X2 = getRandomMatrix(quarterRows, cols, 0, 1, 1, 7);
		double[][] X3 = getRandomMatrix(quarterRows, cols, 0, 1, 1, 8);
		double[][] X4 = getRandomMatrix(quarterRows, cols, 0, 1, 1, 9);
		MatrixCharacteristics mc= new MatrixCharacteristics(quarterRows, cols, blocksize, quarterRows * cols);
		writeInputMatrixWithMTD("X1", X1, false, mc);
		writeInputMatrixWithMTD("X2", X2, false, mc);
		writeInputMatrixWithMTD("X3", X3, false, mc);
		writeInputMatrixWithMTD("X4", X4, false, mc);

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
		programArgs = new String[] {"-stats", "-args", input("X1"), input("X2"), input("X3"), input("X4"),
			String.valueOf(scaleAndShift).toUpperCase(), expected("Z")};
		runTest(true, false, null, -1);

		// Run actual dml script with federated matrix
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[] {"-stats", "-nvargs", 
			"in_X1=" + TestUtils.federatedAddress(port1, input("X1")),
			"in_X2=" + TestUtils.federatedAddress(port2, input("X2")),
			"in_X3=" + TestUtils.federatedAddress(port3, input("X3")),
			"in_X4=" + TestUtils.federatedAddress(port4, input("X4")),
			"rows=" + rows, "cols=" + cols,
			"scaleAndShift=" + String.valueOf(scaleAndShift).toUpperCase(), "out=" + output("Z")};
		runTest(true, false, null, -1);

		// compare via files
		compareResults(1e-9);
		TestUtils.shutdownThreads(t1, t2, t3, t4);
		
		// check for federated operations
		Assert.assertTrue(heavyHittersContainsString("fed_ba+*"));
		Assert.assertTrue(heavyHittersContainsString("fed_uack+"));
		Assert.assertTrue(heavyHittersContainsString("fed_tsmm"));
		if( scaleAndShift ) {
			Assert.assertTrue(heavyHittersContainsString("fed_uacsqk+"));
			Assert.assertTrue(heavyHittersContainsString("fed_uacmean"));
			Assert.assertTrue(heavyHittersContainsString("fed_-"));
			Assert.assertTrue(heavyHittersContainsString("fed_/"));
			Assert.assertTrue(heavyHittersContainsString("fed_replace"));
		}
		
		//check that federated input files are still existing
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X1")));
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X2")));
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X3")));
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X4")));
		
		resetExecMode(platformOld);
	}
}
