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

package org.apache.sysds.test.functions.federated.primitives.part2;

import java.util.Arrays;
import java.util.Collection;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.controlprogram.ProgramBlock;
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
public class FederatedRdiagTest extends AutomatedTestBase {

	private final static String TEST_DIR = "functions/federated/";
	private final static String TEST_NAME = "FederatedRdiagTest";
	private final static String TEST_CLASS_DIR = TEST_DIR + FederatedRdiagTest.class.getSimpleName() + "/";

	private final static int blocksize = 1024;
	@Parameterized.Parameter()
	public int rows;

	@Parameterized.Parameter(1)
	public int cols;

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] {{1000, 1000}, {1000, 1}});
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"S"}));
	}

	@Test
	public void federatedRdiagCP() {
		federatedRdiag(Types.ExecMode.SINGLE_NODE);
	}

	@Test
	public void federatedRdiagSP() {
		federatedRdiag(Types.ExecMode.SPARK);
	}

	@Test
	public void federatedCompilationRDiagCP() {
		federatedRdiag(Types.ExecMode.SINGLE_NODE, true);
	}

	@Test
	public void federatedCompilationRdiagSP() {
		federatedRdiag(Types.ExecMode.SPARK, true);
	}

	public void federatedRdiag(Types.ExecMode execMode) {
		federatedRdiag(execMode, false);
	}

	public void federatedRdiag(Types.ExecMode execMode, boolean activateFedCompilation) {
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		Types.ExecMode platformOld = rtplatform;

		getAndLoadTestConfiguration(TEST_NAME);
		String HOME = SCRIPT_DIR + TEST_DIR;

		int r = rows / 4;

		double[][] X1 = getRandomMatrix(r, cols, 1, 5, 1, 3);
		double[][] X2 = getRandomMatrix(r, cols, 1, 5, 1, 7);
		double[][] X3 = getRandomMatrix(r, cols, 1, 5, 1, 8);
		double[][] X4 = getRandomMatrix(r, cols, 1, 5, 1, 9);

		MatrixCharacteristics mc = new MatrixCharacteristics(r, cols, blocksize, r * cols);
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
		Process t1 = startLocalFedWorker(port1, FED_WORKER_WAIT_S);
		Process t2 = startLocalFedWorker(port2, FED_WORKER_WAIT_S);
		Process t3 = startLocalFedWorker(port3, FED_WORKER_WAIT_S);
		Process t4 = startLocalFedWorker(port4);

		ProgramBlock.CHECK_MATRIX_PROPERTIES = true;
		try {
			if(!isAlive(t1, t2, t3, t4))
				throw new RuntimeException("Failed starting federated worker");
			
			// reference file should not be written to hdfs, so we set platform here
			rtplatform = execMode;
			if(rtplatform == Types.ExecMode.SPARK) {
				DMLScript.USE_LOCAL_SPARK_CONFIG = true;
			}
			// Run reference dml script with normal matrix for Row/Col
			fullDMLScriptName = HOME + TEST_NAME + "Reference.dml";
			programArgs = new String[] {"-stats", "100", "-args", input("X1"), input("X2"), input("X3"), input("X4"),
				expected("S")};
			runTest(null);

			OptimizerUtils.FEDERATED_COMPILATION = activateFedCompilation;
			TestConfiguration config = availableTestConfigurations.get(TEST_NAME);
			loadTestConfiguration(config);

			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-stats", "100", "-nvargs",
				"in_X1=" + TestUtils.federatedAddress(port1, input("X1")),
				"in_X2=" + TestUtils.federatedAddress(port2, input("X2")),
				"in_X3=" + TestUtils.federatedAddress(port3, input("X3")),
				"in_X4=" + TestUtils.federatedAddress(port4, input("X4")), "rows=" + rows, "cols=" + cols,
				"out_S=" + output("S")};
			runTest(null);

			// compare all sums via files
			compareResults(0.01, "DML1", "DML2");

			Assert.assertTrue(heavyHittersContainsString("fed_rdiag"));

			// check that federated input files are still existing
			Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X1")));
			Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X2")));
			Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X3")));
			Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X4")));

		}
		finally {
			TestUtils.shutdownThreads(t1, t2, t3, t4);
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			OptimizerUtils.FEDERATED_COMPILATION = false;
			ProgramBlock.CHECK_MATRIX_PROPERTIES = false;
		}
	}
}
