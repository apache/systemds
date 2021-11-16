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

package org.apache.sysds.test.functions.federated.primitives;

import java.util.Arrays;
import java.util.Collection;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class FederatedRCBindTest extends AutomatedTestBase {

	private final static String TEST_DIR = "functions/federated/";
	private final static String TEST_NAME = "FederatedRCBindTest";
	private final static String TEST_CLASS_DIR = TEST_DIR + FederatedRCBindTest.class.getSimpleName() + "/";

	private final static int blocksize = 1024;
	@Parameterized.Parameter()
	public int rows;
	@Parameterized.Parameter(1)
	public int cols;
	@Parameterized.Parameter(2)
	public boolean partitioned;

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		//TODO add tests and support of aligned blocksized (which is however a special case)
		// rows must be even if paritioned
		return Arrays.asList(new Object[][] {
			// (rows, cols, paritioned)
			// {1, 1001, false},
			{10, 100, false},
			{100, 10, true},
			// {1001, 1, false},
			// {10, 2001, false},
			// {2000, 10, true},
			// {100, 100, true},
		});
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		// we generate 3 datasets, both with rbind and cbind (F...Federated, L...Local):
		// F-F, F-L, L-F
		addTestConfiguration(TEST_NAME,
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,
				new String[] {"R_FF_misaligned", "C_FF_aligned",
					"R_FF", "R_FL", "R_LF", "C_FF", "C_FL", "C_LF"}));
	}

	@Test
	public void federatedRCBindCP() {
		federatedRCBind(ExecMode.SINGLE_NODE);
	}

	@Test
	public void federatedRCBindSP() {
		federatedRCBind(ExecMode.SPARK);
	}

	public void federatedRCBind(ExecMode execMode) {
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		ExecMode platformOld = rtplatform;

		getAndLoadTestConfiguration(TEST_NAME);
		String HOME = SCRIPT_DIR + TEST_DIR;

		if(partitioned)
			rows = rows / 2;

		double[][] A1 = getRandomMatrix(rows, cols, -10, 10, 1, 1);
		writeInputMatrixWithMTD("A1", A1, false, new MatrixCharacteristics(rows, cols, blocksize, rows * cols));
		double[][] B1 = getRandomMatrix(rows, cols, -10, 10, 1, 2);
		writeInputMatrixWithMTD("B1", B1, false, new MatrixCharacteristics(rows, cols, blocksize, rows * cols));
		
		double[][] A2 = partitioned ? getRandomMatrix(rows, cols, -10, 10, 1, 1) : null;
		double[][] B2 = partitioned ? getRandomMatrix(rows, cols, -10, 10, 1, 2) : null;
		if(partitioned) {
			writeInputMatrixWithMTD("A2", A2, false, new MatrixCharacteristics(rows, cols, blocksize, rows * cols));
			writeInputMatrixWithMTD("B2", B2, false, new MatrixCharacteristics(rows, cols, blocksize, rows * cols));
		}

		int port1 = getRandomAvailablePort();
		int port2 = getRandomAvailablePort();
		int port3 = getRandomAvailablePort();
		int port4 = getRandomAvailablePort();
		Thread t1 = startLocalFedWorkerThread(port1, FED_WORKER_WAIT_S);
		Thread t2 = startLocalFedWorkerThread(port2);
		Thread t3 = startLocalFedWorkerThread(port3);
		Thread t4 = startLocalFedWorkerThread(port4);

		// we need the reference file to not be written to hdfs, so we get the correct format
		rtplatform = ExecMode.SINGLE_NODE;
		// Run reference dml script with normal matrix for Row/Col sum
		fullDMLScriptName = HOME + TEST_NAME + "Reference.dml";
		programArgs = new String[] {"-nvargs", "in_A1=" + input("A1"), "in_A2=" + input("A2"),
			"in_B1=" + input("B1"), "in_B2=" + input("B2"),
			"in_partitioned=" + Boolean.toString(partitioned).toUpperCase(),
			"out_R_FF_misaligned=" + expected("R_FF_misaligned"),
			"out_C_FF_aligned=" + expected("C_FF_aligned"), "out_R_FF=" + expected("R_FF"),
			"out_R_FL=" + expected("R_FL"), "out_R_LF=" + expected("R_LF"), "out_C_FF=" + expected("C_FF"),
			"out_C_FL=" + expected("C_FL"), "out_C_LF=" + expected("C_LF")};
		runTest(true, false, null, -1);

		// reference file should not be written to hdfs, so we set platform here
		rtplatform = execMode;
		if(rtplatform == ExecMode.SPARK) {
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		}
		TestConfiguration config = availableTestConfigurations.get(TEST_NAME);
		loadTestConfiguration(config);
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[] {"-stats", "-nvargs",
			"in_A1=" + TestUtils.federatedAddress(port1, input("A1")),
			"in_A2=" + TestUtils.federatedAddress(port2, input("A2")),
			"in_B1=" + TestUtils.federatedAddress(port3, input("B1")),
			"in_B2=" + TestUtils.federatedAddress(port4, input("B2")),
			"in_partitioned=" + Boolean.toString(partitioned).toUpperCase(),
			"in_B1_local=" + input("B1"), "in_B2_local=" + input("B2"), "rows=" + rows, "cols=" + cols,
			"out_R_FF_misaligned=" + output("R_FF_misaligned"), "out_C_FF_aligned=" + output("C_FF_aligned"),
			"out_R_FF=" + output("R_FF"), "out_R_FL=" + output("R_FL"), "out_R_LF=" + output("R_LF"),
			"out_C_FF=" + output("C_FF"), "out_C_FL=" + output("C_FL"), "out_C_LF=" + output("C_LF")};

		runTest(true, false, null, -1);

		// compare all sums via files
		compareResults(1e-11);

		Assert.assertTrue(heavyHittersContainsString(rtplatform == ExecMode.SPARK ? "fed_mappend" : "fed_append", 1, 8));

		TestUtils.shutdownThreads(t1, t2, t3, t4);
		rtplatform = platformOld;
		DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
	}
}
