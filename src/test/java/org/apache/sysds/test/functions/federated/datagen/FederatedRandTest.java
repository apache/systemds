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

package org.apache.sysds.test.functions.federated.datagen;

import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.instructions.fed.FEDInstructionUtils;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class FederatedRandTest extends AutomatedTestBase {

	private final static String TEST_DIR = "functions/federated/datagen/";
	private final static String TEST_NAME1 = "FederatedRandTest1";
	private final static String TEST_NAME2 = "FederatedRandTest2";
	private final static String TEST_NAME3 = "FederatedRandTest3";
	private final static String TEST_CLASS_DIR = TEST_DIR + FederatedRandTest.class.getSimpleName() + "/";

	private final static int blocksize = 1024;
	@Parameterized.Parameter()
	public int rows;
	@Parameterized.Parameter(1)
	public int cols;
	@Parameterized.Parameter(2)
	public double min;
	@Parameterized.Parameter(3)
	public double max;
	@Parameterized.Parameter(4)
	public double sparsity;
	@Parameterized.Parameter(5)
	public int seed;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"Z"}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] {"Z"}));
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] {"Z"}));
	}

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] {
			{12, 10, 0, 1, 0.7, 123},
			{1200, 10, -10, 10, 0.9, 123}
		});
	}

	@Test
	public void federatedRand1CP() {
		federatedRand(TEST_NAME1, Types.ExecMode.SINGLE_NODE);
	}

	@Test
	public void federatedRand2CP() {
		federatedRand(TEST_NAME2, Types.ExecMode.SINGLE_NODE);
	}

	@Test
	public void federatedRand3CP() {
		federatedRand(TEST_NAME3, Types.ExecMode.SINGLE_NODE);
	}

	public void federatedRand(String TEST_NAME, Types.ExecMode execMode) {
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		Types.ExecMode platformOld = rtplatform;
		rtplatform = execMode;
		if(rtplatform == Types.ExecMode.SPARK) {
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		}

		getAndLoadTestConfiguration(TEST_NAME);
		String HOME = SCRIPT_DIR + TEST_DIR;

		// write input matrices
		int halfRows = rows / 2;
		// We have two matrices handled by a single federated worker
		double[][] X1 = getRandomMatrix(halfRows, cols, 0, 1, 1, 42);
		double[][] X2 = getRandomMatrix(halfRows, cols, 0, 1, 1, 1340);

		writeInputMatrixWithMTD("X1", X1, false, new MatrixCharacteristics(halfRows, cols, blocksize, halfRows * cols));
		writeInputMatrixWithMTD("X2", X2, false, new MatrixCharacteristics(halfRows, cols, blocksize, halfRows * cols));

		int port1 = getRandomAvailablePort();
		int port2 = getRandomAvailablePort();
		Thread t1 = startLocalFedWorkerThread(port1, FED_WORKER_WAIT_S);
		Thread t2 = startLocalFedWorkerThread(port2);

		TestConfiguration config = availableTestConfigurations.get(TEST_NAME);
		loadTestConfiguration(config);

		// Run reference dml script with normal matrix
		fullDMLScriptName = HOME + TEST_NAME + "Reference.dml";
		programArgs = new String[] {"-nvargs", input("X1"), input("X2"), expected("Z")};
		runTest(null);

		FEDInstructionUtils.fedDataGen = true;

		// Run actual dml script with federated matrix
		OptimizerUtils.FEDERATED_COMPILATION = true;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";

		programArgs = new String[] {"-nvargs", "X1=" + TestUtils.federatedAddress(port1, input("X1")),
			"X2=" + TestUtils.federatedAddress(port2, input("X2")),
			"r=" + rows, "c=" + cols, "min=" + min, "max=" + max, "sp=" + sparsity, "seed=" + seed, "Z=" + output("Z")};
		runTest(null);

		HashMap<MatrixValue.CellIndex, Double> output = TestUtils.readDMLMatrixFromHDFS(outputDirectories[0]);
		int rowOut = output.keySet().stream().max(Comparator.comparingInt(e -> e.row)).get().row;
		int colOut = output.keySet().stream().max(Comparator.comparingInt(e -> e.column)).get().column;
		double minOut = output.values().stream().min(Comparator.comparingDouble(Double::doubleValue)).get();
		double maxOut = output.values().stream().max(Comparator.comparingDouble(Double::doubleValue)).get();

		// compare via files
		switch(TEST_NAME) {
			case TEST_NAME2: checkResults(rows, cols, minOut, maxOut); break;
			default: checkResults(rows, cols, min, max); break;
		}
		TestUtils.shutdownThreads(t1, t2);

		rtplatform = platformOld;
		DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		OptimizerUtils.FEDERATED_COMPILATION = false;
	}
}
