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

package org.apache.sysds.test.functions.federated.multitenant;

import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class FederatedReuseSlicesTest extends MultiTenantTestBase {
	private static final Log LOG = LogFactory.getLog(FederatedReuseSlicesTest.class.getName());
	private final static String TEST_NAME = "FederatedReuseSlicesTest";
	private final static String TEST_DIR = "functions/federated/multitenant/";
	private static final String TEST_CLASS_DIR = TEST_DIR + FederatedReuseSlicesTest.class.getSimpleName() + "/";
	private final static double TOLERANCE = 0;
	private final static int blocksize = 1024;
	
	@Parameterized.Parameter()
	public int rows;
	@Parameterized.Parameter(1)
	public int cols;
	@Parameterized.Parameter(2)
	public double sparsity;
	@Parameterized.Parameter(3)
	public boolean rowPartitioned;

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		return Arrays.asList(
			new Object[][] {
				{100, 200, 0.9, false},
				// {200, 100, 0.9, true},
				// {100, 1000, 0.01, false},
				// {1000, 100, 0.01, true},
		});
	}

	private enum OpType {
		EW_MULT,
		RM_EMPTY,
		PARFOR_DIV,
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"S"}));
	}

	@Test
	public void testElementWisePlusCP() {
		runReuseSlicesTest(OpType.EW_MULT, 4, ExecMode.SINGLE_NODE);
	}

	@Test
	@Ignore
	public void testElementWisePlusSP() {
		runReuseSlicesTest(OpType.EW_MULT, 4, ExecMode.SPARK);
	}

	@Test
	public void testRemoveEmptyCP() {
		runReuseSlicesTest(OpType.RM_EMPTY, 4, ExecMode.SINGLE_NODE);
	}

	@Test
	@Ignore // NOTE: federated removeEmpty not supported in spark execution mode yet
	public void testRemoveEmptySP() {
		runReuseSlicesTest(OpType.RM_EMPTY, 4, ExecMode.SPARK);
	}

	@Test
	@Ignore
	public void testParforDivCP() {
		runReuseSlicesTest(OpType.PARFOR_DIV, 4, ExecMode.SINGLE_NODE);
	}

	@Test
	public void testParforDivSP() {
		runReuseSlicesTest(OpType.PARFOR_DIV, 4, ExecMode.SPARK);
	}

	private void runReuseSlicesTest(OpType opType, int numCoordinators, ExecMode execMode) {
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		ExecMode platformOld = rtplatform;

		if(rtplatform == ExecMode.SPARK)
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		getAndLoadTestConfiguration(TEST_NAME);
		String HOME = SCRIPT_DIR + TEST_DIR;

		// write input matrices
		int r = rows;
		int c = cols / 4;
		if(rowPartitioned) {
			r = rows / 4;
			c = cols;
		}

		double[][] X1 = getRandomMatrix(r, c, 0, 3, sparsity, 3);
		double[][] X2 = getRandomMatrix(r, c, 0, 3, sparsity, 7);
		double[][] X3 = getRandomMatrix(r, c, 0, 3, sparsity, 8);
		double[][] X4 = getRandomMatrix(r, c, 0, 3, sparsity, 9);

		MatrixCharacteristics mc = new MatrixCharacteristics(r, c, blocksize, r * c);
		writeInputMatrixWithMTD("X1", X1, false, mc);
		writeInputMatrixWithMTD("X2", X2, false, mc);
		writeInputMatrixWithMTD("X3", X3, false, mc);
		writeInputMatrixWithMTD("X4", X4, false, mc);

		// empty script name because we don't execute any script, just start the worker
		fullDMLScriptName = "";

		int[] workerPorts = startFedWorkers(4, new String[]{"-lineage", "reuse"});

		rtplatform = execMode;
		if(rtplatform == ExecMode.SPARK) {
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		}
		TestConfiguration config = availableTestConfigurations.get(TEST_NAME);
		loadTestConfiguration(config);

		// start the coordinator processes
		String scriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[] {"-config", CONFIG_DIR + "SystemDS-MultiTenant-config.xml",
			"-lineage", "reuse", "-stats", "100", "-fedStats", "100", "-nvargs",
			"in_X1=" + TestUtils.federatedAddress(workerPorts[0], input("X1")),
			"in_X2=" + TestUtils.federatedAddress(workerPorts[1], input("X2")),
			"in_X3=" + TestUtils.federatedAddress(workerPorts[2], input("X3")),
			"in_X4=" + TestUtils.federatedAddress(workerPorts[3], input("X4")),
			"rows=" + rows, "cols=" + cols, "testnum=" + Integer.toString(opType.ordinal()),
			"rP=" + Boolean.toString(rowPartitioned).toUpperCase()};

		for(int counter = 0; counter < numCoordinators; counter++) {
			// start coordinators with alternating boolean mod_fedMap --> change order of fed partitions
			startCoordinator(execMode, scriptName,
			ArrayUtils.addAll(programArgs, "out_S=" + output("S" + counter),
			"mod_fedMap=" + Boolean.toString(counter % 2 == 1).toUpperCase()));
			
		}
		// wait for the coordinator processes to end and verify the results
		String coordinatorOutput = waitForCoordinators();
		LOG.debug(coordinatorOutput);

		// verify correctness of results.
		// we no long verify if the number of hitters are correct since it is harder to maintain.
		verifyResults();

		// check that federated input files are still existing
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X1")));
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X2")));
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X3")));
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X4")));

		TestUtils.shutdownThreads(workerProcesses.toArray(new Process[0]));

		rtplatform = platformOld;
		DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
	}

	private void verifyResults() {
		// compare the results via files
		HashMap<CellIndex, Double> refResults0	= readDMLMatrixFromOutputDir("S" + 0);
		HashMap<CellIndex, Double> refResults1	= readDMLMatrixFromOutputDir("S" + 1);
		Assert.assertFalse("The result of the first coordinator, which is taken as reference, is empty.",
			refResults0.isEmpty());
		Assert.assertFalse("The result of the second coordinator, which is taken as reference, is empty.",
			refResults1.isEmpty());

		boolean compareEqual = true;
		for(CellIndex index : refResults0.keySet()) {
			compareEqual &= refResults0.get(index).equals(refResults1.get(index));
			if(!compareEqual)
				break;
		}
		Assert.assertFalse("The result of the first coordinator should be different than the "
			+ "result of the second coordinator (due to modified federated maps).", compareEqual);

		for(int counter = 2; counter < coordinatorProcesses.size(); counter++) {
			HashMap<CellIndex, Double> fedResults = readDMLMatrixFromOutputDir("S" + counter);
			TestUtils.compareMatrices(fedResults, (counter % 2 == 0) ? refResults0 : refResults1,
				TOLERANCE, "Fed" + counter, "FedRef");
		}
	}

}
