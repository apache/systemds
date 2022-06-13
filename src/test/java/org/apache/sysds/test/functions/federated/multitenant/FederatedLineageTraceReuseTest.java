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
import org.apache.commons.lang3.StringUtils;
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
public class FederatedLineageTraceReuseTest extends MultiTenantTestBase {
	private final static String TEST_NAME = "FederatedLineageTraceReuseTest";

	private final static String TEST_DIR = "functions/federated/multitenant/";
	private static final String TEST_CLASS_DIR = TEST_DIR + FederatedLineageTraceReuseTest.class.getSimpleName() + "/";

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
				// {100, 200, 0.9, false},
				{200, 100, 0.9, true},
				// {100, 1000, 0.01, false},
				// {1000, 100, 0.01, true},
		});
	}

	private enum OpType {
		EW_PLUS,
		MM,
		PARFOR_ADD,
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"S"}));
	}

	@Test
	public void testElementWisePlusCP() {
		runLineageTraceReuseTest(OpType.EW_PLUS, 4, ExecMode.SINGLE_NODE);
	}

	@Test
	@Ignore
	public void testElementWisePlusSP() {
		runLineageTraceReuseTest(OpType.EW_PLUS, 4, ExecMode.SPARK);
	}

	@Test
	public void testMatrixMultCP() {
		runLineageTraceReuseTest(OpType.MM, 4, ExecMode.SINGLE_NODE);
	}

	@Test
	@Ignore // TODO: allow for reuse of respective spark instructions
	public void testMatrixMultSP() {
		runLineageTraceReuseTest(OpType.MM, 4, ExecMode.SPARK);
	}

	@Test
	public void testParforAddCP() {
		runLineageTraceReuseTest(OpType.PARFOR_ADD, 3, ExecMode.SINGLE_NODE);
	}

	@Test
	@Ignore
	public void testParforAddSP() {
		runLineageTraceReuseTest(OpType.PARFOR_ADD, 3, ExecMode.SPARK);
	}

	private void runLineageTraceReuseTest(OpType opType, int numCoordinators, ExecMode execMode) {
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
		for(int counter = 0; counter < numCoordinators; counter++)
			startCoordinator(execMode, scriptName,
				ArrayUtils.addAll(programArgs, "out_S=" + output("S" + counter)));

		// wait for the coordinator processes to end and verify the results
		String coordinatorOutput = waitForCoordinators();
		verifyResults(opType, coordinatorOutput, execMode);

		// check that federated input files are still existing
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X1")));
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X2")));
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X3")));
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X4")));

		TestUtils.shutdownThreads(workerProcesses.toArray(new Process[0]));

		rtplatform = platformOld;
		DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
	}

	private void verifyResults(OpType opType, String outputLog, ExecMode execMode) {
		Assert.assertTrue(checkForHeavyHitter(opType, outputLog, execMode));
		// verify that the matrix object has been taken from cache
		Assert.assertTrue(checkForReuses(opType, outputLog, execMode));

		// compare the results via files
		HashMap<CellIndex, Double> refResults	= readDMLMatrixFromOutputDir("S" + 0);
		Assert.assertFalse("The result of the first coordinator, which is taken as reference, is empty.",
			refResults.isEmpty());
		for(int counter = 1; counter < coordinatorProcesses.size(); counter++) {
			HashMap<CellIndex, Double> fedResults = readDMLMatrixFromOutputDir("S" + counter);
			TestUtils.compareMatrices(fedResults, refResults, TOLERANCE, "Fed" + counter, "FedRef");
		}
	}

	private boolean checkForHeavyHitter(OpType opType, String outputLog, ExecMode execMode) {
		boolean retVal = false;
		switch(opType) {
			case EW_PLUS:
				retVal = checkForHeavyHitter(outputLog, "fed_+");
				if(execMode == ExecMode.SINGLE_NODE)
					retVal &= checkForHeavyHitter(outputLog, "fed_uak+");
				break;
			case MM:
				retVal = checkForHeavyHitter(outputLog, (execMode == ExecMode.SPARK) ? "fed_mapmm" : "fed_ba+*");
				retVal &= checkForHeavyHitter(outputLog, "fed_r'");
				if(!rowPartitioned)
					retVal &= checkForHeavyHitter(outputLog, (execMode == ExecMode.SPARK) ? "fed_rblk" : "fed_uak+");
				break;
			case PARFOR_ADD:
				retVal = checkForHeavyHitter(outputLog, "fed_-");
				retVal &= checkForHeavyHitter(outputLog, "fed_+");
				retVal &= checkForHeavyHitter(outputLog, (execMode == ExecMode.SPARK) ? "fed_rblk" : "fed_uak+");
				break;
		}
		return retVal;
	}

	private boolean checkForHeavyHitter(String outputLog, String hhString) {
		int occurrences = StringUtils.countMatches(outputLog, hhString);
		return (occurrences == coordinatorProcesses.size());
	}

	private boolean checkForReuses(OpType opType, String outputLog, ExecMode execMode) {
		final String LINCACHE_MULTILVL = "LinCache MultiLvl (Ins/SB/Fn):\t";
		final String LINCACHE_WRITES = "LinCache writes (Mem/FS/Del):\t";
		final String FED_LINEAGEPUT = "Fed PutLineage (Count, Items):\t";
		boolean retVal = false;
		int multiplier = 1;
		int numInst = -1;
		int serializationWrites = 0;
		switch(opType) {
			case EW_PLUS:
				numInst = (execMode == ExecMode.SPARK) ? 1 : 2;
				break;
			case MM:
				numInst = rowPartitioned ? 2 : 3;
				serializationWrites = rowPartitioned ? 1 : 0;
				break;
			case PARFOR_ADD: // number of instructions times number of iterations of the parfor loop
				multiplier = 3;
				numInst = ((execMode == ExecMode.SPARK) ? 2 : 3) * multiplier;
				break;
		}
		retVal = outputLog.contains(LINCACHE_MULTILVL
			+ Integer.toString(numInst * (coordinatorProcesses.size()-1) * workerProcesses.size()) + "/");
		retVal &= outputLog.contains(LINCACHE_WRITES // read + instructions + serializations
			+ Integer.toString((1 + numInst + serializationWrites) * workerProcesses.size()) + "/");
		retVal &= outputLog.contains(FED_LINEAGEPUT
			+ Integer.toString(coordinatorProcesses.size() * workerProcesses.size() * multiplier) + "/");
		return retVal;
	}
}
