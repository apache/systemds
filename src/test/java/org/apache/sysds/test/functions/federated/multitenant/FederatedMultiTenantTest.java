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

import java.io.IOException;
import java.lang.Math;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;

import static org.junit.Assert.fail;

import org.apache.commons.io.IOUtils;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.After;
import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class FederatedMultiTenantTest extends AutomatedTestBase {
	private final static String TEST_NAME = "FederatedMultiTenantTest";

	private final static String TEST_DIR = "functions/federated/multitenant/";
	private static final String TEST_CLASS_DIR = TEST_DIR + FederatedMultiTenantTest.class.getSimpleName() + "/";

	private final static double TOLERANCE = 0;

	private final static int blocksize = 1024;
	@Parameterized.Parameter()
	public int rows;
	@Parameterized.Parameter(1)
	public int cols;
	@Parameterized.Parameter(2)
	public boolean rowPartitioned;

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		return Arrays.asList(
			new Object[][] {
				{100, 1000, false},
				// {1000, 100, true},
		});
	}

	private ArrayList<Process> workerProcesses = new ArrayList<>();
	private ArrayList<Process> coordinatorProcesses = new ArrayList<>();

	private enum OpType {
		SUM,
		PARFOR_SUM,
		WSIGMOID,
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"S"}));
	}

	@Test
	public void testSumSameWorkersCP() {
		runMultiTenantSameWorkerTest(OpType.SUM, 4, ExecMode.SINGLE_NODE);
	}

	@Test
	@Ignore
	public void testSumSharedWorkersCP() {
		runMultiTenantSharedWorkerTest(OpType.SUM, 3, 9, ExecMode.SINGLE_NODE);
	}

	@Test
	@Ignore
	public void testSumSameWorkersSP() {
		runMultiTenantSameWorkerTest(OpType.SUM, 4, ExecMode.SPARK);
	}

//FIXME still runs into blocking
//	@Test
//	public void testSumSharedWorkersSP() {
//		runMultiTenantSharedWorkerTest(OpType.SUM, 3, 9, ExecMode.SPARK);
//	}

	@Test
	@Ignore
	public void testParforSumSameWorkersCP() {
		runMultiTenantSameWorkerTest(OpType.PARFOR_SUM, 4, ExecMode.SINGLE_NODE);
	}

	@Test
	public void testParforSumSharedWorkersCP() {
		runMultiTenantSharedWorkerTest(OpType.PARFOR_SUM, 3, 9, ExecMode.SINGLE_NODE);
	}

	@Test
	public void testParforSumSameWorkersSP() {
		runMultiTenantSameWorkerTest(OpType.PARFOR_SUM, 4, ExecMode.SPARK);
	}

	@Test
	@Ignore
	public void testParforSumSharedWorkersSP() {
		runMultiTenantSharedWorkerTest(OpType.PARFOR_SUM, 3, 9, ExecMode.SPARK);
	}

	@Test
	public void testWSigmoidSameWorkersCP() {
		runMultiTenantSameWorkerTest(OpType.WSIGMOID, 4, ExecMode.SINGLE_NODE);
	}

	@Test
	@Ignore
	public void testWSigmoidSharedWorkersCP() {
		runMultiTenantSharedWorkerTest(OpType.WSIGMOID, 3, 9, ExecMode.SINGLE_NODE);
	}

	@Test
	@Ignore
	public void testWSigmoidSameWorkersSP() {
		runMultiTenantSameWorkerTest(OpType.WSIGMOID, 4, ExecMode.SPARK);
	}

	@Test
	public void testWSigmoidSharedWorkersSP() {
		runMultiTenantSharedWorkerTest(OpType.WSIGMOID, 3, 9, ExecMode.SPARK);
	}

	// ensure that the processes are killed - even if the test throws an exception
	@After
	public void stopAllProcesses() {
		for(Process p : coordinatorProcesses)
			p.destroyForcibly();
		for(Process p : workerProcesses)
			p.destroyForcibly();
	}

	private void runMultiTenantSameWorkerTest(OpType opType, int numCoordinators, ExecMode execMode) {
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

		double[][] X1 = getRandomMatrix(r, c, 0, 3, 1, 3);
		double[][] X2 = getRandomMatrix(r, c, 0, 3, 1, 7);
		double[][] X3 = getRandomMatrix(r, c, 0, 3, 1, 8);
		double[][] X4 = getRandomMatrix(r, c, 0, 3, 1, 9);

		MatrixCharacteristics mc = new MatrixCharacteristics(r, c, blocksize, r * c);
		writeInputMatrixWithMTD("X1", X1, false, mc);
		writeInputMatrixWithMTD("X2", X2, false, mc);
		writeInputMatrixWithMTD("X3", X3, false, mc);
		writeInputMatrixWithMTD("X4", X4, false, mc);

		// empty script name because we don't execute any script, just start the worker
		fullDMLScriptName = "";

		int[] workerPorts = startFedWorkers(4);

		rtplatform = execMode;
		if(rtplatform == ExecMode.SPARK) {
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		}
		TestConfiguration config = availableTestConfigurations.get(TEST_NAME);
		loadTestConfiguration(config);

		// start the coordinator processes
		String scriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[] {"-stats", "100", "-fedStats", "100", "-nvargs",
			"in_X1=" + TestUtils.federatedAddress(workerPorts[0], input("X1")),
			"in_X2=" + TestUtils.federatedAddress(workerPorts[1], input("X2")),
			"in_X3=" + TestUtils.federatedAddress(workerPorts[2], input("X3")),
			"in_X4=" + TestUtils.federatedAddress(workerPorts[3], input("X4")),
			"rows=" + rows, "cols=" + cols, "testnum=" + Integer.toString(opType.ordinal()),
			"rP=" + Boolean.toString(rowPartitioned).toUpperCase()};
		for(int counter = 0; counter < numCoordinators; counter++)
			coordinatorProcesses.add(startCoordinator(execMode, scriptName,
				ArrayUtils.addAll(programArgs, "out_S=" + output("S" + counter))));

		joinCoordinatorsAndVerify(opType, execMode);

		// check that federated input files are still existing
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X1")));
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X2")));
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X3")));
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X4")));

		TestUtils.shutdownThreads(workerProcesses.toArray(new Process[0]));

		rtplatform = platformOld;
		DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
	}

	private void runMultiTenantSharedWorkerTest(OpType opType, int numCoordinators, int maxNumWorkers, ExecMode execMode) {
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		ExecMode platformOld = rtplatform;

		if(rtplatform == ExecMode.SPARK)
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		getAndLoadTestConfiguration(TEST_NAME);
		String HOME = SCRIPT_DIR + TEST_DIR;

		final int numPartitions = 4;
		final int numSharedWorkers = numPartitions - (int)Math.floor(maxNumWorkers / numCoordinators);
		final int numFedWorkers = (numCoordinators * (numPartitions - numSharedWorkers)) + numSharedWorkers;

		// write input matrices
		int r = rows;
		int c = cols / 4;
		if(rowPartitioned) {
			r = rows / 4;
			c = cols;
		}

		double[][] X1 = getRandomMatrix(r, c, 0, 3, 1, 3);
		double[][] X2 = getRandomMatrix(r, c, 0, 3, 1, 7);
		double[][] X3 = getRandomMatrix(r, c, 0, 3, 1, 8);
		double[][] X4 = getRandomMatrix(r, c, 0, 3, 1, 9);

		MatrixCharacteristics mc = new MatrixCharacteristics(r, c, blocksize, r * c);
		writeInputMatrixWithMTD("X1", X1, false, mc);
		writeInputMatrixWithMTD("X2", X2, false, mc);
		writeInputMatrixWithMTD("X3", X3, false, mc);
		writeInputMatrixWithMTD("X4", X4, false, mc);

		// empty script name because we don't execute any script, just start the worker
		fullDMLScriptName = "";

		int[] workerPorts = startFedWorkers(numFedWorkers);

		rtplatform = execMode;
		if(rtplatform == ExecMode.SPARK) {
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		}
		TestConfiguration config = availableTestConfigurations.get(TEST_NAME);
		loadTestConfiguration(config);

		// start the coordinator processes
		final String scriptName = HOME + TEST_NAME + ".dml";
		for(int counter = 0; counter < numCoordinators; counter++) {
			int workerIndexOffset = (numPartitions - numSharedWorkers) * counter;
			programArgs = new String[] {"-config", CONFIG_DIR + "SystemDS-MultiTenant-config.xml",
				"-stats", "100", "-fedStats", "100", "-nvargs",
				"in_X1=" + TestUtils.federatedAddress(workerPorts[workerIndexOffset], input("X1")),
				"in_X2=" + TestUtils.federatedAddress(workerPorts[workerIndexOffset + 1], input("X2")),
				"in_X3=" + TestUtils.federatedAddress(workerPorts[workerIndexOffset + 2], input("X3")),
				"in_X4=" + TestUtils.federatedAddress(workerPorts[workerIndexOffset + 3], input("X4")),
				"rows=" + rows, "cols=" + cols, "testnum=" + Integer.toString(opType.ordinal()),
				"rP=" + Boolean.toString(rowPartitioned).toUpperCase(), "out_S=" + output("S" + counter)};
			coordinatorProcesses.add(startCoordinator(execMode, scriptName, programArgs));
		}

		joinCoordinatorsAndVerify(opType, execMode);

		// check that federated input files are still existing
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X1")));
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X2")));
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X3")));
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X4")));

		TestUtils.shutdownThreads(workerProcesses.toArray(new Process[0]));

		rtplatform = platformOld;
		DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
	}

	private int[] startFedWorkers(int numFedWorkers) {
		int[] ports = new int[numFedWorkers];
		for(int counter = 0; counter < numFedWorkers; counter++) {
			ports[counter] = getRandomAvailablePort();
			@SuppressWarnings("deprecation")
			Process tmpProcess = startLocalFedWorker(ports[counter]);
			workerProcesses.add(tmpProcess);
		}
		return ports;
	}

	private Process startCoordinator(ExecMode execMode, String scriptPath, String[] args) {
		String separator = System.getProperty("file.separator");
		String classpath = System.getProperty("java.class.path");
		String path = System.getProperty("java.home") + separator + "bin" + separator + "java";

		String em = null;
		switch(execMode) {
			case SINGLE_NODE:
			em = "singlenode";
			break;
			case HYBRID:
			em = "hybrid";
			break;
			case SPARK:
			em = "spark";
			break;
		}

		ArrayList<String> argsList = new ArrayList<>();
		argsList.add("-f");
		argsList.add(scriptPath);
		argsList.add("-exec");
		argsList.add(em);
		argsList.addAll(Arrays.asList(args));

		ProcessBuilder processBuilder = new ProcessBuilder(ArrayUtils.addAll(new String[]{
			path, "-cp", classpath, DMLScript.class.getName()}, argsList.toArray(new String[0])))
			.redirectErrorStream(true);
		
		Process process = null;
		try {
			process = processBuilder.start();
		} catch(IOException ioe) {
			ioe.printStackTrace();
		}
		
		return process;
	}

	private void joinCoordinatorsAndVerify(OpType opType, ExecMode execMode) {
		// join the coordinator processes
		for(int counter = 0; counter < coordinatorProcesses.size(); counter++) {
			Process coord = coordinatorProcesses.get(counter);
			
			//wait for process, but obtain logs before to avoid blocking
			String outputLog = null, errorLog = null;
			try {
				outputLog = IOUtils.toString(coord.getInputStream());
				errorLog = IOUtils.toString(coord.getErrorStream());
				
				coord.waitFor();
			}
			catch(Exception ex) {
				ex.printStackTrace();
			}
			
			// get and print the output
			System.out.println("Output of coordinator #" + Integer.toString(counter + 1) + ":\n");
			System.out.println(outputLog);
			System.out.println(errorLog);
			Assert.assertTrue(checkForHeavyHitter(opType, outputLog, execMode));
		}

		// compare the results via files
		HashMap<CellIndex, Double> refResults = readDMLMatrixFromOutputDir("S" + 0);
		if(refResults.isEmpty())
			fail("The result of the first coordinator, which is taken as reference, is empty.");
		for(int counter = 1; counter < coordinatorProcesses.size(); counter++) {
			HashMap<CellIndex, Double> fedResults = readDMLMatrixFromOutputDir("S" + counter);
			TestUtils.compareMatrices(fedResults, refResults, TOLERANCE, "Fed" + counter, "FedRef");
		}
	}

	private static boolean checkForHeavyHitter(OpType opType, String outputLog, ExecMode execMode) {
		switch(opType) {
			case SUM:
				return outputLog.contains("fed_uak+");
			case PARFOR_SUM:
				return outputLog.contains(execMode == ExecMode.SPARK ? "fed_rblk" : "fed_uak+");
			case WSIGMOID:
				return outputLog.contains("fed_wsigmoid");
			default:
				return false;
		}
	}
}
