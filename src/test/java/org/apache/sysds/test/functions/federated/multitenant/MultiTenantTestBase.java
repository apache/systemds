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

import static org.junit.Assert.fail;

import java.io.IOException;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import org.apache.commons.io.IOUtils;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.test.AutomatedTestBase;
import org.junit.After;

import com.google.crypto.tink.subtle.Random;

public abstract class MultiTenantTestBase extends AutomatedTestBase {
	protected static final Log LOG = LogFactory.getLog(MultiTenantTestBase.class.getName());

	protected ArrayList<Process> workerProcesses = new ArrayList<>();
	protected ArrayList<Process> coordinatorProcesses = new ArrayList<>();

	@Override
	public abstract void setUp();

	// ensure that the processes are killed - even if the test throws an exception
	@After
	public void stopAllProcesses() {
		for(Process p : coordinatorProcesses)
			p.destroyForcibly();
		for(Process p : workerProcesses)
			p.destroyForcibly();
	}

	protected int[] startFedWorkers(int numFedWorkers) {
		return startFedWorkers(numFedWorkers, null);
	}

	/**
	 * Start numFedWorkers federated worker processes on available ports and add them to the workerProcesses
	 *
	 * @param numFedWorkers the number of federated workers to start
	 * @return int[] the ports of the created federated workers
	 */
	protected int[] startFedWorkers(int numFedWorkers, String[] addArgs) {
		int[] ports = new int[numFedWorkers];
		for(int counter = 0; counter < numFedWorkers; counter++) {
			ports[counter] = getRandomAvailablePort();
			// start process but only wait long for last one.
			Process tmpProcess = startLocalFedWorker(ports[counter], addArgs,
				counter == numFedWorkers - 1 ? (FED_WORKER_WAIT + Random.randInt(1000)) * 3 : FED_WORKER_WAIT_S);
			workerProcesses.add(tmpProcess);
		}
		return ports;
	}

	/**
	 * Start a coordinator process running the specified script with given arguments and add it to the
	 * coordinatorProcesses
	 *
	 * @param execMode   the execution mode of the coordinator
	 * @param scriptPath the path to the dml script
	 * @param args       the program arguments for running the dml script
	 */
	protected void startCoordinator(ExecMode execMode, String scriptPath, String[] args) {
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

		// create the processBuilder and redirect the stderr to its stdout
		ProcessBuilder processBuilder = new ProcessBuilder(ArrayUtils
			.addAll(new String[] {path,
				"-Xmx1000m", "-Xms1000m", "-Xmn100m", 
				"--add-opens=java.base/java.nio=ALL-UNNAMED" ,
				"--add-opens=java.base/java.io=ALL-UNNAMED" ,
				"--add-opens=java.base/java.util=ALL-UNNAMED" ,
				"--add-opens=java.base/java.lang=ALL-UNNAMED" ,
				"--add-opens=java.base/java.lang.ref=ALL-UNNAMED" ,
				"--add-opens=java.base/java.util.concurrent=ALL-UNNAMED" ,
				"--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
				"--add-modules=jdk.incubator.vector",
				"-cp", classpath, DMLScript.class.getName()}, argsList.toArray(new String[0])));

		Process process = null;
		try {
			process = processBuilder.start();
		}
		catch(IOException ioe) {
			ioe.printStackTrace();
			fail("Can't start the coordinator process.");
		}
		coordinatorProcesses.add(process);
	}

	/**
	 * Wait for all processes of coordinatorProcesses to terminate and collect their output
	 *
	 * @return String the collected output of the coordinator processes
	 */
	protected String waitForCoordinators() {
		return waitForCoordinators(500);
	}

	protected String waitForCoordinators(int timeout){
		ExecutorService executor = Executors.newSingleThreadExecutor();
		try{
			return executor.submit(() -> waitForCoordinatorsActual()).get(timeout, TimeUnit.SECONDS);
		}
		catch(Exception e){
			throw new RuntimeException(e);
		}
		finally{
			executor.shutdown();
		}
	}

	private String waitForCoordinatorsActual(){
		// wait for the coordinator processes to finish and collect their output
		StringBuilder outputLog = new StringBuilder();
		for(int counter = 0; counter < coordinatorProcesses.size(); counter++) {
			Process coord = coordinatorProcesses.get(counter);
			try {
				outputLog.append("\n");
				outputLog.append("Output of coordinator #" + Integer.toString(counter + 1) + ":\n");
				outputLog.append(IOUtils.toString(coord.getInputStream(), Charset.defaultCharset()));
				outputLog.append(IOUtils.toString(coord.getErrorStream(), Charset.defaultCharset()));

				coord.waitFor();
			}
			catch(Exception ex) {
				fail(ex.getClass().getSimpleName() + " thrown while collecting log output of coordinator #"
					+ Integer.toString(counter + 1) + ".\n");
				ex.printStackTrace();
			}
		}
		return outputLog.toString();
	}
}
