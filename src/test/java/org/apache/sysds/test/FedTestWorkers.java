/*
 *  Licensed to the Apache Software Foundation (ASF) under one
 *  or more contributor license agreements.  See the NOTICE file
 *  distributed with this work for additional information
 *  regarding copyright ownership.  The ASF licenses this file
 *  to you under the Apache License, Version 2.0 (the
 *  "License"); you may not use this file except in compliance
 *  with the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing,
 *  software distributed under the License is distributed on an
 *  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 *  KIND, either express or implied.  See the License for the
 *  specific language governing permissions and limitations
 *  under the License.
 */

package org.apache.sysds.test;

import java.security.InvalidParameterException;
import java.util.Arrays;

import org.apache.sysds.runtime.DMLRuntimeException;

/**
 * This class is used for running federated tests. It will run the required workers, a single one as a thread and the
 * rest as processes, this allows for easy debugging, while getting good performance.
 */
public class FedTestWorkers {
	private final AutomatedTestBase _test;
	private final int _numWorkers;
	private int[] _ports = null;
	private Thread _debugWorker = null;
	private Process[] _remainingWorkers = null;
	
	/**
	 * Create wrapper for federated workers and start it.
	 * @param numWorkers the number of federated workers
	 */
	public FedTestWorkers(AutomatedTestBase test, int numWorkers) {
		if (numWorkers < 1) {
			throw new InvalidParameterException("at least 1 worker required");
		}
		_test = test;
		_numWorkers = numWorkers;
	}
	
	/**
	 * Start the workers, returning the ports for them. The first port is the one of the worker running on the thread,
	 * for which breakpoints can be set, all other workers will not stop at debug breakpoints.
	 * @return ports
	 */
	public int[] start() throws Exception {
		if (_ports != null &&  _ports.length != 0)
			throw new DMLRuntimeException("Federated workers have to be stopped before starting again.");
		_ports = new int[_numWorkers];
		for (int i = 0; i < _numWorkers; i++)
			_ports[i] = AutomatedTestBase.getRandomAvailablePort();
		
		// first start all processes, during their startup we start the thread and then wait for processes
		_remainingWorkers = new Process[_numWorkers - 1];
		for(int i = 0; i < _remainingWorkers.length; i++) {
			_remainingWorkers[i] = _test.startLocalFedWorker(_ports[i + 1], false);
		}
		_debugWorker = _test.startLocalFedWorkerThread(_ports[0]);
		for(int i = 1; i < _ports.length; i++) {
			AutomatedTestBase.waitForFederatedWorker(_ports[i]);
		}
		return _ports;
	}
	
	/**
	 * Stop the workers.
	 */
	public void stop() {
		TestUtils.shutdownThread(_debugWorker);
		TestUtils.shutdownProcesses(_remainingWorkers);
		_ports = new int[0];
	}
}
