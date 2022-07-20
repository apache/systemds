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

package org.apache.sysds.api;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.jmlc.Connection;
import org.apache.sysds.conf.CompilerConfig;

import py4j.GatewayServer;
import py4j.GatewayServerListener;
import py4j.Py4JNetworkException;
import py4j.Py4JServerConnection;

public class PythonDMLScript {

	private Connection _connection;

	/**
	 * Entry point for Python API.
	 * 
	 * @param args Command line arguments.
	 * @throws Exception Throws exceptions if there is issues in startup or while running.
	 */
	public static void main(String[] args) throws Exception {
		final DMLOptions dmlOptions = DMLOptions.parseCLArguments(args);
		DMLScript.loadConfiguration(dmlOptions.configFile);
		start(dmlOptions.pythonPort);
	}

	private static void start(int port) throws Py4JNetworkException {
		GatewayServer GwS = new GatewayServer(new PythonDMLScript(), port);
		GwS.addListener(new DMLGateWayListener());
		GwS.start();
	}

	private PythonDMLScript() {
		// we enable multi-threaded I/O and operations for a single JMLC
		// connection because the calling Python process is unlikely to run
		// multi-threaded streams of operations on the same shared context
		_connection = new Connection();
	}

	public Connection getConnection() {
		return _connection;
	}
	
	protected static class DMLGateWayListener implements GatewayServerListener {
		private static final Log LOG = LogFactory.getLog(DMLGateWayListener.class.getName());
	
		@Override
		public void connectionError(Exception e) {
			LOG.warn("Connection error: " + e.getMessage());
			System.exit(1);
		}
	
		@Override
		public void connectionStarted(Py4JServerConnection gatewayConnection) {
			LOG.debug("Connection Started: " + gatewayConnection.toString());
		}
	
		@Override
		public void connectionStopped(Py4JServerConnection gatewayConnection) {
			LOG.debug("Connection stopped: " + gatewayConnection.toString());
		}
	
		@Override
		public void serverError(Exception e) {
			LOG.error("Server Error " + e.getMessage());
		}
	
		@Override
		public void serverPostShutdown() {
			LOG.info("Shutdown done");
			System.exit(0);
		}
	
		@Override
		public void serverPreShutdown() {
			LOG.info("Starting JVM shutdown");
		}
	
		@Override
		public void serverStarted() {
			// message the python interface that the JVM is ready.
			System.out.println("GatewayServer Started");
		}
	
		@Override
		public void serverStopped() {
			System.out.println("GatewayServer Stopped");
			System.exit(0);
		}
	}
}

