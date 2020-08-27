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
import py4j.Py4JServerConnection;

public class PythonDMLScript {
	private static final Log LOG = LogFactory.getLog(PythonDMLScript.class.getName());
	private Connection _connection;

	/**
	 * Entry point for Python API.
	 * 
	 * The system returns with exit code 1, if the startup process fails, and 0 if the startup was successful.
	 * 
	 * @param args Command line arguments.
	 */
	public static void main(String[] args) {
		start(Integer.parseInt(args[0]));
	}

	private static void start(int port) {
		try {
			// TODO Add argument parsing here.
			GatewayServer GwS = new GatewayServer(new PythonDMLScript(), port);
			GwS.addListener(new DMLGateWayListener());
			GwS.start();
		}
		catch(py4j.Py4JNetworkException ex) {
			LOG.error("Py4JNetworkException while executing the GateWay. Is a server instance already running?");
			System.exit(-1);
		}
	}

	private PythonDMLScript() {
		// we enable multi-threaded I/O and operations for a single JMLC
		// connection because the calling Python process is unlikely to run
		// multi-threaded streams of operations on the same shared context
		_connection = new Connection(
			CompilerConfig.ConfigType.PARALLEL_CP_READ_TEXTFORMATS,
			CompilerConfig.ConfigType.PARALLEL_CP_WRITE_TEXTFORMATS,
			CompilerConfig.ConfigType.PARALLEL_CP_READ_BINARYFORMATS,
			CompilerConfig.ConfigType.PARALLEL_CP_WRITE_BINARYFORMATS,
			CompilerConfig.ConfigType.PARALLEL_CP_MATRIX_OPERATIONS,
			CompilerConfig.ConfigType.PARALLEL_LOCAL_OR_REMOTE_PARFOR,
			CompilerConfig.ConfigType.ALLOW_DYN_RECOMPILATION);
	}

	public Connection getConnection() {
		return _connection;
	}
}

class DMLGateWayListener implements GatewayServerListener {
	private static final Log LOG = LogFactory.getLog(DMLGateWayListener.class.getName());

	@Override
	public void connectionError(Exception e) {
		LOG.warn("Connection error: " + e.getMessage());
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
	}
}
