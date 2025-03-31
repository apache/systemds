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
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.sysds.api.jmlc.Connection;

import py4j.DefaultGatewayServerListener;
import py4j.GatewayServer;
import py4j.Py4JNetworkException;


public class PythonDMLScript {

	private static final Log LOG = LogFactory.getLog(PythonDMLScript.class.getName());
	final private Connection _connection;
	public static GatewayServer GwS;

	/**
	 * Entry point for Python API.
	 * 
	 * @param args Command line arguments.
	 * @throws Exception Throws exceptions if there is issues in startup or while running.
	 */
	public static void main(String[] args) throws Exception {
		final DMLOptions dmlOptions = DMLOptions.parseCLArguments(args);
		DMLScript.loadConfiguration(dmlOptions.configFile);
		GwS = new GatewayServer(new PythonDMLScript(), dmlOptions.pythonPort);
		GwS.addListener(new DMLGateWayListener());
		try {
			GwS.start();
		}
		catch(Py4JNetworkException p4e) {
			/**
			 * This sometimes happens when the startup is using a port already in use. In this case we handle it in python
			 * therefore use logging framework. and terminate program.
			 */
			LOG.info("failed startup", p4e);
			System.exit(-1);
		}
		catch(Exception e) {
			throw new DMLException("Failed startup and maintaining Python gateway", e);
		}
	}

	private PythonDMLScript() {
		// we enable multi-threaded I/O and operations for a single JMLC
		// connection because the calling Python process is unlikely to run
		// multi-threaded streams of operations on the same shared context
		_connection = new Connection();
	}

	public static void setDMLGateWayListenerLoggerLevel(Level l){
		Logger.getLogger(DMLGateWayListener.class).setLevel(l);
	}

	public Connection getConnection() {
		return _connection;
	}

	protected static class DMLGateWayListener extends DefaultGatewayServerListener {
		private static final Log LOG = LogFactory.getLog(DMLGateWayListener.class.getName());

		@Override
		public void serverPostShutdown() {
			LOG.info("Shutdown done");
		}

		@Override
		public void serverPreShutdown() {
			LOG.info("Starting JVM shutdown");
		}

		@Override
		public void serverStarted() {
			LOG.info("GatewayServer started");
		}

		@Override
		public void serverStopped() {
			LOG.info("GatewayServer stopped");
		}
	}

}
