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

package org.apache.sysds.pythonapi;

import org.apache.sysds.api.jmlc.Connection;
import py4j.GatewayServer;

import java.io.BufferedReader;
import java.io.InputStreamReader;

public class PythonDMLScript {
	private Connection _connection;
	
	public static void main(String[] args) {
		if (start()) {
			System.exit(0);
		}
		else {
			System.exit(1);
		}
	}
	
	public static boolean start() {
		GatewayServer gatewayServer = new GatewayServer(new PythonDMLScript());
		try {
			gatewayServer.start();
		} catch (py4j.Py4JNetworkException ex) {
			System.err.println("Could not start gateway server. Is a server instance already running?");
			return false;
		}
		System.out.println("Gateway Server Started. Press Enter to stop...");
		try {
			BufferedReader stdin = new BufferedReader(new InputStreamReader(System.in));
			stdin.readLine();
			return true;
		}
		catch (java.io.IOException e) {
			return false;
		}
	}
	
	public PythonDMLScript() {
		_connection = new Connection();
	}
	
	public Connection getConnection() {
		return _connection;
	}
}
