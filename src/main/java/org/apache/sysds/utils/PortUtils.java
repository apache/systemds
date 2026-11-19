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

package org.apache.sysds.utils;

import java.net.BindException;

/**
 * Helpers to validate TCP port numbers of the servers started from the command line, i.e., the federated worker (-w)
 * and the federated monitoring backend (-fedMonitoring).
 */
public class PortUtils {
	/**
	 * The lowest port a server is allowed to bind to. Port 0 is excluded as it binds an arbitrary free port, which
	 * peers of a federated worker cannot address.
	 */
	public static final int MIN_PORT = 1;

	/** Highest port representable in the 16-bit TCP port field. */
	public static final int MAX_PORT = 65535;

	/** Ports up to and including this one are reserved system ports on unix-like systems. */
	public static final int MAX_PRIVILEGED_PORT = 1023;

	private PortUtils() {
		// prevent instantiation of this utility class
	}

	/**
	 * Check whether the given port is inside the range of ports a server can be bound to.
	 *
	 * @param port the port to check
	 * @return true if the port is in [{@link #MIN_PORT}, {@link #MAX_PORT}]
	 */
	public static boolean isValidPort(int port) {
		return port >= MIN_PORT && port <= MAX_PORT;
	}

	/**
	 * Check whether the given port is a reserved system port, i.e., a port that typically requires elevated privileges
	 * to bind to.
	 *
	 * @param port the port to check
	 * @return true if the port is a privileged port
	 */
	public static boolean isPrivilegedPort(int port) {
		return port >= MIN_PORT && port <= MAX_PRIVILEGED_PORT;
	}

	/**
	 * Parse a port given as a command line argument.
	 *
	 * @param value  the raw argument value
	 * @param option the name of the option the value belongs to, used for the error message
	 * @return the parsed port
	 * @throws IllegalArgumentException if the value is not an integer inside the valid port range
	 */
	public static int parsePort(String value, String option) {
		final int port;
		try {
			port = Integer.parseInt(value.trim());
		}
		catch(NumberFormatException e) {
			throw new IllegalArgumentException(
				"Invalid port '" + value + "' for option " + option + ": not an integer, " + rangeHint());
		}
		if(!isValidPort(port))
			throw new IllegalArgumentException(
				"Invalid port " + port + " for option " + option + ": out of range, " + rangeHint());
		return port;
	}

	/** The valid range, phrased for an error message. */
	private static String rangeHint() {
		return "expected a port in [" + MIN_PORT + ", " + MAX_PORT + "]";
	}

	/**
	 * Translate a failure of a server bind into a message that names the likely cause, since the exceptions of the
	 * underlying socket implementation are platform-specific and terse.
	 *
	 * @param port the port the server tried to bind to
	 * @param e    the exception the bind failed with
	 * @return a human-readable explanation of the failure
	 */
	public static String explainBindFailure(int port, Throwable e) {
		final String msg = (e.getMessage() != null) ? e.getMessage() : e.getClass().getSimpleName();
		if(!isValidPort(port))
			return "port " + port + " is out of range, " + rangeHint() + " (" + msg + ")";
		if(e instanceof BindException) {
			if(msg.toLowerCase().contains("permission denied"))
				return "no permission to bind port " + port + (isPrivilegedPort(port) ? ", ports below "
					+ (MAX_PRIVILEGED_PORT + 1) + " are reserved and require elevated privileges" : "") + " (" + msg
					+ ")";
			return "port " + port + " is already in use (" + msg + ")";
		}
		return msg;
	}
}
