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

package org.apache.sysds.test.component.federated;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.IOException;
import java.net.ServerSocket;

import org.apache.commons.cli.ParseException;
import org.apache.sysds.api.DMLOptions;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.federated.FederatedWorker;
import org.apache.sysds.utils.PortUtils;
import org.junit.Test;

/**
 * Tests that a federated worker started with a port it cannot bind to reports the reason instead of terminating
 * silently, covering out of range, reserved and already occupied ports.
 */
public class FederatedWorkerPortTest {

	@Test
	public void validPortRange() {
		assertFalse(PortUtils.isValidPort(-1));
		assertFalse(PortUtils.isValidPort(0));
		assertTrue(PortUtils.isValidPort(1));
		assertTrue(PortUtils.isValidPort(DMLConfig.DEFAULT_FEDERATED_PORT));
		assertTrue(PortUtils.isValidPort(65535));
		assertFalse(PortUtils.isValidPort(65536));
		assertFalse(PortUtils.isValidPort(80505));
	}

	@Test
	public void privilegedPortRange() {
		assertTrue(PortUtils.isPrivilegedPort(80));
		assertTrue(PortUtils.isPrivilegedPort(1023));
		assertFalse(PortUtils.isPrivilegedPort(1024));
		assertFalse(PortUtils.isPrivilegedPort(DMLConfig.DEFAULT_FEDERATED_PORT));
	}

	@Test
	public void parseValidPort() {
		assertEquals(4040, PortUtils.parsePort("4040", "-w"));
		assertEquals(4040, PortUtils.parsePort(" 4040 ", "-w"));
	}

	@Test
	public void parseOutOfRangePort() {
		try {
			PortUtils.parsePort("80505", "-w");
			fail("expected an exception for a port outside the valid range");
		}
		catch(IllegalArgumentException e) {
			assertTrue(e.getMessage(), e.getMessage().contains("out of range"));
		}
	}

	@Test
	public void parseNonNumericPort() {
		try {
			PortUtils.parsePort("notAPort", "-w");
			fail("expected an exception for a non numeric port");
		}
		catch(IllegalArgumentException e) {
			assertTrue(e.getMessage(), e.getMessage().contains("not an integer"));
		}
	}

	@Test
	public void cliRejectsOutOfRangePort() {
		assertParseFails(new String[] {"-w", "80505"}, "out of range");
	}

	@Test
	public void cliRejectsNegativePort() {
		// -1 is the internal marker for 'use the default port' and must not be accepted from outside
		assertParseFails(new String[] {"-w", "-1"}, "out of range");
	}

	@Test
	public void cliRejectsPortZero() {
		// port 0 binds an arbitrary free port, which no coordinator could address
		assertParseFails(new String[] {"-w", "0"}, "out of range");
	}

	@Test
	public void cliRejectsNonNumericPort() {
		assertParseFails(new String[] {"-w", "notAPort"}, "not an integer");
	}

	@Test
	public void cliRejectsOutOfRangeMonitoringPort() {
		assertParseFails(new String[] {"-fedMonitoring", "80505"}, "out of range");
	}

	@Test
	public void cliAcceptsValidPort() throws ParseException {
		DMLOptions opts = DMLOptions.parseCLArguments(new String[] {"-w", "8001"});
		assertTrue(opts.fedWorker);
		assertEquals(8001, opts.fedWorkerPort);
	}

	@Test
	public void cliDefaultsWithoutPort() throws ParseException {
		// the port argument is optional, a missing one falls back to the default federated port
		DMLOptions opts = DMLOptions.parseCLArguments(new String[] {"-w"});
		assertTrue(opts.fedWorker);
		assertEquals(-1, opts.fedWorkerPort);
	}

	@Test
	public void workerRejectsOutOfRangePort() {
		try {
			new FederatedWorker(80505, false);
			fail("expected the federated worker to reject a port outside the valid range");
		}
		catch(DMLRuntimeException e) {
			assertTrue(e.getMessage(), e.getMessage().contains("out of range"));
		}
	}

	@Test
	public void workerReportsOccupiedPort() throws IOException {
		try(ServerSocket occupied = new ServerSocket(0)) {
			final int port = occupied.getLocalPort();
			try {
				new FederatedWorker(port, false);
				fail("expected the federated worker to reject the already occupied port " + port);
			}
			catch(DMLRuntimeException e) {
				assertTrue(e.getMessage(), e.getMessage().contains("already in use"));
				assertTrue(e.getMessage(), e.getMessage().contains(String.valueOf(port)));
			}
		}
	}

	private static void assertParseFails(String[] args, String expectedMessagePart) {
		try {
			DMLOptions.parseCLArguments(args);
			fail("expected a parse exception for arguments " + String.join(" ", args));
		}
		catch(ParseException e) {
			assertTrue(e.getMessage(), e.getMessage().contains(expectedMessagePart));
		}
	}
}
