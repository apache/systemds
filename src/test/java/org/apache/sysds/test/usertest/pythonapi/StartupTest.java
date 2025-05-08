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

package org.apache.sysds.test.usertest.pythonapi;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.log4j.spi.LoggingEvent;
import org.apache.sysds.api.PythonDMLScript;
import org.apache.sysds.test.LoggingUtils;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import py4j.GatewayServer;

import java.security.Permission;
import java.util.List;


/** Simple tests to verify startup of Python Gateway server happens without crashes */
public class StartupTest {
	private LoggingUtils.TestAppender appender;
	@SuppressWarnings("removal")
	private SecurityManager sm;

	@Before
	@SuppressWarnings("removal")
	public void setUp() {
		appender = LoggingUtils.overwrite();
		sm = System.getSecurityManager();
		System.setSecurityManager(new NoExitSecurityManager());
		PythonDMLScript.setDMLGateWayListenerLoggerLevel(Level.ALL);
		Logger.getLogger(PythonDMLScript.class.getName()).setLevel(Level.ALL);
	}

	@After
	@SuppressWarnings("removal")
	public void tearDown() {
		LoggingUtils.reinsert(appender);
		System.setSecurityManager(sm);
	}

	private void assertLogMessages(String... expectedMessages) {
		List<LoggingEvent> log = LoggingUtils.reinsert(appender);
		log.stream().forEach(l -> System.out.println(l.getMessage()));
		Assert.assertEquals("Unexpected number of log messages", expectedMessages.length, log.size());

		for (int i = 0; i < expectedMessages.length; i++) {
			// order does not matter
			boolean found = false;
			for (String message : expectedMessages) {
				found |= log.get(i).getMessage().toString().startsWith(message);
			}
			Assert.assertTrue("Unexpected log message: " + log.get(i).getMessage(),found);
		}
	}

	@Test(expected = Exception.class)
	public void testStartupIncorrect_1() throws Exception {
		PythonDMLScript.main(new String[] {});
	}

	@Test(expected = Exception.class)
	public void testStartupIncorrect_2() throws Exception {
		PythonDMLScript.main(new String[] {""});
	}

	@Test(expected = Exception.class)
	public void testStartupIncorrect_3() throws Exception {
		PythonDMLScript.main(new String[] {"131", "131"});
	}

	@Test(expected = Exception.class)
	public void testStartupIncorrect_4() throws Exception {
		PythonDMLScript.main(new String[] {"Hello"});
	}

	@Test(expected = Exception.class)
	public void testStartupIncorrect_5() throws Exception {
		// Number out of range
		PythonDMLScript.main(new String[] {"-python", "918757"});
	}

	@Test
	public void testStartupIncorrect_6() throws Exception {
		GatewayServer gws1 = null;
		try {
			PythonDMLScript.main(new String[]{"-python", "4001"});
			gws1 = PythonDMLScript.GwS;
			Thread.sleep(200);
			PythonDMLScript.main(new String[]{"-python", "4001"});
			Thread.sleep(200);
		} catch (SecurityException e) {
			assertLogMessages(
					"GatewayServer started",
					"failed startup"
			);
			gws1.shutdown();
		}
	}

	@Test
	public void testStartupCorrect() throws Exception {
		PythonDMLScript.main(new String[]{"-python", "4002"});
		Thread.sleep(200);
		PythonDMLScript script = (PythonDMLScript) PythonDMLScript.GwS.getGateway().getEntryPoint();
		script.getConnection();
		PythonDMLScript.GwS.shutdown();
		Thread.sleep(200);
		assertLogMessages(
				"GatewayServer started",
				"Starting JVM shutdown",
				"Shutdown done",
				"GatewayServer stopped"
		);
	}

	@SuppressWarnings("removal")
	class NoExitSecurityManager extends SecurityManager {
		@Override
		public void checkPermission(Permission perm) { }

		@Override
		public void checkExit(int status) {
			throw new SecurityException("Intercepted exit()");
		}
	}
}
