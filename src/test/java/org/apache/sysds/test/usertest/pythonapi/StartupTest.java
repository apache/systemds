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
import org.apache.log4j.spi.LoggingEvent;
import org.apache.sysds.api.PythonDMLScript;
import org.apache.sysds.test.LoggingUtils;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.util.List;


/** Simple tests to verify startup of Python Gateway server happens without crashes */
public class StartupTest {
	private LoggingUtils.TestAppender appender;

	@Before
	public void setUp() {
		appender = LoggingUtils.overwrite();
		PythonDMLScript.setDMLGateWayListenerLoggerLevel(Level.ALL);
	}

	@After
	public void tearDown() {
		LoggingUtils.reinsert(appender);
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
	public void testStartupCorrect() throws Exception {
		PythonDMLScript.main(new String[]{"-python", "4001"});
		Thread.sleep(200);
		PythonDMLScript.GwS.shutdown();
		Thread.sleep(200);
		assertLogMessages(
				"GatewayServer started",
				"Starting JVM shutdown",
				"Shutdown done",
				"GatewayServer stopped"
		);
	}
}
