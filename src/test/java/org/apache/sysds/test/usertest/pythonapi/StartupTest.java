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
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.util.UnixPipeUtils;
import org.apache.sysds.test.LoggingUtils;
import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import py4j.GatewayServer;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.security.Permission;
import java.util.List;

import static org.junit.Assert.assertArrayEquals;


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
		assertLogMessages(true, expectedMessages);
	}

	private void assertLogMessages(boolean strict, String... expectedMessages) {
		List<LoggingEvent> log = LoggingUtils.reinsert(appender);
		log.stream().forEach(l -> System.out.println(l.getMessage()));
		if (strict){
			Assert.assertEquals("Unexpected number of log messages", expectedMessages.length, log.size());

			for (int i = 0; i < expectedMessages.length; i++) {
				// order does not matter
				boolean found = false;
				for (String message : expectedMessages) {
					found |= log.get(i).getMessage().toString().startsWith(message);
				}
				Assert.assertTrue("Unexpected log message: " + log.get(i).getMessage(),found);
			}
		} else {
			for (String message : expectedMessages) {
				// order does not matter
				boolean found = false;

                for (LoggingEvent loggingEvent : log) {
                    found |= loggingEvent.getMessage().toString().startsWith(message);
                }
				Assert.assertTrue("Expected log message not found: " + message,found);
			}
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
			assertLogMessages(false,
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
				false,
				"GatewayServer started",
				"Starting JVM shutdown",
				"Shutdown done",
				"GatewayServer stopped"
		);
	}

	@Rule
	public TemporaryFolder folder = new TemporaryFolder();

	@Test
	public void testDataTransfer() throws Exception {
		PythonDMLScript.main(new String[]{"-python", "4003"});
		Thread.sleep(200);
		PythonDMLScript script = (PythonDMLScript) PythonDMLScript.GwS.getGateway().getEntryPoint();

		File in = folder.newFile("py2java-0");
		File out = folder.newFile("java2py-0");

		// Init Test
		BufferedOutputStream py2java = UnixPipeUtils.openOutput(in.getAbsolutePath(), 0);
		script.openPipes(folder.getRoot().getPath(), 1);
		BufferedInputStream java2py = UnixPipeUtils.openInput(out.getAbsolutePath(), 0);

		// Write Test
		double[] data = new double[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
		MatrixBlock mb = new MatrixBlock(2, 3, data);
		script.startWritingMbToPipe(0, mb);
		double[] rcv_data = new double[data.length];
		UnixPipeUtils.readNumpyArrayInBatches(java2py, 0, 32, data.length, Types.ValueType.FP64, rcv_data, 0);
		assertArrayEquals(data, rcv_data, 1e-9);

		// Read Test
		UnixPipeUtils.writeNumpyArrayInBatches(py2java, 0, 32, data.length, Types.ValueType.FP64, mb);
		MatrixBlock rcv_mb = script.startReadingMbFromPipe(0, 2, 3, Types.ValueType.FP64);
		assertArrayEquals(data, rcv_mb.getDenseBlockValues(), 1e-9);


		script.closePipes();

		PythonDMLScript.GwS.shutdown();
		Thread.sleep(200);
	}

	@Test
	public void testDataTransferMultiPipes() throws Exception {
		PythonDMLScript.main(new String[]{"-python", "4004"});
		Thread.sleep(200);
		PythonDMLScript script = (PythonDMLScript) PythonDMLScript.GwS.getGateway().getEntryPoint();

		File in = folder.newFile("py2java-0");
		folder.newFile("java2py-0");
		File in2 = folder.newFile("py2java-1");
		folder.newFile("java2py-1");

		// Init Test
		BufferedOutputStream py2java = UnixPipeUtils.openOutput(in.getAbsolutePath(), 0);
		BufferedOutputStream py2java2 = UnixPipeUtils.openOutput(in2.getAbsolutePath(), 1);
		script.openPipes(folder.getRoot().getPath(), 2);

		// Read Test
		double[] data = new double[]{1.0, 2.0, 3.0};
		MatrixBlock mb = new MatrixBlock(3, 1, data);
		UnixPipeUtils.writeNumpyArrayInBatches(py2java, 0, 32, 3, Types.ValueType.FP64, mb);
		UnixPipeUtils.writeNumpyArrayInBatches(py2java2, 1, 32, 3, Types.ValueType.FP64, mb);
		MatrixBlock rcv_mb = script.startReadingMbFromPipes(new int[]{3,3}, 6, 1, Types.ValueType.FP64);
		data = new double[]{1.0, 2.0, 3.0, 1.0, 2.0, 3.0};
		assertArrayEquals(data, rcv_mb.getDenseBlockValues(), 1e-9);

		script.closePipes();

		PythonDMLScript.GwS.shutdown();
		Thread.sleep(200);
	}

	@Test(expected = DMLRuntimeException.class)
	public void testDataTransferNotInit1() throws Exception {
		PythonDMLScript.main(new String[]{"-python", "4005"});
		Thread.sleep(200);
		PythonDMLScript script = (PythonDMLScript) PythonDMLScript.GwS.getGateway().getEntryPoint();
		script.startReadingMbFromPipe(0, 2, 3, Types.ValueType.FP64);
	}

	@Test(expected = DMLRuntimeException.class)
	public void testDataTransferNotInit2() throws Exception {
		PythonDMLScript.main(new String[]{"-python", "4006"});
		Thread.sleep(200);
		PythonDMLScript script = (PythonDMLScript) PythonDMLScript.GwS.getGateway().getEntryPoint();
		script.startWritingMbToPipe(0, null);
	}

	@Test(expected = DMLRuntimeException.class)
	public void testDataTransferNotInit3() throws Exception {
		PythonDMLScript.main(new String[]{"-python", "4007"});
		Thread.sleep(200);
		PythonDMLScript script = (PythonDMLScript) PythonDMLScript.GwS.getGateway().getEntryPoint();
		script.startReadingMbFromPipes(new int[]{3,3}, 2, 3, Types.ValueType.FP64);
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

	@Test(expected = DMLRuntimeException.class)
	public void testDataTransferMaxValue1() throws Exception {
		PythonDMLScript.main(new String[]{"-python", "4008"});
		Thread.sleep(200);
		PythonDMLScript script = (PythonDMLScript) PythonDMLScript.GwS.getGateway().getEntryPoint();
		script.startReadingMbFromPipe(0, Integer.MAX_VALUE, 3, Types.ValueType.FP64);
	}

	@Test(expected = DMLRuntimeException.class)
	public void testDataTransferMaxValue2() throws Exception {
		PythonDMLScript.main(new String[]{"-python", "4009"});
		Thread.sleep(200);
		PythonDMLScript script = (PythonDMLScript) PythonDMLScript.GwS.getGateway().getEntryPoint();
		script.startReadingMbFromPipes(new int[]{3,3}, Integer.MAX_VALUE, 2, Types.ValueType.FP64);
	}
}
