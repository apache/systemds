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
package org.apache.sysds.test.functions.federated.io;

import java.io.File;
import java.io.IOException;
import java.net.ServerSocket;
import java.util.Arrays;
import java.util.Collection;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types;
import org.apache.sysds.conf.ConfigurationManager;
import org.apache.sysds.conf.DMLConfig;
import org.apache.sysds.runtime.DMLRuntimeException;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.functions.federated.FederatedTestObjectConstructor;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class FederatedTimeoutTest extends AutomatedTestBase {
	private static final Log LOG = LogFactory.getLog(FederatedTimeoutTest.class.getName());

	// This test use the same scripts as the Federated Reader tests, just with intended to fail connecting
	private final static String TEST_DIR = "functions/federated/io/";
	private final static String TEST_NAME = "FederatedReaderTest";
	private final static String TEST_CLASS_DIR = TEST_DIR + FederatedTimeoutTest.class.getSimpleName() + "/";
	private final static int blocksize = 1024;
	private final static File TEST_CONF_FILE = new File("src/test/config/SystemDS-config.xml");

	@Parameterized.Parameter()
	public int rows;
	@Parameterized.Parameter(1)
	public int cols;
	@Parameterized.Parameter(2)
	public boolean rowPartitioned;
	@Parameterized.Parameter(3)
	public int fedCount;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME));
	}

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		// number of rows or cols has to be >= number of federated locations.
		return Arrays.asList(new Object[][] {{10, 13, true, 2}});
	}

	@Test
	public void federatedSinglenodeRead() {
		Types.ExecMode execMode = Types.ExecMode.SINGLE_NODE;
		Types.ExecMode oldPlatform = setExecMode(execMode);

		// write input matrices
		int halfRows = rows / 2;
		long[][] begins = new long[][] {new long[] {0, 0}, new long[] {halfRows, 0}};
		long[][] ends = new long[][] {new long[] {halfRows, cols}, new long[] {rows, cols}};
		// We have two matrices handled by a single federated worker
		double[][] X1 = getRandomMatrix(halfRows, cols, 0, 1, 1, 42);
		double[][] X2 = getRandomMatrix(halfRows, cols, 0, 1, 1, 1340);
		writeInputMatrixWithMTD("X1", X1, false, new MatrixCharacteristics(halfRows, cols, blocksize, halfRows * cols));
		writeInputMatrixWithMTD("X2", X2, false, new MatrixCharacteristics(halfRows, cols, blocksize, halfRows * cols));
		// empty script name because we don't execute any script, just start the worker
		fullDMLScriptName = "";
		int port1 = getRandomAvailablePort();
		int port2 = getRandomAvailablePort();
		String host = "localhost";

		ServerSocket clientSocket = null;
		ServerSocket clientSocket2 = null;
		try {
			DMLConfig dmlconf = DMLConfig.readConfigurationFile(TEST_CONF_FILE.getPath());
			ConfigurationManager.setGlobalConfig(dmlconf);
			clientSocket = new ServerSocket(port1);
			clientSocket2 = new ServerSocket(port2);
			MatrixObject fed = FederatedTestObjectConstructor.constructFederatedInput(rows,
				cols,
				blocksize,
				host,
				begins,
				ends,
				new int[] {port1, port2},
				new String[] {input("X1"), input("X2")},
				input("X"));
			writeInputFederated("X", fed, null);

		}
		catch(DMLRuntimeException e) {
			LOG.info("Correctly timeout");
			// Great the test parsed.
		}
		catch(Exception e) {
			e.printStackTrace();
			Assert.assertTrue(false);
		}
		finally {
			resetExecMode(oldPlatform);
			if(clientSocket != null)
				try {
					clientSocket.close();
				}
				catch(IOException e) {
					e.printStackTrace();
				}
			if(clientSocket2 != null)
				try {
					clientSocket2.close();
				}
				catch(IOException e) {
					e.printStackTrace();
				}
		}

	}

}
