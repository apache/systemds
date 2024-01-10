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

package org.apache.sysds.test.functions.federated.primitives.part3;

import java.util.Arrays;
import java.util.Collection;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class FederatedSplitTest extends AutomatedTestBase {

	private static final Log LOG = LogFactory.getLog(FederatedSplitTest.class.getName());
	private final static String TEST_DIR = "functions/federated/";
	private final static String TEST_NAME = "FederatedSplitTest";
	private final static String TEST_CLASS_DIR = TEST_DIR + FederatedSplitTest.class.getSimpleName() + "/";

	private final static int blocksize = 1024;
	@Parameterized.Parameter()
	public int rows;
	@Parameterized.Parameter(1)
	public int cols;
	@Parameterized.Parameter(2)
	public String cont;

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] {
			// {152, 12, "TRUE"},
			{132, 11, "FALSE"}});
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"Z"}));
	}

	@Test
	public void federatedSplitCP() {
		federatedSplit(Types.ExecMode.SINGLE_NODE);
	}

	@Test
	public void federatedSplitSP() {
		federatedSplit(Types.ExecMode.SPARK);
	}

	public void federatedSplit(Types.ExecMode execMode) {
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		Types.ExecMode platformOld = rtplatform;
		rtplatform = execMode;
		if(rtplatform == Types.ExecMode.SPARK) {
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		}

		getAndLoadTestConfiguration(TEST_NAME);
		String HOME = SCRIPT_DIR + TEST_DIR;

		// write input matrices
		int halfRows = rows / 2;
		// We have two matrices handled by a single federated worker
		double[][] X1 = getRandomMatrix(halfRows, cols, 0, 1, 1, 42);
		double[][] X2 = getRandomMatrix(halfRows, cols, 0, 1, 1, 1340);
		// And another two matrices handled by a single federated worker
		double[][] Y1 = getRandomMatrix(halfRows, cols, 0, 1, 1, 44);
		double[][] Y2 = getRandomMatrix(halfRows, cols, 0, 1, 1, 21);

		writeInputMatrixWithMTD("X1", X1, false, new MatrixCharacteristics(halfRows, cols, blocksize, halfRows * cols));
		writeInputMatrixWithMTD("X2", X2, false, new MatrixCharacteristics(halfRows, cols, blocksize, halfRows * cols));
		writeInputMatrixWithMTD("Y1", Y1, false, new MatrixCharacteristics(halfRows, cols, blocksize, halfRows * cols));
		writeInputMatrixWithMTD("Y2", Y2, false, new MatrixCharacteristics(halfRows, cols, blocksize, halfRows * cols));

		TestConfiguration config = availableTestConfigurations.get(TEST_NAME);
		loadTestConfiguration(config);
		setOutputBuffering(true); // otherwise NPE

		int port1 = getRandomAvailablePort();
		int port2 = getRandomAvailablePort();
		Process t1 = startLocalFedWorker(port1, FED_WORKER_WAIT_S);
		Process t2 = startLocalFedWorker(port2);

		try {
			if(!isAlive(t1, t2))
				throw new RuntimeException("Failed starting federated worker");

			// Run reference dml script with normal matrix
			fullDMLScriptName = HOME + TEST_NAME + "Reference.dml";
			programArgs = new String[] {"-nvargs", "X1=" + input("X1"), "X2=" + input("X2"), "Y1=" + input("Y1"),
				"Y2=" + input("Y2"), "Z=" + expected("Z"), "Cont=" + cont};
			String out = runTest(null).toString();

			// Run actual dml script with federated matrix
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-stats", "100", "-nvargs", "X1=" + TestUtils.federatedAddress(port1, input("X1")),
				"X2=" + TestUtils.federatedAddress(port2, input("X2")),
				"Y1=" + TestUtils.federatedAddress(port1, input("Y1")),
				"Y2=" + TestUtils.federatedAddress(port2, input("Y2")), "r=" + rows, "c=" + cols, "Z=" + output("Z"),
				"Cont=" + cont};
			String fedOut = runTest(null).toString();

			LOG.debug(out);
			LOG.debug(fedOut);
			// compare via files
			compareResults(1e-9, "Stat-DML1", "Stat-DML2");

			if(cont.equals("TRUE"))
				Assert.assertTrue(heavyHittersContainsString("fed_rightIndex"));
			else if(execMode != Types.ExecMode.SPARK) {
				Assert.assertTrue(heavyHittersContainsString("fed_rmempty"));
			}

		}
		finally {
			TestUtils.shutdownThreads(t1, t2);
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}
