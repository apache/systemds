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

package org.apache.sysds.test.functions.federated.primitives;

import java.util.Arrays;
import java.util.Collection;

import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.util.HDFSTool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class FederatedQuantileTest extends AutomatedTestBase {

	private final static String TEST_DIR = "functions/federated/quantile/";
	private final static String TEST_NAME1 = "FederatedQuantileTest";
	private final static String TEST_NAME2 = "FederatedMedianTest";
	private final static String TEST_NAME3 = "FederatedIQMTest";
	private final static String TEST_NAME4 = "FederatedQuantilesTest";
	private final static String TEST_CLASS_DIR = TEST_DIR + FederatedQuantileTest.class.getSimpleName() + "/";

	private final static int blocksize = 1024;
	@Parameterized.Parameter()
	public int rows;

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] {{1000}});
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"S.scalar"}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] {"S.scalar"}));
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] {"S.scalar"}));
		addTestConfiguration(TEST_NAME4, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4, new String[] {"S"}));
	}

	@Test
	public void federatedQuantile1CP() { federatedQuartile(Types.ExecMode.SINGLE_NODE, TEST_NAME1, 0.25); }

	@Test
	public void federatedQuantile2CP() { federatedQuartile(Types.ExecMode.SINGLE_NODE, TEST_NAME1, 0.5); }

	@Test
	public void federatedQuantile3CP() { federatedQuartile(Types.ExecMode.SINGLE_NODE, TEST_NAME1, 0.75); }

	@Test
	public void federatedMedianCP() { federatedQuartile(Types.ExecMode.SINGLE_NODE, TEST_NAME2, -1); }

	@Test
	public void federatedIQMCP() { federatedQuartile(Types.ExecMode.SINGLE_NODE, TEST_NAME1, -1); }

	@Test
	public void federatedQuantilesCP() { federatedQuartile(Types.ExecMode.SINGLE_NODE, TEST_NAME1, -1); }

	@Test
//	@Ignore
	public void federatedQuantile1SP() { federatedQuartile(Types.ExecMode.SPARK, TEST_NAME1, 0.25); }

	@Test
//	@Ignore
	public void federatedQuantile2SP() { federatedQuartile(Types.ExecMode.SPARK, TEST_NAME1, 0.5); }

	@Test
//	@Ignore
	public void federatedQuantile3SP() { federatedQuartile(Types.ExecMode.SPARK, TEST_NAME1, 0.75); }

	@Test
//	@Ignore
	public void federatedMedianSP() { federatedQuartile(Types.ExecMode.SPARK, TEST_NAME2, -1); }

	@Test
//	@Ignore
	public void federatedIQMSP() { federatedQuartile(Types.ExecMode.SPARK, TEST_NAME1, -1); }

	@Test
//	@Ignore
	public void federatedQuantilesSP() { federatedQuartile(Types.ExecMode.SPARK, TEST_NAME1, -1); }





	public void federatedQuartile(Types.ExecMode execMode, String TEST_NAME, double p) {
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		Types.ExecMode platformOld = rtplatform;

		getAndLoadTestConfiguration(TEST_NAME);
		String HOME = SCRIPT_DIR + TEST_DIR;

		double[][] X1 = getRandomMatrix(rows, 1, 1, 5, 1, 3);

		MatrixCharacteristics mc = new MatrixCharacteristics(rows, 1, blocksize, rows);
		writeInputMatrixWithMTD("X1", X1, false, mc);

		// empty script name because we don't execute any script, just start the worker
		fullDMLScriptName = "";
		int port1 = getRandomAvailablePort();
		Thread t1 = startLocalFedWorkerThread(port1);

		// we need the reference file to not be written to hdfs, so we get the correct format
		rtplatform = Types.ExecMode.SINGLE_NODE;
		// Run reference dml script with normal matrix for Row/Col
		fullDMLScriptName = HOME + TEST_NAME + "Reference.dml";
		programArgs = new String[] {"-explain", "-stats", "100", "-args", input("X1"), expected("S"), String.valueOf(p)};
		runTest(true, false, null, -1);

		// reference file should not be written to hdfs, so we set platform here
		rtplatform = execMode;
		if(rtplatform == Types.ExecMode.SPARK) {
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		}
		TestConfiguration config = availableTestConfigurations.get(TEST_NAME);
		loadTestConfiguration(config);

		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[] {"-explain", "-stats", "100", "-nvargs",
			"in_X1=" + TestUtils.federatedAddress(port1, input("X1")),
			"rows=" + rows, "cols=" + 1, "p=" + String.valueOf(p),
			"out_S=" + output("S")
		};
		runTest(true, false, null, -1);

		// compare all sums via files
		compareResults(1e-9);
		Assert.assertTrue(heavyHittersContainsString("fed_qsort"));
		Assert.assertTrue(heavyHittersContainsString("fed_qpick"));

		// check that federated input files are still existing
		Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X1")));

		TestUtils.shutdownThreads(t1);
		rtplatform = platformOld;
		DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
	}
}
