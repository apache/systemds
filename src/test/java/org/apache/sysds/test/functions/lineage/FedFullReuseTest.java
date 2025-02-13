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

package org.apache.sysds.test.functions.lineage;

import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import java.util.Collection;

import org.apache.sysds.common.Opcodes;
import org.apache.sysds.runtime.lineage.Lineage;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class FedFullReuseTest extends AutomatedTestBase {

	private final static String TEST_DIR = "functions/lineage/";
	private final static String TEST_NAME1 = "FedFullReuse1";
	private final static String TEST_NAME2 = "FedFullReuse2";
	private final static String TEST_CLASS_DIR = TEST_DIR + FedFullReuseTest.class.getSimpleName() + "/";

	private final static int blocksize = 1024;
	@Parameterized.Parameter()
	public int rows;
	@Parameterized.Parameter(1)
	public int cols;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"Z"}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] {"Z"}));
	}

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		// rows have to be even and > 1
		return Arrays.asList(new Object[][] {
			// {2, 1000}, {10, 100},
			{100, 10}, 
			//{1000, 1},
			// {10, 2000}, {2000, 10}
		});
	}

	@Test
	public void federatedOutputReuse() {
		//don't cache federated outputs in the coordinator
		//reuse inside federated workers
		federatedReuse(TEST_NAME1);
	}

	@Test
	public void nonfederatedOutputReuse() {
		//cache non-federated outputs in the coordinator
		federatedReuse(TEST_NAME2);
	}
	
	public void federatedReuse(String test) {
		getAndLoadTestConfiguration(test);
		String HOME = SCRIPT_DIR + TEST_DIR;

		// write input matrices
		int halfRows = rows / 2;
		// Share two matrices between two federated worker
		double[][] X1 = getRandomMatrix(halfRows, cols, 0, 1, 1, 42);
		double[][] X2 = getRandomMatrix(halfRows, cols, 0, 1, 1, 1340);
		double[][] Y1 = getRandomMatrix(cols, halfRows, 0, 1, 1, 44);
		double[][] Y2 = getRandomMatrix(cols, halfRows, 0, 1, 1, 21);

		writeInputMatrixWithMTD("X1", X1, false, new MatrixCharacteristics(halfRows, cols, blocksize, halfRows * cols));
		writeInputMatrixWithMTD("X2", X2, false, new MatrixCharacteristics(halfRows, cols, blocksize, halfRows * cols));
		writeInputMatrixWithMTD("Y1", Y1, false, new MatrixCharacteristics(cols, halfRows, blocksize, halfRows * cols));
		writeInputMatrixWithMTD("Y2", Y2, false, new MatrixCharacteristics(cols, halfRows, blocksize, halfRows * cols));

		int port1 = getRandomAvailablePort();
		int port2 = getRandomAvailablePort();
		String[] otherargs = new String[] {"-lineage", "reuse_full"};
		Lineage.resetInternalState();
		Thread t1 = startLocalFedWorkerThread(port1, otherargs, FED_WORKER_WAIT_S);
		Thread t2 = startLocalFedWorkerThread(port2, otherargs);

		TestConfiguration config = availableTestConfigurations.get(test);
		loadTestConfiguration(config);

		// Run reference dml script with normal matrix. Reuse of ba+*.
		fullDMLScriptName = HOME + test + "Reference.dml";
		programArgs = new String[] {"-stats", "-lineage", "reuse_full",
			"-nvargs", "X1=" + input("X1"), "X2=" + input("X2"), "Y1=" + input("Y1"),
			"Y2=" + input("Y2"), "Z=" + expected("Z")};
		runTest(true, false, null, -1);
		long mmCount = Statistics.getCPHeavyHitterCount(Opcodes.MMULT.toString());

		// Run actual dml script with federated matrix
		// The fed workers reuse ba+*
		fullDMLScriptName = HOME + test + ".dml";
		programArgs = new String[] {"-stats","-lineage", "reuse_full",
			"-nvargs", "X1=" + TestUtils.federatedAddress(port1, input("X1")),
			"X2=" + TestUtils.federatedAddress(port2, input("X2")),
			"Y1=" + TestUtils.federatedAddress(port1, input("Y1")),
			"Y2=" + TestUtils.federatedAddress(port2, input("Y2")), "r=" + rows, "c=" + cols, "Z=" + output("Z")};
		runTest(true, false, null, -1);
		long mmCount_fed = Statistics.getCPHeavyHitterCount(Opcodes.MMULT.toString());
		long fedMMCount = Statistics.getCPHeavyHitterCount("fed_ba+*");

		// compare results 
		compareResults(1e-9);
		// compare matrix multiplication count
		// #federated execution of ba+* = #threads times #non-federated execution of ba+* (after reuse) 
		Assert.assertTrue("Violated reuse count: "+mmCount_fed+" == "+mmCount*2, 
				mmCount_fed == mmCount * 2); // #threads = 2
		switch(test) {
			case TEST_NAME1:
				// If the o/p is federated, fed_ba+* will be called everytime
				// but the workers should be able to reuse ba+*
				assertTrue(fedMMCount > mmCount_fed);
				break;
			case TEST_NAME2:
				// If the o/p is non-federated, fed_ba+* will be called once
				// and each worker will call ba+* once.
				assertTrue(fedMMCount < mmCount_fed);
				break;
		}


		TestUtils.shutdownThreads(t1, t2);
	}

}
