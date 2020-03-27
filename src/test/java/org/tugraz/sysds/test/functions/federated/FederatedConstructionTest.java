/*
 * Copyright 2019 Graz University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package org.tugraz.sysds.test.functions.federated;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.tugraz.sysds.api.DMLScript;
import org.tugraz.sysds.common.Types;
import org.tugraz.sysds.runtime.meta.MatrixCharacteristics;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestConfiguration;
import org.tugraz.sysds.test.TestUtils;

import java.util.Arrays;
import java.util.Collection;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class FederatedConstructionTest extends AutomatedTestBase {

	private final static String TEST_DIR = "functions/federated/";
	private final static String TEST_NAME = "FederatedConstructionTest";
	private final static String TEST_CLASS_DIR = TEST_DIR + FederatedConstructionTest.class.getSimpleName() + "/";

	private int blocksize = 1024;
	private int rows, cols;

	public FederatedConstructionTest(int rows, int cols) {

		this.rows = rows;
		this.cols = cols;
	}

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		Object[][] data = new Object[][] {{1, 1000}, {10, 100}, {100, 10}, {1000, 1}, {10, 2000}, {2000, 10}};
		return Arrays.asList(data);
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"B"}));
	}

	@Test
	public void federatedConstructionCP() {
		federatedConstruction(Types.ExecMode.SINGLE_NODE);
	}

	@Test
	public void federatedConstructionSP() {
		federatedConstruction(Types.ExecMode.SPARK);
	}

	public void federatedConstruction(Types.ExecMode execMode) {
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		Types.ExecMode platformOld = rtplatform;

		Thread t = null;

		getAndLoadTestConfiguration(TEST_NAME);
		String HOME = SCRIPT_DIR + TEST_DIR;

		// write input matrix
		double[][] A = getRandomMatrix(rows, cols, -1, 1, 1, 1234);
		writeInputMatrixWithMTD("A", A, false, new MatrixCharacteristics(rows, cols, blocksize, rows * cols));

		int port = getRandomAvailablePort();
		t = startLocalFedWorker(port);

		TestConfiguration config = availableTestConfigurations.get(TEST_NAME);
		loadTestConfiguration(config);

		// we need the reference file to not be written to hdfs, so we get the correct format
		rtplatform = Types.ExecMode.SINGLE_NODE;
		// Run reference dml script with normal matrix
		fullDMLScriptName = HOME + TEST_NAME + "Reference.dml";
		programArgs = new String[] {"-args", input("A"), expected("B")};
		runTest(true, false, null, -1);

		// reference file should not be written to hdfs
		rtplatform = execMode;
		if(rtplatform == Types.ExecMode.SPARK) {
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		}
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[] {"-args", "\"localhost:" + port + "/" + input("A") + "\"", Integer.toString(rows),
			Integer.toString(cols), Integer.toString(rows * 2), output("B")};

		runTest(true, false, null, -1);
		// compare via files
		compareResults(1e-12);

		TestUtils.shutdownThread(t);
		rtplatform = platformOld;
		DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
	}
}
