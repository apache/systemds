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
import org.tugraz.sysds.api.DMLScript;
import org.tugraz.sysds.runtime.meta.MatrixCharacteristics;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestConfiguration;
import org.tugraz.sysds.test.TestUtils;

import static java.lang.Thread.sleep;

public class FederatedConstructionTest extends AutomatedTestBase {
	
	private final static String TEST_DIR = "functions/federated/";
	private final static String TEST_NAME = "FederatedConstructionTest";
	private final static String TEST_CLASS_DIR = TEST_DIR + FederatedConstructionTest.class.getSimpleName() + "/";
	
	private final static int port = 1222;
	private final static int rows = 10;
	private final static int cols = 10;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, null));
	}
	
	@Test
	public void federatedConstruction() {
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		try {
			getAndLoadTestConfiguration(TEST_NAME);
			String HOME = SCRIPT_DIR + TEST_DIR;
			
			// empty script name because we don't execute any script, just start the worker
			fullDMLScriptName = "";
			programArgs = new String[]{"-w", Integer.toString(port)};
			
			Thread t = new Thread(() ->
					runTest(true, false, null, -1));
			t.start();
			sleep(1000);
			
			TestConfiguration config = availableTestConfigurations.get(TEST_NAME);
			loadTestConfiguration(config);
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			double[][] A = getRandomMatrix(rows, cols, -1, 1, 1, System.currentTimeMillis());
			writeInputMatrixWithMTD("A", A, false, new MatrixCharacteristics(rows, cols, 1000, 1000));
			programArgs = new String[]{"-explain", "-args", "\"localhost:" + port + "/" + input("A") + "\"",
					Integer.toString(rows), Integer.toString(cols), Integer.toString(rows * 2)};
			
			runTest(true, false, null, -1);
			// kill the worker
			t.interrupt();
		}
		catch (InterruptedException e) {
			e.printStackTrace();
		}
		finally {
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}
