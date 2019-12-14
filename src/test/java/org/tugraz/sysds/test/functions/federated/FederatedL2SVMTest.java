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
import org.tugraz.sysds.common.Types;
import org.tugraz.sysds.runtime.matrix.data.InputInfo;
import org.tugraz.sysds.runtime.matrix.data.MatrixBlock;
import org.tugraz.sysds.runtime.meta.MatrixCharacteristics;
import org.tugraz.sysds.runtime.util.DataConverter;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestConfiguration;
import org.tugraz.sysds.test.TestUtils;

import java.io.IOException;

import static java.lang.Thread.sleep;

public class FederatedL2SVMTest extends AutomatedTestBase {
	
	private final static String TEST_DIR = "functions/federated/";
	private final static String TEST_NAME = "FederatedL2SVMTest";
	private final static String TEST_CLASS_DIR = TEST_DIR + FederatedL2SVMTest.class.getSimpleName() + "/";
	
	private final static int blocksize = 1000;
	private final static int port1 = 1222;
	private final static int port2 = 1223;
	private final static int rows = 1000;
	private final static int cols = 100;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, null));
	}
	
	@Test
	public void federatedL2SVM() throws IOException {
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		Types.ExecMode platformOld = rtplatform;
		rtplatform = Types.ExecMode.SINGLE_NODE;
		try {
			getAndLoadTestConfiguration(TEST_NAME);
			String HOME = SCRIPT_DIR + TEST_DIR;
			
			// empty script name because we don't execute any script, just start the worker
			fullDMLScriptName = "";
			programArgs = new String[]{"-w", Integer.toString(port1)};
			Thread t1 = new Thread(() ->
					runTest(true, false, null, -1));
			t1.start();
			sleep(1000);
			fullDMLScriptName = "";
			programArgs = new String[]{"-w", Integer.toString(port2)};
			Thread t2 = new Thread(() ->
					runTest(true, false, null, -1));
			t2.start();
			
			TestConfiguration config = availableTestConfigurations.get(TEST_NAME);
			loadTestConfiguration(config);
			
			int halfRows = rows / 2;
			// We have two matrices handled by a single federated worker
			double[][] X1 = getRandomMatrix(halfRows, cols, 0, 1, 1, 42);
			double[][] X2 = getRandomMatrix(halfRows, cols, 0, 1, 1, 1337);
			double[][] Y = getRandomMatrix(rows, 1, -1, 1, 1, 1234);
			for (int i = 0; i < rows; i++)
				Y[i][0] = (Y[i][0] > 0) ? 1 : -1;
			
			writeInputMatrixWithMTD("X1", X1, false, new MatrixCharacteristics(halfRows, cols, 1000, halfRows * cols));
			writeInputMatrixWithMTD("X2", X2, false, new MatrixCharacteristics(halfRows, cols, 1000, halfRows * cols));
			writeInputMatrixWithMTD("Y", Y, false, new MatrixCharacteristics(rows, 1, 1000, rows));
			
			fullDMLScriptName = HOME + TEST_NAME + "Reference.dml";
			programArgs = new String[]{"-args", input("X1"), input("X2"), input("Y"), output("Z2")};
			runTest(true, false, null, -1);
			double[][] expected = readMatrix(output("Z2"), InputInfo.BinaryBlockInputInfo, cols, 1, blocksize);
			
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain", "-args", "\"localhost:" + port1 + "/" + input("X1") + "\"",
					"\"localhost:" + port2 + "/" + input("X2") + "\"", Integer.toString(rows), Integer.toString(cols),
					Integer.toString(halfRows), input("Y"), output("Z1")};
			runTest(true, false, null, -1);
			double[][] actual = readMatrix(output("Z1"), InputInfo.BinaryBlockInputInfo, cols, 1, blocksize);
			
			// epsilon because aggregation works slightly differently for federated
			TestUtils.compareMatrices(expected, actual, cols, 1, 1e-6);
			// kill the worker
			t1.interrupt();
			t2.interrupt();
		}
		catch (InterruptedException e) {
			e.printStackTrace();
			assert(false);
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
	
	private static double[][] readMatrix(String fname, InputInfo ii, long rows, long cols, int blocksize)
			throws IOException {
		MatrixBlock mb = DataConverter.readMatrixFromHDFS(fname, ii, rows, cols, blocksize, blocksize);
		return DataConverter.convertToDoubleMatrix(mb);
	}
}
