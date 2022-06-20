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

import org.apache.sysds.common.Types.ExecMode;
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
public class FederatedLeftIndexTest extends AutomatedTestBase {

	private final static String TEST_NAME1 = "FederatedLeftIndexFullTest";
	private final static String TEST_NAME2 = "FederatedLeftIndexFrameFullTest";
	private final static String TEST_NAME3 = "FederatedLeftIndexScalarTest";

	private final static String TEST_DIR = "functions/federated/";
	private static final String TEST_CLASS_DIR = TEST_DIR + FederatedLeftIndexTest.class.getSimpleName() + "/";

	private final static int blocksize = 1024;
	@Parameterized.Parameter()
	public int rows1;
	@Parameterized.Parameter(1)
	public int cols1;

	@Parameterized.Parameter(2)
	public int rows2;
	@Parameterized.Parameter(3)
	public int cols2;

	@Parameterized.Parameter(4)
	public int from;
	@Parameterized.Parameter(5)
	public int to;

	@Parameterized.Parameter(6)
	public int from2;
	@Parameterized.Parameter(7)
	public int to2;

	@Parameterized.Parameter(8)
	public boolean rowPartitioned;

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] {
			{8, 2, 8, 1, 1, 8, 1, 1, true},
			{24, 12, 20, 8, 3, 22, 1, 8, true},
			{24, 12, 10, 8, 7, 16, 1, 8, true},
			{24, 12, 20, 11, 3, 22, 1, 11, false},
			{24, 12, 20, 8, 3, 22, 1, 8, false},
			{24, 12, 20, 8, 3, 22, 5, 12, false},
		});
	}

	private enum DataType {
		MATRIX, FRAME, SCALAR
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"S"}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] {"S"}));
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] {"S"}));
	}

	@Test
	public void testLeftIndexFullDenseMatrixCP() {
		runAggregateOperationTest(DataType.MATRIX, ExecMode.SINGLE_NODE);
	}

	@Test
	public void testLeftIndexFullDenseFrameCP() {
		runAggregateOperationTest(DataType.FRAME, ExecMode.SINGLE_NODE);
	}

	@Test
	public void testLeftIndexFullDenseMatrixSP() {
		runAggregateOperationTest(DataType.MATRIX, ExecMode.SPARK);
	}

	@Test
	public void testLeftIndexFullDenseFrameSP() {
		runAggregateOperationTest(DataType.FRAME, ExecMode.SPARK);
	}

	@Test
	public void testLeftIndexScalarCP() {
		runAggregateOperationTest(DataType.SCALAR, ExecMode.SINGLE_NODE);
	}

	@Test
	public void testLeftIndexScalarSP() {
		runAggregateOperationTest(DataType.SCALAR, ExecMode.SPARK);
	}

	private void runAggregateOperationTest(DataType dataType, ExecMode execMode) {
		ExecMode oldPlatform = setExecMode(execMode);
		
		try {
			String TEST_NAME = null;
	
			if(dataType == DataType.MATRIX)
				TEST_NAME = TEST_NAME1;
			else if(dataType == DataType.FRAME)
				TEST_NAME = TEST_NAME2;
			else
				TEST_NAME = TEST_NAME3;
	
			getAndLoadTestConfiguration(TEST_NAME);
			String HOME = SCRIPT_DIR + TEST_DIR;
	
			// write input matrices
			int r1 = rows1;
			int c1 = cols1 / 4;
			if(rowPartitioned) {
				r1 = rows1 / 4;
				c1 = cols1;
			}
	
			double[][] X1 = getRandomMatrix(r1, c1, 1, 5, 1, 3);
			double[][] X2 = getRandomMatrix(r1, c1, 1, 5, 1, 7);
			double[][] X3 = getRandomMatrix(r1, c1,  1, 5, 1, 8);
			double[][] X4 = getRandomMatrix(r1, c1, 1, 5, 1, 9);
	
			MatrixCharacteristics mc = new MatrixCharacteristics(r1, c1,  blocksize, r1 * c1);
			writeInputMatrixWithMTD("X1", X1, false, mc);
			writeInputMatrixWithMTD("X2", X2, false, mc);
			writeInputMatrixWithMTD("X3", X3, false, mc);
			writeInputMatrixWithMTD("X4", X4, false, mc);
	
			if(dataType != DataType.SCALAR) {
				double[][] Y = getRandomMatrix(rows2, cols2, 1, 5, 1, 3);
	
				MatrixCharacteristics mc2 = new MatrixCharacteristics(rows2, cols2, blocksize, rows2 * cols2);
				writeInputMatrixWithMTD("Y", Y, false, mc2);
			}
	
			// empty script name because we don't execute any script, just start the worker
			fullDMLScriptName = "";
			int port1 = getRandomAvailablePort();
			int port2 = getRandomAvailablePort();
			int port3 = getRandomAvailablePort();
			int port4 = getRandomAvailablePort();
			Thread t1 = startLocalFedWorkerThread(port1, FED_WORKER_WAIT_S);
			Thread t2 = startLocalFedWorkerThread(port2, FED_WORKER_WAIT_S);
			Thread t3 = startLocalFedWorkerThread(port3, FED_WORKER_WAIT_S);
			Thread t4 = startLocalFedWorkerThread(port4);
	
			TestConfiguration config = availableTestConfigurations.get(TEST_NAME);
			loadTestConfiguration(config);
	
			var lfrom = Math.min(from, to);
			var lfrom2 = Math.min(from2, to2);
	
			// Run reference dml script with normal matrix
			fullDMLScriptName = HOME + TEST_NAME + "Reference.dml";
			programArgs = new String[] {"-args", input("X1"), input("X2"), input("X3"), input("X4"),
				input("Y"), String.valueOf(lfrom), String.valueOf(to),
				String.valueOf(lfrom2), String.valueOf(to2),
				Boolean.toString(rowPartitioned).toUpperCase(), expected("S")};
			runTest(null);
			// Run actual dml script with federated matrix
	
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-stats", "100", "-nvargs",
				"in_X1=" + TestUtils.federatedAddress(port1, input("X1")),
				"in_X2=" + TestUtils.federatedAddress(port2, input("X2")),
				"in_X3=" + TestUtils.federatedAddress(port3, input("X3")),
				"in_X4=" + TestUtils.federatedAddress(port4, input("X4")),
				"in_Y=" + input("Y"), "rows=" + rows1, "cols=" + cols1,
				"rows2=" + rows2, "cols2=" + cols2,
				"from=" + from, "to=" + to,"from2=" + from2, "to2=" + to2,
				"rP=" + Boolean.toString(rowPartitioned).toUpperCase(), "out_S=" + output("S")};
	
			runTest(null);
	
			// compare via files
			compareResults(1e-9, "Stat-DML1", "Stat-DML2");
	
			Assert.assertTrue(rtplatform ==ExecMode.SPARK ?
				heavyHittersContainsString("fed_mapLeftIndex") : heavyHittersContainsString("fed_leftIndex"));
	
			// check that federated input files are still existing
			Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X1")));
			Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X2")));
			Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X3")));
			Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X4")));
	
			TestUtils.shutdownThreads(t1, t2, t3, t4);
		}
		finally {
			resetExecMode(oldPlatform);
		}
	}
}
