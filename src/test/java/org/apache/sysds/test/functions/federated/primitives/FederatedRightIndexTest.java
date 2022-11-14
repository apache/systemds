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

import static org.junit.Assert.fail;

import java.util.Arrays;
import java.util.Collection;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
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
public class FederatedRightIndexTest extends AutomatedTestBase {
	private static final Log LOG = LogFactory.getLog(FederatedRightIndexTest.class.getName());

	private final static String TEST_NAME1 = "FederatedRightIndexRightTest";
	private final static String TEST_NAME2 = "FederatedRightIndexLeftTest";
	private final static String TEST_NAME3 = "FederatedRightIndexFullTest";
	private final static String TEST_NAME4 = "FederatedRightIndexFrameFullTest";

	private final static String TEST_DIR = "functions/federated/";
	private static final String TEST_CLASS_DIR = TEST_DIR + FederatedRightIndexTest.class.getSimpleName() + "/";

	private final static int blocksize = 1024;
	@Parameterized.Parameter()
	public int rows;
	@Parameterized.Parameter(1)
	public int cols;

	@Parameterized.Parameter(2)
	public int from;

	@Parameterized.Parameter(3)
	public int to;

	@Parameterized.Parameter(4)
	public boolean rowPartitioned;

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] {
			{20, 10, 1, 1, true}, //
			// {20, 10, 3, 5, true}, //
			// {10, 12, 1, 10, false} //
		});
	}

	private enum IndexType {
		RIGHT, LEFT, FULL
	}

	private enum DataType {
		MATRIX, FRAME
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"S"}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] {"S"}));
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] {"S"}));
		addTestConfiguration(TEST_NAME4, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4, new String[] {"S"}));
	}

	// @Test
	// public void testRightIndexRightDenseMatrixCP() {
	// runAggregateOperationTest(IndexType.RIGHT, DataType.MATRIX, ExecMode.SINGLE_NODE);
	// }

	// @Test
	// public void testRightIndexLeftDenseMatrixCP() {
	// runAggregateOperationTest(IndexType.LEFT, DataType.MATRIX, ExecMode.SINGLE_NODE);
	// }

	@Test
	public void testRightIndexFullDenseMatrixCP() {
		runAggregateOperationTest(IndexType.FULL, DataType.MATRIX, ExecMode.SINGLE_NODE);
	}

	@Test
	public void testRightIndexFullDenseFrameCP() {
		runAggregateOperationTest(IndexType.FULL, DataType.FRAME, ExecMode.SINGLE_NODE);
	}

	@Test
	public void testRightIndexFullDenseMatrixSP() {
		runAggregateOperationTest(IndexType.FULL, DataType.MATRIX, ExecMode.SPARK);
	}

	@Test
	public void testRightIndexFullDenseFrameSP() {
		runAggregateOperationTest(IndexType.FULL, DataType.FRAME, ExecMode.SPARK);
	}

	private void runAggregateOperationTest(IndexType indexType, DataType dataType, ExecMode execMode) {
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		ExecMode platformOld = rtplatform;

		if(rtplatform == ExecMode.SPARK)
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		String TEST_NAME = null;
		switch(indexType) {
			case RIGHT:
				from = from <= cols ? from : cols;
				to = to <= cols ? to : cols;
				TEST_NAME = TEST_NAME1;
				break;
			case LEFT:
				from = from <= rows ? from : rows;
				to = to <= rows ? to : rows;
				TEST_NAME = TEST_NAME2;
				break;
			case FULL:
				if(dataType == DataType.MATRIX)
					TEST_NAME = TEST_NAME3;
				else
					TEST_NAME = TEST_NAME4;
				from = from <= rows && from <= cols ? from : Math.min(rows, cols);
				to = to <= rows && to <= cols ? to : Math.min(rows, cols);
				break;
		}

		getAndLoadTestConfiguration(TEST_NAME);
		String HOME = SCRIPT_DIR + TEST_DIR;

		// write input matrices
		int r = rows;
		int c = cols / 4;
		if(rowPartitioned) {
			r = rows / 4;
			c = cols;
		}

		double[][] X1 = getRandomMatrix(r, c, 1, 5, 1, 3);
		double[][] X2 = getRandomMatrix(r, c, 1, 5, 1, 7);
		double[][] X3 = getRandomMatrix(r, c, 1, 5, 1, 8);
		double[][] X4 = getRandomMatrix(r, c, 1, 5, 1, 9);

		MatrixCharacteristics mc = new MatrixCharacteristics(r, c, blocksize, r * c);
		writeInputMatrixWithMTD("X1", X1, false, mc);
		writeInputMatrixWithMTD("X2", X2, false, mc);
		writeInputMatrixWithMTD("X3", X3, false, mc);
		writeInputMatrixWithMTD("X4", X4, false, mc);

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

		rtplatform = execMode;
		if(rtplatform == ExecMode.SPARK) 
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		
		TestConfiguration config = availableTestConfigurations.get(TEST_NAME);
		loadTestConfiguration(config);

		if(from > to) {
			from = to;
		}

		// Run reference dml script with normal matrix
		fullDMLScriptName = HOME + TEST_NAME + "Reference.dml";
		programArgs = new String[] {"-args", input("X1"), input("X2"), input("X3"), input("X4"), String.valueOf(from),
			String.valueOf(to), Boolean.toString(rowPartitioned).toUpperCase(), expected("S")};
		LOG.debug(runTest(null));
		// Run actual dml script with federated matrix

		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[] {"-stats", "100", "-nvargs",
			"in_X1=" + TestUtils.federatedAddress(port1, input("X1")),
			"in_X2=" + TestUtils.federatedAddress(port2, input("X2")),
			"in_X3=" + TestUtils.federatedAddress(port3, input("X3")),
			"in_X4=" + TestUtils.federatedAddress(port4, input("X4")), "rows=" + rows, "cols=" + cols, "from=" + from,
			"to=" + to, "rP=" + Boolean.toString(rowPartitioned).toUpperCase(), "out_S=" + output("S")};

		try{

			LOG.debug(runTest(null));
	
			// compare via files
			compareResults(1e-9, "Stat-DML1", "Stat-DML2");
	
			Assert.assertTrue(heavyHittersContainsString("fed_rightIndex"));
	
			// check that federated input files are still existing
			Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X1")));
			Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X2")));
			Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X3")));
			Assert.assertTrue(HDFSTool.existsFileOnHDFS(input("X4")));
	
			TestUtils.shutdownThreads(t1, t2, t3, t4);
	
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
		catch(Exception e){
			e.printStackTrace();
			fail(e.getMessage());
		}

	}
}
