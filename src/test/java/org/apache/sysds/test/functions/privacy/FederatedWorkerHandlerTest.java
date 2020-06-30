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

package org.apache.sysds.test.functions.privacy;

import java.util.Arrays;

import org.apache.sysds.api.DMLException;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.runtime.privacy.PrivacyConstraint;
import org.apache.sysds.runtime.privacy.PrivacyConstraint.PrivacyLevel;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.apache.sysds.common.Types;
import static java.lang.Thread.sleep;

@net.jcip.annotations.NotThreadSafe
public class FederatedWorkerHandlerTest extends AutomatedTestBase {

	private static final String TEST_DIR = "functions/federated/";
	private static final String TEST_DIR_SCALAR = TEST_DIR + "matrix_scalar/";
	private final static String TEST_CLASS_DIR = TEST_DIR + FederatedWorkerHandlerTest.class.getSimpleName() + "/";
	private final static String TEST_CLASS_DIR_SCALAR = TEST_DIR_SCALAR + FederatedWorkerHandlerTest.class.getSimpleName() + "/";
	private static final String TEST_PROG_SCALAR_ADDITION_MATRIX = "ScalarAdditionFederatedMatrix";
	private final static String AGGREGATION_TEST_NAME = "FederatedSumTest";
	private final static String TRANSFER_TEST_NAME = "FederatedRCBindTest";
	private final static String MATVECMULT_TEST_NAME = "FederatedMultiplyTest";
	private static final String FEDERATED_WORKER_HOST = "localhost";
	private static final int FEDERATED_WORKER_PORT = 1222;

	private final static int blocksize = 1024;
	private int rows = 10;
	private int cols = 10;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration("scalar", new TestConfiguration(TEST_CLASS_DIR_SCALAR, TEST_PROG_SCALAR_ADDITION_MATRIX, new String [] {"R"}));
		addTestConfiguration("aggregation", new TestConfiguration(TEST_CLASS_DIR, AGGREGATION_TEST_NAME, new String[] {"S.scalar", "R", "C"}));
		addTestConfiguration("transfer", new TestConfiguration(TEST_CLASS_DIR, TRANSFER_TEST_NAME, new String[] {"R", "C"}));
		addTestConfiguration("matvecmult", new TestConfiguration(TEST_CLASS_DIR, MATVECMULT_TEST_NAME, new String[] {"Z"}));
	}

	@Test
	public void scalarPrivateTest(){
		scalarTest(PrivacyLevel.Private, DMLException.class);
	}

	@Test
	public void scalarPrivateAggregationTest(){
		scalarTest(PrivacyLevel.PrivateAggregation, DMLException.class);
	}

	@Test
	public void scalarNonePrivateTest(){
		scalarTest(PrivacyLevel.None, null);
	}

	private void scalarTest(PrivacyLevel privacyLevel, Class<?> expectedException){
		getAndLoadTestConfiguration("scalar");

		double[][] m = getRandomMatrix(this.rows, this.cols, -1, 1, 1.0, 1);
		
		PrivacyConstraint pc = new PrivacyConstraint(privacyLevel);
		writeInputMatrixWithMTD("M", m, false, new MatrixCharacteristics(rows, cols, blocksize, rows * cols), pc);

		int s = TestUtils.getRandomInt();
		double[][] r = new double[rows][cols];
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				r[i][j] = m[i][j] + s;
			}
		}
		if (expectedException == null)
			writeExpectedMatrix("R", r);

		runGenericScalarTest(TEST_PROG_SCALAR_ADDITION_MATRIX, s, expectedException, privacyLevel);
	}


	private void runGenericScalarTest(String dmlFile, int s, Class<?> expectedException, PrivacyLevel privacyLevel)
	{
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		Types.ExecMode platformOld = rtplatform;

		Thread t = null;
		try {
			// we need the reference file to not be written to hdfs, so we get the correct format
			rtplatform = Types.ExecMode.SINGLE_NODE;
			if (rtplatform == Types.ExecMode.SPARK) {
				DMLScript.USE_LOCAL_SPARK_CONFIG = true;
			}
			programArgs = new String[] {"-w", Integer.toString(FEDERATED_WORKER_PORT)};
			t = new Thread(() -> runTest(true, false, null, -1));
			t.start();
			sleep(FED_WORKER_WAIT);
			fullDMLScriptName = SCRIPT_DIR + TEST_DIR_SCALAR + dmlFile + ".dml";
			programArgs = new String[]{"-checkPrivacy", "-args",
					TestUtils.federatedAddress(FEDERATED_WORKER_HOST, FEDERATED_WORKER_PORT, input("M")),
					Integer.toString(rows), Integer.toString(cols),
					Integer.toString(s),
					output("R")};
			boolean exceptionExpected = (expectedException != null);
			runTest(true, exceptionExpected, expectedException, -1);

			if ( !exceptionExpected )
				compareResults();
		} catch (InterruptedException e) {
			e.printStackTrace();
			assert (false);
		} finally {
			assert(checkedPrivacyConstraintsContains(privacyLevel));
			rtplatform = platformOld;
			TestUtils.shutdownThread(t);
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}

	@Test
	public void aggregatePrivateTest() {
		federatedSum(Types.ExecMode.SINGLE_NODE, PrivacyLevel.Private, DMLException.class);
	}

	@Test
	public void aggregatePrivateAggregationTest() {
		federatedSum(Types.ExecMode.SINGLE_NODE, PrivacyLevel.PrivateAggregation, null);
	}

	@Test
	public void aggregateNonePrivateTest() {
		federatedSum(Types.ExecMode.SINGLE_NODE, PrivacyLevel.None, null);
	}

	public void federatedSum(Types.ExecMode execMode, PrivacyLevel privacyLevel, Class<?> expectedException) {
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		Types.ExecMode platformOld = rtplatform;

		Thread t = null;

		getAndLoadTestConfiguration("aggregation");
		String HOME = SCRIPT_DIR + TEST_DIR;

		double[][] A = getRandomMatrix(rows, cols, -10, 10, 1, 1);
		writeInputMatrixWithMTD("A", A, false, new MatrixCharacteristics(rows, cols, blocksize, rows * cols), new PrivacyConstraint(privacyLevel));
		int port = getRandomAvailablePort();
		t = startLocalFedWorker(port);

		// we need the reference file to not be written to hdfs, so we get the correct format
		rtplatform = Types.ExecMode.SINGLE_NODE;
		// Run reference dml script with normal matrix for Row/Col sum
		fullDMLScriptName = HOME + AGGREGATION_TEST_NAME + "Reference.dml";
		programArgs = new String[] {"-args", input("A"), input("A"), expected("R"), expected("C")};
		runTest(true, false, null, -1);

		// write expected sum
		double sum = 0;
		for(double[] doubles : A) {
			sum += Arrays.stream(doubles).sum();
		}
		sum *= 2;
		
		if ( expectedException == null )
			writeExpectedScalar("S", sum);

		// reference file should not be written to hdfs, so we set platform here
		rtplatform = execMode;
		if(rtplatform == Types.ExecMode.SPARK) {
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		}
		TestConfiguration config = availableTestConfigurations.get("aggregation");
		loadTestConfiguration(config);
		fullDMLScriptName = HOME + AGGREGATION_TEST_NAME + ".dml";
		programArgs = new String[] {"-checkPrivacy", "-args", "\"localhost:" + port + "/" + input("A") + "\"", Integer.toString(rows),
			Integer.toString(cols), Integer.toString(rows * 2), output("S"), output("R"), output("C")};

		runTest(true, (expectedException != null), expectedException, -1);

		// compare all sums via files
		if ( expectedException == null )
			compareResults(1e-11);

		assert(checkedPrivacyConstraintsContains(privacyLevel));

		TestUtils.shutdownThread(t);
		rtplatform = platformOld;
		DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
	}

	@Test
	public void transferPrivateTest() {
		federatedRCBind(Types.ExecMode.SINGLE_NODE, PrivacyLevel.Private, DMLException.class);
	}

	@Test
	public void transferPrivateAggregationTest() {
		federatedRCBind(Types.ExecMode.SINGLE_NODE, PrivacyLevel.PrivateAggregation, DMLException.class);
	}

	@Test
	public void transferNonePrivateTest() {
		federatedRCBind(Types.ExecMode.SINGLE_NODE, PrivacyLevel.None, null);
	}

	public void federatedRCBind(Types.ExecMode execMode, PrivacyLevel privacyLevel, Class<?> expectedException) {
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		Types.ExecMode platformOld = rtplatform;

		Thread t = null;

		getAndLoadTestConfiguration("transfer");
		String HOME = SCRIPT_DIR + TEST_DIR;

		double[][] A = getRandomMatrix(rows, cols, -10, 10, 1, 1);
		writeInputMatrixWithMTD("A", A, false, new MatrixCharacteristics(rows, cols, blocksize, rows * cols), new PrivacyConstraint(privacyLevel));

		int port = getRandomAvailablePort();
		t = startLocalFedWorker(port);

		// we need the reference file to not be written to hdfs, so we get the correct format
		rtplatform = Types.ExecMode.SINGLE_NODE;
		// Run reference dml script with normal matrix for Row/Col sum
		fullDMLScriptName = HOME + TRANSFER_TEST_NAME + "Reference.dml";
		programArgs = new String[] {"-checkPrivacy", "-args", input("A"), expected("R"), expected("C")};
		runTest(true, false, null, -1);

		// reference file should not be written to hdfs, so we set platform here
		rtplatform = execMode;
		if(rtplatform == Types.ExecMode.SPARK) {
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		}
		TestConfiguration config = availableTestConfigurations.get("transfer");
		loadTestConfiguration(config);
		fullDMLScriptName = HOME + TRANSFER_TEST_NAME + ".dml";
		programArgs = new String[] {"-checkPrivacy", "-args", "\"localhost:" + port + "/" + input("A") + "\"", Integer.toString(rows),
			Integer.toString(cols), output("R"), output("C")};

		runTest(true, (expectedException != null), expectedException, -1);

		// compare all sums via files
		if ( expectedException == null )
			compareResults(1e-11);
		
		assert(checkedPrivacyConstraintsContains(privacyLevel));

		TestUtils.shutdownThread(t);
		rtplatform = platformOld;
		DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
	}

	@Test
	public void matVecMultPrivateTest() {
		federatedMultiply(Types.ExecMode.SINGLE_NODE, PrivacyLevel.Private, DMLException.class);
	}

	@Test
	public void matVecMultPrivateAggregationTest() {
		federatedMultiply(Types.ExecMode.SINGLE_NODE, PrivacyLevel.PrivateAggregation, DMLException.class);
	}

	@Test
	public void matVecMultNonePrivateTest() {
		federatedMultiply(Types.ExecMode.SINGLE_NODE, PrivacyLevel.None, null);
	}

	public void federatedMultiply(Types.ExecMode execMode, PrivacyLevel privacyLevel, Class<?> expectedException) {
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		Types.ExecMode platformOld = rtplatform;
		rtplatform = execMode;
		if(rtplatform == Types.ExecMode.SPARK) {
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		}

		Thread t1, t2;

		getAndLoadTestConfiguration("matvecmult");
		String HOME = SCRIPT_DIR + TEST_DIR;

		// write input matrices
		int halfRows = rows / 2;
		// We have two matrices handled by a single federated worker
		double[][] X1 = getRandomMatrix(halfRows, cols, 0, 1, 1, 42);
		double[][] X2 = getRandomMatrix(halfRows, cols, 0, 1, 1, 1340);
		// And another two matrices handled by a single federated worker
		double[][] Y1 = getRandomMatrix(cols, halfRows, 0, 1, 1, 44);
		double[][] Y2 = getRandomMatrix(cols, halfRows, 0, 1, 1, 21);

		writeInputMatrixWithMTD("X1", X1, false, new MatrixCharacteristics(halfRows, cols, blocksize, halfRows * cols), new PrivacyConstraint(privacyLevel));
		writeInputMatrixWithMTD("X2", X2, false, new MatrixCharacteristics(halfRows, cols, blocksize, halfRows * cols));
		writeInputMatrixWithMTD("Y1", Y1, false, new MatrixCharacteristics(cols, halfRows, blocksize, halfRows * cols));
		writeInputMatrixWithMTD("Y2", Y2, false, new MatrixCharacteristics(cols, halfRows, blocksize, halfRows * cols));

		int port1 = getRandomAvailablePort();
		int port2 = getRandomAvailablePort();
		t1 = startLocalFedWorker(port1);
		t2 = startLocalFedWorker(port2);

		TestConfiguration config = availableTestConfigurations.get("matvecmult");
		loadTestConfiguration(config);

		// Run reference dml script with normal matrix
		fullDMLScriptName = HOME + MATVECMULT_TEST_NAME + "Reference.dml";
		programArgs = new String[] {"-nvargs", "X1=" + input("X1"), "X2=" + input("X2"), "Y1=" + input("Y1"),
			"Y2=" + input("Y2"), "Z=" + expected("Z")};
		runTest(true, false, null, -1);

		// Run actual dml script with federated matrix
		fullDMLScriptName = HOME + MATVECMULT_TEST_NAME + ".dml";
		programArgs = new String[] {"-checkPrivacy", 
			"-nvargs",
			"X1=" + TestUtils.federatedAddress("localhost", port1, input("X1")),
			"X2=" + TestUtils.federatedAddress("localhost", port2, input("X2")),
			"Y1=" + TestUtils.federatedAddress("localhost", port1, input("Y1")),
			"Y2=" + TestUtils.federatedAddress("localhost", port2, input("Y2")), "r=" + rows, "c=" + cols,
			"hr=" + halfRows, "Z=" + output("Z")};
		runTest(true, (expectedException != null), expectedException, -1);

		// compare via files
		if (expectedException == null)
			compareResults(1e-9);

		assert(checkedPrivacyConstraintsContains(privacyLevel));

		TestUtils.shutdownThreads(t1, t2);

		rtplatform = platformOld;
		DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
	}
}
