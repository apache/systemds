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

package org.apache.sysds.test.functions.federated.primitives.part5;

import org.junit.Test;
import org.junit.runners.Parameterized;
import org.junit.runner.RunWith;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.util.CommonThreadPool;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

import java.util.Arrays;

import static java.lang.Thread.sleep;

@RunWith(Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class FederatedMatrixScalarOperationsTest extends AutomatedTestBase {
	@Parameterized.Parameters
	public static Iterable<Object[]> data() {
		return Arrays.asList(new Object[][] {
			{100, 100}, 
		// {10000, 100}
		});
	}

	// internals 4 parameterized tests
	@Parameterized.Parameter()
	public int rows;
	@Parameterized.Parameter(1)
	public int cols;

	// System test paths
	private static final String TEST_DIR = "functions/federated/matrix_scalar/";
	private static final String TEST_CLASS_DIR = TEST_DIR + FederatedMatrixScalarOperationsTest.class.getSimpleName()
		+ "/";
	private static final String TEST_PROG_MATRIX_ADDITION_SCALAR = "FederatedMatrixAdditionScalar";
	private static final String TEST_PROG_MATRIX_SUBTRACTION_SCALAR = "FederatedMatrixSubtractionScalar";
	private static final String TEST_PROG_MATRIX_MULTIPLICATION_SCALAR = "FederatedMatrixMultiplicationScalar";
	private static final String TEST_PROG_SCALAR_ADDITION_MATRIX = "FederatedScalarAdditionMatrix";
	private static final String TEST_PROG_SCALAR_SUBTRACTION_MATRIX = "FederatedScalarSubtractionMatrix";
	private static final String TEST_PROG_SCALAR_MULTIPLICATION_MATRIX = "FederatedScalarMultiplicationMatrix";

	private static final String FEDERATED_WORKER_HOST = "localhost";
	private static final int FEDERATED_WORKER_PORT = 1222;

	@Override
	public void setUp() {
		// Save Result to File R
		addTestConfiguration(
			new TestConfiguration(TEST_CLASS_DIR, TEST_PROG_MATRIX_ADDITION_SCALAR, new String[] {"R"}));
		addTestConfiguration(
			new TestConfiguration(TEST_CLASS_DIR, TEST_PROG_MATRIX_SUBTRACTION_SCALAR, new String[] {"R"}));
		addTestConfiguration(
			new TestConfiguration(TEST_CLASS_DIR, TEST_PROG_MATRIX_MULTIPLICATION_SCALAR, new String[] {"R"}));
		addTestConfiguration(
			new TestConfiguration(TEST_CLASS_DIR, TEST_PROG_SCALAR_ADDITION_MATRIX, new String[] {"R"}));
		addTestConfiguration(
			new TestConfiguration(TEST_CLASS_DIR, TEST_PROG_SCALAR_SUBTRACTION_MATRIX, new String[] {"R"}));
		addTestConfiguration(
			new TestConfiguration(TEST_CLASS_DIR, TEST_PROG_SCALAR_MULTIPLICATION_MATRIX, new String[] {"R"}));
	}

	@Test
	public void testFederatedMatrixAdditionScalar() {
		getAndLoadTestConfiguration(TEST_PROG_MATRIX_ADDITION_SCALAR);

		double[][] m = getRandomMatrix(this.rows, this.cols, -1, 1, 1.0, 1);
		writeInputMatrixWithMTD("M", m, true);
		int scalar = TestUtils.getRandomInt();
		double[][] r = new double[rows][cols];
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				r[i][j] = m[i][j] + scalar;
			}
		}
		writeExpectedMatrix("R", r);

		runGenericTest(TEST_PROG_MATRIX_ADDITION_SCALAR, scalar);
	}

	@Test
	public void testFederatedMatrixSubtractionScalar() {
		getAndLoadTestConfiguration(TEST_PROG_MATRIX_SUBTRACTION_SCALAR);

		double[][] m = getRandomMatrix(this.rows, this.cols, -1, 1, 1.0, 1);
		writeInputMatrixWithMTD("M", m, true);
		int scalar = TestUtils.getRandomInt();
		double[][] r = new double[rows][cols];
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				r[i][j] = m[i][j] - scalar;
			}
		}
		writeExpectedMatrix("R", r);

		runGenericTest(TEST_PROG_MATRIX_SUBTRACTION_SCALAR, scalar);
	}

	@Test
	public void testFederatedMatrixMultiplicationScalar() {
		getAndLoadTestConfiguration(TEST_PROG_MATRIX_MULTIPLICATION_SCALAR);

		double[][] m = getRandomMatrix(this.rows, this.cols, -1, 1, 1.0, 1);
		writeInputMatrixWithMTD("M", m, true);
		int scalar = TestUtils.getRandomInt();
		double[][] r = new double[rows][cols];
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				r[i][j] = m[i][j] * scalar;
			}
		}
		writeExpectedMatrix("R", r);

		runGenericTest(TEST_PROG_MATRIX_MULTIPLICATION_SCALAR, scalar);
	}

	@Test
	public void testScalarAdditionFederatedMatrix() {
		getAndLoadTestConfiguration(TEST_PROG_SCALAR_ADDITION_MATRIX);

		double[][] m = getRandomMatrix(this.rows, this.cols, -1, 1, 1.0, 1);
		writeInputMatrixWithMTD("M", m, true);
		int scalar = TestUtils.getRandomInt();
		double[][] r = new double[rows][cols];
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				r[i][j] = m[i][j] + scalar;
			}
		}
		writeExpectedMatrix("R", r);

		runGenericTest(TEST_PROG_SCALAR_ADDITION_MATRIX, scalar);
	}

	@Test
	public void testScalarSubtractionFederatedMatrix() {
		getAndLoadTestConfiguration(TEST_PROG_SCALAR_SUBTRACTION_MATRIX);

		double[][] m = getRandomMatrix(this.rows, this.cols, -1, 1, 1.0, 1);
		writeInputMatrixWithMTD("M", m, true);
		int scalar = TestUtils.getRandomInt();
		double[][] r = new double[rows][cols];
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				r[i][j] = scalar - m[i][j];
			}
		}
		writeExpectedMatrix("R", r);

		runGenericTest(TEST_PROG_SCALAR_SUBTRACTION_MATRIX, scalar);
	}

	@Test
	public void testScalarMultiplicationFederatedMatrix() {
		getAndLoadTestConfiguration(TEST_PROG_SCALAR_MULTIPLICATION_MATRIX);

		double[][] m = getRandomMatrix(this.rows, this.cols, -1, 1, 1.0, 1);
		writeInputMatrixWithMTD("M", m, true);
		int scalar = TestUtils.getRandomInt();
		double[][] r = new double[rows][cols];
		for(int i = 0; i < rows; i++) {
			for(int j = 0; j < cols; j++) {
				r[i][j] = m[i][j] * scalar;
			}
		}
		writeExpectedMatrix("R", r);

		runGenericTest(TEST_PROG_SCALAR_MULTIPLICATION_MATRIX, scalar);
	}

	private void runGenericTest(String dmlFile, int scalar) {
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		Types.ExecMode platformOld = rtplatform;

		Thread t = null;
		try {
			// we need the reference file to not be written to hdfs, so we get the correct format
			rtplatform = Types.ExecMode.SINGLE_NODE;
			programArgs = new String[] {"-w", Integer.toString(FEDERATED_WORKER_PORT)};
			CommonThreadPool.get().submit(() -> runTest(true, false, null, -1));
			sleep(FED_WORKER_WAIT);
			fullDMLScriptName = SCRIPT_DIR + TEST_DIR + dmlFile + ".dml";
			programArgs = new String[] {"-nvargs",
				"in=" + TestUtils.federatedAddress(FEDERATED_WORKER_HOST, FEDERATED_WORKER_PORT, input("M")),
				"rows=" + rows, "cols=" + cols, "scalar=" + scalar, "out=" + output("R")};
			runTest(true, false, null, -1);

			compareResults();
		}
		catch(InterruptedException e) {
			e.printStackTrace();
			assert (false);
		}
		finally {
			rtplatform = platformOld;
			TestUtils.shutdownThread(t);
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}
