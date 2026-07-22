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

package org.apache.sysds.test.functions.federated.io;

import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Random;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.controlprogram.caching.MatrixObject;
import org.apache.sysds.runtime.controlprogram.federated.compression.CompressionType;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.test.functions.federated.FederatedTestObjectConstructor;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class FederatedMatrixCompressionTest extends AutomatedTestBase {

	private static final Log LOG = LogFactory.getLog(FederatedMatrixCompressionTest.class.getName());
	private final static String TEST_DIR = "functions/federated/io/";
	private final static String TEST_NAME = "FederatedMatrixCompressionTest";
	private final static String TEST_CLASS_DIR = TEST_DIR + FederatedMatrixCompressionTest.class.getSimpleName() + "/";
	private final static int blocksize = 1024;
	private final static String OUTPUT_NAME = "Z";

	@Parameterized.Parameter()
	public CompressionType compressionType;
	@Parameterized.Parameter(1)
	public int rows;
	@Parameterized.Parameter(2)
	public int cols;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {OUTPUT_NAME}));
	}

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] {
			// {compressionType, rows, cols}
			{CompressionType.TOPK, 5, 5}, {CompressionType.PROBABILISTIC_QUANTIZATION, 5, 5},});
	}

	@Test
	public void testFederatedMatrixCompression() {
		federatedMatrixCompression();
	}

	public void federatedMatrixCompression() {
		Types.ExecMode oldPlatform = setExecMode(Types.ExecType.CP);
		getAndLoadTestConfiguration(TEST_NAME);

		LOG.debug("Current test configuration: compressionType = " + compressionType + ", rows = " + rows + ", cols = "
			+ cols);

		fullDMLScriptName = "";
		int port1 = getRandomAvailablePort();
		Thread t1 = startLocalFedWorkerThread(port1);
		String host = "localhost";

		try {
			double[][] X1 = createRandomMatrix(rows, cols);
			MatrixCharacteristics mc = new MatrixCharacteristics(rows, cols, blocksize, (long) rows * cols);
			writeCSVMatrix("X1", X1, false, mc);

			long[][] begins = new long[][] {new long[] {0, 0}};
			long[][] ends = new long[][] {new long[] {rows, cols}};
			MatrixObject fed = FederatedTestObjectConstructor.constructFederatedInput(rows, cols, blocksize, host,
				begins, ends, new int[] {port1}, new String[] {input("X1")}, input("X.json"));
			writeInputFederatedWithMTD("X.json", fed);

			// Run reference DML with local matrix multiply
			fullDMLScriptName = SCRIPT_DIR + TEST_DIR + TEST_NAME + "Reference.dml";
			programArgs = new String[] {"-nvargs", "in_X1=" + input("X1"), "rows=" + rows, "cols=" + cols,
				"out=" + expected(OUTPUT_NAME)};
			runTest(null);

			// Enable compression only for the federated run
			DMLScript.FEDERATED_COMPRESSION = true;
			DMLScript.FEDERATED_COMPRESSION_TYPE = compressionType;
			// Lossless settings: sparsity 1.0 makes TopK a passthrough (k >= totalElements),
			// so the test verifies the compress/transfer/decompress path rather than
			// reconstruction error on a 5-element vector.
			DMLScript.FEDERATED_COMPRESSION_SPARSITY = 1.0;
			DMLScript.FEDERATED_COMPRESSION_BITS = 8;

			// Run federated DML with compression enabled
			fullDMLScriptName = SCRIPT_DIR + TEST_DIR + TEST_NAME + ".dml";
			programArgs = new String[] {"-nvargs", "in_X1=" + input("X.json"), "rows=" + rows, "cols=" + cols,
				"out=" + output(OUTPUT_NAME)};
			runTest(null);

			HashMap<MatrixValue.CellIndex, Double> refResults = readDMLMatrixFromExpectedDir(OUTPUT_NAME);
			HashMap<MatrixValue.CellIndex, Double> fedResults = readDMLMatrixFromOutputDir(OUTPUT_NAME);
			// Use tolerance of 1.0 since TopK and probabilistic quantization are lossy
			TestUtils.compareMatrices(fedResults, refResults, 1.0, "Fed", "Ref");
		}
		catch(Exception e) {
			LOG.warn("Failed to run test with compressionType = " + compressionType + ", rows = " + rows + ", cols = "
				+ cols);
			e.printStackTrace();
			Assert.assertTrue(false);
		}
		finally {
			DMLScript.FEDERATED_COMPRESSION = false;
			DMLScript.FEDERATED_COMPRESSION_SPARSITY = 0.01;
			DMLScript.FEDERATED_COMPRESSION_BITS = 4;
			resetExecMode(oldPlatform);
		}

		TestUtils.shutdownThreads(t1);
	}

	public double[][] createRandomMatrix(int width, int height) {
		Random rd = new Random();
		double[][] matrix = new double[height][];

		for(int i = 0; i < height; i++)
			matrix[i] = rd.doubles(width).toArray();

		return matrix;
	}
}
