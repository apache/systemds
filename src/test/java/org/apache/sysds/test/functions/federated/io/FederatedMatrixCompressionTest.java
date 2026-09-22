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
import java.util.HashSet;
import java.util.Set;

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
	@Parameterized.Parameter(3)
	public double sparsity;
	@Parameterized.Parameter(4)
	public int bits;
	@Parameterized.Parameter(5)
	public int workers;

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {OUTPUT_NAME}));
	}

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] {
			// {compressionType, rows, cols, sparsity, bits, workers}
			{CompressionType.TOPK, 60, 20, 0.5, 8, 1}, {CompressionType.TOPK, 60, 20, 0.5, 8, 2},
			{CompressionType.PROBABILISTIC_QUANTIZATION, 60, 20, 1.0, 2, 1},
			{CompressionType.PROBABILISTIC_QUANTIZATION, 60, 20, 1.0, 2, 2},});
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

		int halfRows = rows / 2;
		long[][] begins = new long[][] {new long[] {0, 0}, new long[] {halfRows, 0}};
		long[][] ends = new long[][] {new long[] {halfRows, cols}, new long[] {rows, cols}};
		double[][] X1 = getRandomMatrix(halfRows, cols, 0, 1, 1, 42);
		double[][] X2 = getRandomMatrix(rows - halfRows, cols, 0, 1, 1, 1340);
		writeInputMatrixWithMTD("X1", X1, false,
			new MatrixCharacteristics(halfRows, cols, blocksize, (long) halfRows * cols));
		writeInputMatrixWithMTD("X2", X2, false,
			new MatrixCharacteristics(rows - halfRows, cols, blocksize, (long) (rows - halfRows) * cols));

		fullDMLScriptName = "";
		int port1 = getRandomAvailablePort();
		int port2 = getRandomAvailablePort();
		Thread[] threads = startLocalFedWorkerThreads(new int[] {port1, port2}, null, FED_WORKER_WAIT);
		String host = "localhost";

		try {
			MatrixObject fed = FederatedTestObjectConstructor.constructFederatedInput(rows, cols, blocksize, host,
				begins, ends, workers == 2 ? new int[] {port1, port2} : new int[] {port1},
				workers == 2 ? new String[] {input("X1"), input("X2")} : new String[] {input("X1")}, input("X.json"));
			writeInputFederatedWithMTD("X.json", fed);

			// Reference: recombine in DML (no Java loops). One-worker reads one half,
			// two-worker rbinds both halves.
			if(workers == 1) {
				fullDMLScriptName = SCRIPT_DIR + TEST_DIR + TEST_NAME + "1Reference.dml";
				programArgs = new String[] {"-nvargs", "in_X1=" + input("X1"), "cols=" + cols,
					"out=" + expected(OUTPUT_NAME)};
			}
			else {
				fullDMLScriptName = SCRIPT_DIR + TEST_DIR + TEST_NAME + "2Reference.dml";
				programArgs = new String[] {"-nvargs", "in_X1=" + input("X1"), "in_X2=" + input("X2"), "cols=" + cols,
					"out=" + expected(OUTPUT_NAME)};
			}
			runTest(null);
			HashMap<MatrixValue.CellIndex, Double> refResults = readDMLMatrixFromExpectedDir(OUTPUT_NAME);

			// Federated WITHOUT compression - must match the reference exactly
			DMLScript.FEDERATED_COMPRESSION = false;
			fullDMLScriptName = SCRIPT_DIR + TEST_DIR + TEST_NAME + ".dml";
			programArgs = new String[] {"-nvargs", "in_X=" + input("X.json"), "cols=" + cols,
				"out=" + output(OUTPUT_NAME)};
			runTest(null);
			HashMap<MatrixValue.CellIndex, Double> uncompressed = readDMLMatrixFromOutputDir(OUTPUT_NAME);
			TestUtils.compareMatrices(uncompressed, refResults, 1e-9, "FedUncompressed", "Ref");

			// Federated WITH compression - must run and must differ from uncompressed,
			// proving compression is actually applied on the transferred matrices.
			DMLScript.FEDERATED_COMPRESSION = true;
			DMLScript.FEDERATED_COMPRESSION_TYPE = compressionType;
			DMLScript.FEDERATED_COMPRESSION_SPARSITY = sparsity;
			DMLScript.FEDERATED_COMPRESSION_BITS = bits;
			runTest(null);
			HashMap<MatrixValue.CellIndex, Double> compressed = readDMLMatrixFromOutputDir(OUTPUT_NAME);

			double maxDiff = maxAbsDifference(compressed, uncompressed);
			Assert.assertTrue("Compression with " + compressionType
				+ " did not alter the transferred data - the feature is not being exercised", maxDiff > 1e-9);
		}
		catch(Exception e) {
			LOG.warn("Failed with compressionType = " + compressionType + ", workers = " + workers);
			e.printStackTrace();
			Assert.assertTrue(false);
		}
		finally {
			DMLScript.FEDERATED_COMPRESSION = false;
			DMLScript.FEDERATED_COMPRESSION_SPARSITY = 0.01;
			DMLScript.FEDERATED_COMPRESSION_BITS = 4;
			resetExecMode(oldPlatform);
		}

		TestUtils.shutdownThreads(threads);
	}

	/**
	 * Maximum absolute difference between two result matrices, treating a cell absent from either map as zero.
	 */
	private static double maxAbsDifference(HashMap<MatrixValue.CellIndex, Double> a,
		HashMap<MatrixValue.CellIndex, Double> b) {
		double max = 0;
		Set<MatrixValue.CellIndex> keys = new HashSet<>(a.keySet());
		keys.addAll(b.keySet());
		for(MatrixValue.CellIndex k : keys) {
			double va = a.getOrDefault(k, 0.0);
			double vb = b.getOrDefault(k, 0.0);
			max = Math.max(max, Math.abs(va - vb));
		}
		return max;
	}
}
