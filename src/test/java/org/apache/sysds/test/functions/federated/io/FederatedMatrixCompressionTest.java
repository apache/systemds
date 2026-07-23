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
import java.util.Random;
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
	/** Analytical bounds are derived, not measured; allow headroom. */
	private static final double SAFETY = 2.0;

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

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {OUTPUT_NAME}));
	}

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] {
			// {compressionType, rows, cols, sparsity, bits}
			{CompressionType.TOPK, 60, 20, 0.5, 8}, {CompressionType.PROBABILISTIC_QUANTIZATION, 60, 20, 1.0, 2},});
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

			// 1. Local reference (no federation, no compression)
			fullDMLScriptName = SCRIPT_DIR + TEST_DIR + TEST_NAME + "Reference.dml";
			programArgs = new String[] {"-nvargs", "in_X1=" + input("X1"), "rows=" + rows, "cols=" + cols,
				"out=" + expected(OUTPUT_NAME)};
			runTest(null);
			HashMap<MatrixValue.CellIndex, Double> refResults = readDMLMatrixFromExpectedDir(OUTPUT_NAME);

			// 2. Federated WITHOUT compression - must match the reference exactly
			DMLScript.FEDERATED_COMPRESSION = false;
			fullDMLScriptName = SCRIPT_DIR + TEST_DIR + TEST_NAME + ".dml";
			programArgs = new String[] {"-nvargs", "in_X1=" + input("X.json"), "rows=" + rows, "cols=" + cols,
				"out=" + output(OUTPUT_NAME)};
			runTest(null);
			HashMap<MatrixValue.CellIndex, Double> uncompressed = readDMLMatrixFromOutputDir(OUTPUT_NAME);
			TestUtils.compareMatrices(uncompressed, refResults, 1e-9, "FedUncompressed", "Ref");

			// 3. Federated WITH compression
			DMLScript.FEDERATED_COMPRESSION = true;
			DMLScript.FEDERATED_COMPRESSION_TYPE = compressionType;
			DMLScript.FEDERATED_COMPRESSION_SPARSITY = sparsity;
			DMLScript.FEDERATED_COMPRESSION_BITS = bits;
			runTest(null);
			HashMap<MatrixValue.CellIndex, Double> compressed = readDMLMatrixFromOutputDir(OUTPUT_NAME);

			// 4. Compression must have altered the data
			double maxDiff = maxAbsDifference(compressed, uncompressed);
			Assert.assertTrue("Compression with " + compressionType
				+ " did not alter the broadcast data - the feature is not being exercised", maxDiff > 1e-9);

			// 5. ... but reconstruction must still be bounded
			double bound = reconstructionBound();
			Assert.assertTrue("Reconstruction error " + maxDiff + " exceeds bound " + bound + " for " + compressionType,
				maxDiff <= bound);
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

	/**
	 * Upper bound on the reconstruction error in Z = X %*% B, derived from the compression parameters rather than
	 * measured. X entries are in [0,1) and B = seq(1, cols), so each element of Z is a weighted sum of at most cols
	 * terms with weights 1..cols.
	 */
	private double reconstructionBound() {
		if(compressionType == CompressionType.TOPK) {
			// TopK keeps the k largest weights and zeroes the rest; the dropped weights are 1..d
			int kept = (int) Math.max(1, Math.ceil(cols * sparsity));
			int d = Math.max(0, cols - kept);
			return (double) d * (d + 1) / 2 * SAFETY;
		}
		// Quantization: each weight moves by at most one level step
		double step = (double) (cols - 1) / ((1 << bits) - 1);
		return cols * step * SAFETY;
	}

	public double[][] createRandomMatrix(int width, int height) {
		Random rd = new Random();
		double[][] matrix = new double[height][];

		for(int i = 0; i < height; i++)
			matrix[i] = rd.doubles(width).toArray();

		return matrix;
	}
}
