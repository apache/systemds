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

package org.apache.sysml.test.gpu;

import java.util.ArrayList;
import java.util.Formatter;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.spark.sql.SparkSession;
import org.apache.sysml.api.mlcontext.MLContext;
import org.apache.sysml.api.mlcontext.Matrix;
import org.apache.sysml.api.mlcontext.Script;
import org.apache.sysml.api.mlcontext.ScriptFactory;
import org.apache.sysml.runtime.DMLRuntimeException;
import org.apache.sysml.runtime.instructions.gpu.context.GPUContext;
import org.apache.sysml.runtime.instructions.gpu.context.GPUContextPool;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.utils.Statistics;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Assert;
import org.junit.BeforeClass;

/**
 * Parent class for all GPU tests
 */
public abstract class GPUTests extends AutomatedTestBase {

	protected final static String TEST_DIR = "org/apache/sysml/api/mlcontext";
	protected static SparkSession spark;
	protected final double THRESHOLD = 1e-9;    // for relative error

	@BeforeClass
	public static void beforeClass() {
		spark = createSystemMLSparkSession("GPUTests", "local");
	}

	@AfterClass
	public static void afterClass() {
		spark.close();
	}

	/**
	 * Gets threshold for relative error in tests
	 *
	 * @return a valid threshold
	 */
	protected double getTHRESHOLD() {
		return THRESHOLD;
	}

	@After
	public void tearDown() {
		clearGPUMemory();
		super.tearDown();
	}

	/**
	 * Clear out the memory on all GPUs
	 */
	protected void clearGPUMemory() {
		try {
			int count = GPUContextPool.getDeviceCount();
			int freeCount = GPUContextPool.getAvailableCount();
			Assert.assertTrue("All GPUContexts have not been returned to the GPUContextPool", count == freeCount);

			List<GPUContext> gCtxs = GPUContextPool.reserveAllGPUContexts();
			for (GPUContext gCtx : gCtxs) {
				gCtx.initializeThread();
				gCtx.clearMemory();
			}
			GPUContextPool.freeAllGPUContexts();


		} catch (DMLRuntimeException e) {
			// Ignore
		}
	}

	/**
	 * Generates a random input matrix with a given size and sparsity
	 *
	 * @param spark    valid instance of {@link SparkSession}
	 * @param m        number of rows
	 * @param n        number of columns
	 * @param sparsity sparsity (1 = completely dense, 0 = completely sparse)
	 * @return a random matrix with given size and sparsity
	 */
	protected Matrix generateInputMatrix(SparkSession spark, int m, int n, double sparsity, int seed) {
		// Generate a random matrix of size m * n
		MLContext genMLC = new MLContext(spark);
		String scriptStr;
		if (sparsity == 0.0) {
			scriptStr = "in1 = matrix(0, rows=" + m + ", cols=" + n + ")";
		} else {
			scriptStr = "in1 = rand(rows=" + m + ", cols=" + n + ", sparsity = " + sparsity + ", seed= " + seed
					+ ", min=-1.0, max=1.0)";
		}
		Script generateScript = ScriptFactory.dmlFromString(scriptStr).out("in1");
		Matrix in1 = genMLC.execute(generateScript).getMatrix("in1");
		genMLC.close();
		return in1;
	}

	/**
	 * Generates a random input matrix with a given size and sparsity
	 *
	 * @param spark    valid instance of {@link SparkSession}
	 * @param m        number of rows
	 * @param n        number of columns
	 * @param min      min for RNG
	 * @param max      max for RNG
	 * @param sparsity sparsity (1 = completely dense, 0 = completely sparse)
	 * @return a random matrix with given size and sparsity
	 */
	protected Matrix generateInputMatrix(SparkSession spark, int m, int n, double min, double max, double sparsity, int seed) {
		// Generate a random matrix of size m * n
		MLContext genMLC = new MLContext(spark);
		String scriptStr;
		if (sparsity == 0.0) {
			scriptStr = "in1 = matrix(0, rows=" + m + ", cols=" + n + ")";
		} else {
			scriptStr = "in1 = rand(rows=" + m + ", cols=" + n + ", sparsity = " + sparsity + ", seed= " + seed
					+ ", min=" + min + ", max=" + max + ")";
		}
		Script generateScript = ScriptFactory.dmlFromString(scriptStr).out("in1");
		Matrix in1 = genMLC.execute(generateScript).getMatrix("in1");
		genMLC.close();
		return in1;
	}

	/**
	 * Asserts that the values in two matrices are in {@link UnaryOpTests#THRESHOLD} of each other
	 *
	 * @param expected expected matrix
	 * @param actual   actual matrix
	 */
	private void assertEqualMatrices(Matrix expected, Matrix actual) {
		try {
			MatrixBlock expectedMB = expected.toMatrixObject().acquireRead();
			MatrixBlock actualMB = actual.toMatrixObject().acquireRead();

			long rows = expectedMB.getNumRows();
			long cols = expectedMB.getNumColumns();
			Assert.assertEquals(rows, actualMB.getNumRows());
			Assert.assertEquals(cols, actualMB.getNumColumns());

			for (int i = 0; i < rows; i++) {
				for (int j = 0; j < cols; j++) {
					double expectedDouble = expectedMB.quickGetValue(i, j);
					double actualDouble = actualMB.quickGetValue(i, j);
					if (expectedDouble != 0.0 && !Double.isNaN(expectedDouble) && Double.isFinite(expectedDouble)) {
						double relativeError = Math.abs((expectedDouble - actualDouble) / expectedDouble);
						Formatter format = new Formatter();
						format.format(
								"Relative error(%f) is more than threshold (%f). Expected = %f, Actual = %f, differed at [%d, %d]",
								relativeError, getTHRESHOLD(), expectedDouble, actualDouble, i, j);
						Assert.assertTrue(format.toString(), relativeError < getTHRESHOLD());
						format.close();
					} else {
						Assert.assertEquals(expectedDouble, actualDouble, getTHRESHOLD());
					}
				}
			}
			expected.toMatrixObject().release();
			actual.toMatrixObject().release();
		} catch (DMLRuntimeException e) {
			throw new RuntimeException(e);
		}
	}

	/**
	 * asserts that the expected op was executed
	 *
	 * @param heavyHitterOpCode opcode of the heavy hitter for the unary op
	 */
	protected void assertHeavyHitterPresent(String heavyHitterOpCode) {
		Set<String> heavyHitterOpCodes = Statistics.getCPHeavyHitterOpCodes();
		Assert.assertTrue(heavyHitterOpCodes.contains(heavyHitterOpCode));
	}

	/**
	 * Runs a program on the CPU
	 *
	 * @param spark     a valid {@link SparkSession}
	 * @param scriptStr the script to run (as a string)
	 * @param inputs    map of input variables names in the scriptStr (of variable_name -> object)
	 * @param outStrs   list of variable names needed as output from the scriptStr
	 * @return list of output objects in order of outStrs
	 */
	protected List<Object> runOnCPU(SparkSession spark, String scriptStr, Map<String, Object> inputs,
			List<String> outStrs) {
		MLContext cpuMLC = new MLContext(spark);
		List<Object> outputs = new ArrayList<>();
		Script script = ScriptFactory.dmlFromString(scriptStr).in(inputs).out(outStrs);
		for (String outStr : outStrs) {
			Object output = cpuMLC.execute(script).get(outStr);
			outputs.add(output);
		}
		cpuMLC.close();
		return outputs;
	}

	/**
	 * Runs a program on the GPU
	 *
	 * @param spark     a valid {@link SparkSession}
	 * @param scriptStr the script to run (as a string)
	 * @param inputs    map of input variables names in the scriptStr (of variable_name -> object)
	 * @param outStrs   list of variable names needed as output from the scriptStr
	 * @return list of output objects in order of outStrs
	 */
	protected List<Object> runOnGPU(SparkSession spark, String scriptStr, Map<String, Object> inputs,
			List<String> outStrs) {
		MLContext gpuMLC = new MLContext(spark);
		gpuMLC.setGPU(true);
		gpuMLC.setForceGPU(true);
		gpuMLC.setStatistics(true);
		List<Object> outputs = new ArrayList<>();
		Script script = ScriptFactory.dmlFromString(scriptStr).in(inputs).out(outStrs);
		for (String outStr : outStrs) {
			Object output = gpuMLC.execute(script).get(outStr);
			outputs.add(output);
		}
		gpuMLC.close();
		return outputs;
	}

	/**
	 * Assert that the two objects are equal. Supported types are Boolean, Integer, String, Double and Matrix
	 *
	 * @param expected
	 * @param actual
	 */
	protected void assertEqualObjects(Object expected, Object actual) {
		Assert.assertEquals(expected.getClass(), actual.getClass());

		if (expected instanceof Boolean) {
			Assert.assertEquals(((Boolean) expected).booleanValue(), ((Boolean) actual).booleanValue());
		} else if (expected instanceof Double) {
			double expectedDouble = ((Double) expected).doubleValue();
			double actualDouble = ((Double) actual).doubleValue();
			if (expectedDouble != 0.0 && !Double.isNaN(expectedDouble) && Double.isFinite(expectedDouble)) {
				double relativeError = Math.abs((expectedDouble - actualDouble) / expectedDouble);
				Assert.assertTrue("Comparing floating point numbers, relative error(" + relativeError
						+ ") is more than threshold (" + getTHRESHOLD() + ")", relativeError < getTHRESHOLD());
			} else {
				Assert.assertEquals(expectedDouble, actualDouble, getTHRESHOLD());
			}
		} else if (expected instanceof String) {
			Assert.assertEquals(expected.toString(), actual.toString());
		} else if (expected instanceof Integer) {
			Assert.assertEquals(((Integer) expected).intValue(), ((Integer) actual).intValue());
		} else if (expected instanceof Matrix)
			assertEqualMatrices((Matrix) expected, (Matrix) actual);
		else {
			Assert.fail("Invalid types for comparison");
		}
	}
}
