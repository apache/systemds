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

import org.apache.spark.sql.SparkSession;
import org.apache.sysml.api.mlcontext.MLContext;
import org.apache.sysml.api.mlcontext.Matrix;
import org.apache.sysml.api.mlcontext.Script;
import org.apache.sysml.api.mlcontext.ScriptFactory;
import org.junit.Assert;

/**
 * Abstract class for all Unary Op tests
 */
public abstract class UnaryOpTestsBase extends GPUTests {

	// Set of rows and column sizes & sparsities to test unary ops
	final int[] unaryOpRowSizes = new int[] { 1, 64, 130, 1024, 2049 };
	final int[] unaryOpColSizes = new int[] { 1, 64, 130, 1024, 2049 };
	final double[] unaryOpSparsities = new double[] { 0.00, 0.3, 0.9 };
	final int unaryOpSeed = 42;

	/**
	 * Tests unary ops with a variety of matrix shapes and sparsities.
	 * Test is skipped for blocks of size 1x1.
	 *
	 * @param function          name of the dml builtin unary op
	 * @param heavyHitterOpCode the string printed for the unary op heavy hitter when executed on gpu
	 * @param rows              array of row sizes
	 * @param columns           array of column sizes
	 * @param sparsities        array of sparsities
	 * @param seed              seed to use for random input matrix generation
	 */
	protected void testUnaryOpMatrixOutput(String function, String heavyHitterOpCode, int[] rows, int[] columns,
			double[] sparsities, int seed) {
		for (int i = 0; i < rows.length; i++) {
			for (int j = 0; j < columns.length; j++) {
				for (int k = 0; k < sparsities.length; k++) {
					int row = rows[i];
					int column = columns[j];
					double sparsity = sparsities[k];
					// Skip the case of a scalar unary op
					if (row == 1 && column == 1)
						continue;

					System.out.println("Matrix of size [" + row + ", " + column + "], sparsity = " + sparsity);
					Matrix in1 = generateInputMatrix(spark, row, column, sparsity, seed);
					Object outCPU = runUnaryOpOnCPU(spark, function, in1);
					Object outGPU = runUnaryOpOnGPU(spark, function, in1);
					//assertHeavyHitterPresent(heavyHitterOpCode);
					assertEqualObjects(outCPU, outGPU);
				}
			}
		}
	}

	protected void assertEqualObjects(Object outCPU, Object outGPU) {
		Assert.assertEquals(outCPU.getClass(), outGPU.getClass());

		if (outCPU instanceof Boolean) {
			Assert.assertEquals(((Boolean) outCPU).booleanValue(), ((Boolean) outGPU).booleanValue());
		} else if (outCPU instanceof Double) {
			Assert.assertEquals(((Double) outCPU).doubleValue(), ((Double) outGPU).doubleValue(), THRESHOLD);

		} else if (outCPU instanceof String) {
			Assert.assertEquals(outCPU.toString(), outGPU.toString());

		} else if (outCPU instanceof Integer) {
			Assert.assertEquals(((Integer) outCPU).intValue(), ((Integer) outGPU).intValue());

		} else if (outCPU instanceof Matrix)
			assertEqualMatrices((Matrix) outCPU, (Matrix) outGPU);
		else {
			Assert.fail("Invalid types for comparison");
		}
	}

	/**
	 * Runs the program for a unary op on the GPU
	 *
	 * @param spark    valid instance of {@link SparkSession}
	 * @param function dml builtin function string of the unary op
	 * @param in1      input matrix
	 * @return output from running the unary op on the input matrix (on GPU)
	 */
	protected Object runUnaryOpOnGPU(SparkSession spark, String function, Matrix in1) {
		String scriptStr2 = "out = " + function + "(in1)";
		MLContext gpuMLC = new MLContext(spark);
		gpuMLC.setGPU(true);
		gpuMLC.setForceGPU(true);
		gpuMLC.setStatistics(true);
		Script sinScript2 = ScriptFactory.dmlFromString(scriptStr2).in("in1", in1).out("out");
		Object outGPU = gpuMLC.execute(sinScript2).get("out");
		gpuMLC.close();
		return outGPU;
	}

	/**
	 * Runs the program for a unary op on the cpu
	 *
	 * @param spark    valid instance of {@link SparkSession}
	 * @param function dml builtin function string of the unary op
	 * @param in1      input matrix
	 * @return output from running the unary op on the input matrix
	 */
	protected Object runUnaryOpOnCPU(SparkSession spark, String function, Matrix in1) {
		String scriptStr1 = "out = " + function + "(in1)";
		MLContext cpuMLC = new MLContext(spark);
		Script sinScript = ScriptFactory.dmlFromString(scriptStr1).in("in1", in1).out("out");
		Object outCPU = cpuMLC.execute(sinScript).get("out");
		cpuMLC.close();
		return outCPU;
	}
}
