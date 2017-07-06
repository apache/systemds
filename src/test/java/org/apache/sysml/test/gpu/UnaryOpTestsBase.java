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

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import org.apache.sysml.api.mlcontext.Matrix;

/**
 * Abstract class for all Unary Op tests
 */
public abstract class UnaryOpTestsBase extends GPUTests {

	// Set of rows and column sizes & sparsities to test unary ops
	private final int[] rowSizes = new int[] { 2049, 1024, 140, 64, 1 };
	private final int[] columnSizes = new int[] { 2049, 1024, 140, 64, 1 };
	private final double[] sparsities = new double[] { 0.9, 0.3, 0.03, 0.0 };
	private final int seed = 42;

	/**
	 * Tests unary ops with a variety of matrix shapes and sparsities.
	 * Test is skipped for blocks of size 1x1.
	 *
	 * @param function          name of the dml builtin unary op
	 * @param heavyHitterOpCode the string printed for the unary op heavy hitter when executed on gpu
	 */
	protected void testSimpleUnaryOpMatrixOutput(String function, String heavyHitterOpCode) {
		String scriptStr = "out = " + function + "(in1)";
		testUnaryOpMatrixOutput(scriptStr, heavyHitterOpCode, "in1", "out");
	}

	/**
	 * Tests slightly more involved unary ops with a variety of matrix shapes and sparsities.
	 * Test is skipped for blocks of size 1x1
	 *
	 * @param scriptStr         script string
	 * @param heavyHitterOpCode the string printed for the unary op heavy hitter when executed on gpu
	 * @param inStr             name of input variable in provided script string
	 * @param outStr            name of output variable in script string
	 */
	protected void testUnaryOpMatrixOutput(String scriptStr, String heavyHitterOpCode, String inStr, String outStr) {
		int[] rows = rowSizes;
		int[] columns = columnSizes;
		double[] sparsities = this.sparsities;
		int seed = this.seed;

		for (int i = 0; i < rows.length; i++) {
			for (int j = 0; j < columns.length; j++) {
				for (int k = 0; k < sparsities.length; k++) {
					int row = rows[i];
					int column = columns[j];
					double sparsity = sparsities[k];
					// Skip the case of a scalar unary op
					if (row == 1 && column == 1)
						continue;

					testUnaryOpMatrixOutput(scriptStr, heavyHitterOpCode, inStr, outStr, seed, row, column, sparsity);
				}
			}
		}
	}

	/**
	 * Tests a single unary op with inputs and outputs of the specified size and sparsity
	 *
	 * @param scriptStr         script string
	 * @param heavyHitterOpCode the string printed for the unary op heavy hitter when executed on gpu
	 * @param inStr             name of input variable in provided script string
	 * @param outStr            name of output variable in script string
	 * @param seed              seed for the random number generator for the random input matrix
	 * @param row               number of rows of input matrix
	 * @param column            number of rows of input matrix
	 * @param sparsity          sparsity of the input matrix
	 */
	public void testUnaryOpMatrixOutput(String scriptStr, String heavyHitterOpCode, String inStr, String outStr,
			int seed, int row, int column, double sparsity) {
		System.out.println("Matrix of size [" + row + ", " + column + "], sparsity = " + sparsity);
		Matrix in1 = generateInputMatrix(spark, row, column, sparsity, seed);
		HashMap<String, Object> inputs = new HashMap<>();
		inputs.put(inStr, in1);
		List<Object> outCPU = runOnCPU(spark, scriptStr, inputs, Arrays.asList(outStr));
		List<Object> outGPU = runOnGPU(spark, scriptStr, inputs, Arrays.asList(outStr));
		//assertHeavyHitterPresent(heavyHitterOpCode);
		assertEqualObjects(outCPU.get(0), outGPU.get(0));
	}

}
