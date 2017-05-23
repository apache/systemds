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

import org.apache.sysml.api.mlcontext.Matrix;
import org.junit.Assert;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

/**
 * Abstract class for all Unary Op tests
 */
public abstract class UnaryOpTestsBase extends GPUTests {

	// Set of rows and column sizes & sparsities to test unary ops
	final int[] unaryOpRowSizes = new int[] { 1, 64, 130, 1024, 2049 };
	final int[] unaryOpColSizes = new int[] { 1, 64, 130, 1024, 2049 };
	final double[] unaryOpSparsities = new double[] { 0.0, 0.03, 0.3, 0.9 };
	final int unaryOpSeed = 42;

	/**
	 * Tests unary ops with a variety of matrix shapes and sparsities.
	 * Test is skipped for blocks of size 1x1.
	 *  @param function          name of the dml builtin unary op
	 * @param heavyHitterOpCode the string printed for the unary op heavy hitter when executed on gpu
	 */
	protected void testSimpleUnaryOpMatrixOutput(String function, String heavyHitterOpCode) {
		String scriptStr = "out = " + function + "(in1)";
		testUnaryOpMatrixOutput(scriptStr, heavyHitterOpCode, "in1", "out");
	}

	/**
	 * Tests slightly more involved unary ops with a variety of matrix shapes and sparsities.
	 * Test is skipped for blocks of size 1x1
	 * @param scriptStr script string
	 * @param heavyHitterOpCode the string printed for the unary op heavy hitter when executed on gpu
	 * @param inStr name of input variable in provided script string
	 * @param outStr name of output variable in script string
	 */
	protected void testUnaryOpMatrixOutput(String scriptStr, String heavyHitterOpCode, String inStr, String outStr) {
		int[] rows = unaryOpRowSizes;
		int[] columns = unaryOpColSizes;
		double[] sparsities = unaryOpSparsities;
		int seed = unaryOpSeed;

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
					HashMap<String, Object> inputs = new HashMap<>();
					inputs.put(inStr, in1);
					List<Object> outCPU = runOnCPU(spark, scriptStr, inputs, Arrays.asList(outStr));
					List<Object> outGPU = runOnGPU(spark, scriptStr, inputs, Arrays.asList(outStr));
					//assertHeavyHitterPresent(heavyHitterOpCode);
					assertEqualObjects(outCPU.get(0), outGPU.get(0));
				}
			}
		}
	}

	/**
	 * Assert that the two objects are equal. Supported types are Boolean, Integer, String, Double and Matrix
	 * @param expected
	 * @param actual
	 */
	protected void assertEqualObjects(Object expected, Object actual) {
		Assert.assertEquals(expected.getClass(), actual.getClass());

		if (expected instanceof Boolean) {
			Assert.assertEquals(((Boolean) expected).booleanValue(), ((Boolean) actual).booleanValue());
		} else if (expected instanceof Double) {
			Assert.assertEquals(((Double) expected).doubleValue(), ((Double) actual).doubleValue(), THRESHOLD);
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
