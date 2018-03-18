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

import org.apache.sysml.test.utils.TestUtils;
import org.junit.Test;

/**
 * Tests Aggregate Unary ops
 */
public class AggregateUnaryOpTests extends UnaryOpTestsBase {

	private final static String TEST_NAME = "AggregateUnaryOpTests";

	@Override
	public void setUp() {
		super.setUp();
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_DIR, TEST_NAME);
		getAndLoadTestConfiguration(TEST_NAME);
	}

	@Test
	public void sum() {
		testSimpleUnaryOpMatrixOutput("sum", "gpu_uak+");
	}

	@Test
	public void colSums() {
		testSimpleUnaryOpMatrixOutput("colSums", "gpu_uack+");
	}
	
	@Test
	public void channelSums() {
		int[] rows = rowSizes;
		int[] C = new int[] { 2, 5, 10, 50 };
		int[] HW = new int[] { 10, 12, 21, 51 };
		double[] sparsities = this.sparsities;
		int seed = this.seed;	

		for (int k = 0; k < sparsities.length; k++) {
			double sparsity = sparsities[k];
			if(sparsity == 0)
				continue; // sparsity == 0 has been independently tested but it fails with non-informative mlcontext error
			for (int i = 0; i < rows.length; i++) {
				int row = rows[i];
				if(row == 1)
					continue; // Currently channel_sums rewrite is enabled only for row > 1
				for (int c : C) {
					if(c == 1)
						continue; // C == 1 will result in scalar value, but this case has been independently tested
					for (int hw : HW) {
						// Skip the case of a scalar unary op
						// System.out.println("Started channelSum test for " + row + " " + c + " " + hw + " " +  sparsity);
						String scriptStr = "out = rowSums(matrix(colSums(in1), rows=" + c + ", cols=" + hw + "));";
						testUnaryOpMatrixOutput(scriptStr, "gpu_channel_sums", "in1", "out", seed, row, c*hw, sparsity);
						// System.out.println("Ended channelSum test for " + row + " " + c + " " + hw + " " +  sparsity);
					}
				}
			}
		}
	}

	@Test
	public void rowSums() {
		testSimpleUnaryOpMatrixOutput("rowSums", "gpu_uark+");
	}

	@Test
	public void mult() {
		testSimpleUnaryOpMatrixOutput("prod", "gpu_ua*");
	}

	@Test
	public void mean() {
		testSimpleUnaryOpMatrixOutput("mean", "gpu_uamean");
	}

	@Test
	public void colMeans() {
		testSimpleUnaryOpMatrixOutput("colMeans", "gpu_uacmean");
	}

	@Test
	public void rowMeans() {
		testSimpleUnaryOpMatrixOutput("rowMeans", "gpu_uarmean");
	}

	@Test
	public void max() {
		testSimpleUnaryOpMatrixOutput("max", "gpu_uamax");
	}

	@Test
	public void rowMaxs() {
		testSimpleUnaryOpMatrixOutput("rowMaxs", "gpu_uarmax");
	}

	@Test
	public void colMaxs() {
		testSimpleUnaryOpMatrixOutput("colMaxs", "gpu_uacmax");
	}

	@Test
	public void min() {
		testSimpleUnaryOpMatrixOutput("min", "gpu_uamin");
	}

	@Test
	public void rowMins() {
		testSimpleUnaryOpMatrixOutput("rowMins", "gpu_uarmin");
	}

	@Test
	public void colMins() {
		testSimpleUnaryOpMatrixOutput("colMins", "gpu_uacmin");
	}

	@Test
	public void var() {
		testSimpleUnaryOpMatrixOutput("var", "gpu_uavar");
	}

	@Test
	public void colVars() {
		testSimpleUnaryOpMatrixOutput("colVars", "gpu_uacvar");
	}

	@Test
	public void rowVars() {
		testSimpleUnaryOpMatrixOutput("rowVars", "gpu_uarvar");
	}

	@Test
	public void sumsq() {
		testUnaryOpMatrixOutput("out = sum(in1*in1)", "gpu_uasqk+", "in1", "out");
	}

	@Test
	public void rowSumsqs() {
		testUnaryOpMatrixOutput("out = rowSums(in1*in1)", "gpu_uarsqk+", "in1", "out");
	}

	@Test
	public void colSumsqs() {
		testUnaryOpMatrixOutput("out = colSums(in1*in1)", "gpu_uacsqk+", "in1", "out");
	}
}
