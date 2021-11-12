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

package org.apache.sysds.test.component.matrix;

import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.junit.Assert;
import org.junit.Test;

public class RandTest {
	@Test
	public void dense_0_1_Test() {
		checkRand(1000, 1000, 0.9, 0, 1, 7);
	}
	@Test
	public void dense_10_100_Test() {
		checkRand(1000, 1000, 0.9, 10, 100, 7);
	}
	@Test
	public void sparse_0_1_Test() {
		checkRand(10000, 10000, 0.01, 0, 1, 7);
	}
	@Test
	public void sparse_10_100_Test() {
		checkRand(10000, 10000, 0.01, 10, 100, 7);
	}
	@Test
	public void ultrasparse_0_1_Test() {
		checkRand(10000000, 100000, 1e-8, 0, 1, 7);
	}
	@Test
	public void ultrasparse_10_100_Test() {
		checkRand(10000000, 100000, 1e-8, 10, 100, 7);
	}
	private static void checkRand(int rows, int cols, double sparsity, double min, double max, int seed) {
		MatrixBlock tmp = MatrixBlock.randOperations(rows, cols, sparsity, min, max, "uniform", seed);
		double actual = tmp.sum();
		double expected = (min + (max-min)/2)
			* OptimizerUtils.getNnz(rows, cols, sparsity);
		Assert.assertEquals(expected, actual, expected * 0.01); //1% range
	}
}
