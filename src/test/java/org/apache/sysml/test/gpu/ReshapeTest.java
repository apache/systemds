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
import org.apache.sysml.test.utils.TestUtils;
import org.junit.Test;

/**
 * Tests gpu reshape
 */
public class ReshapeTest extends GPUTests {

	private final static String TEST_NAME = "ReshapeTests";
	private final int seed = 42;

	@Override
	public void setUp() {
		super.setUp();
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_DIR, TEST_NAME);
		getAndLoadTestConfiguration(TEST_NAME);
	}

	@Test
	public void testDenseReshape1() {
		testReshape(1, 10, 10, 1, true, 0.9);
	}
	
	@Test
	public void testDenseReshape2() {
		testReshape(1, 10, 10, 1, false, 0.9);
	}
	
	@Test
	public void testDenseReshape5() {
		testReshape(10, 3, 3, 10, true, 0.9);
	}
	
	@Test
	public void testDenseReshape6() {
		testReshape(10, 3, 3, 10, false, 0.9);
	}
	
	@Test
	public void testDenseReshape3() {
		testReshape(10, 3, 15, 2, true, 0.9);
	}
	
	@Test
	public void testDenseReshape4() {
		testReshape(10, 3, 15, 2, false, 0.9);
	}
	
	@Test
	public void testSparseReshape7() {
		testReshape(10, 3, 15, 2, true, 0.1);
	}
	
	@Test
	public void testSparseReshape8() {
		testReshape(10, 3, 15, 2, false, 0.1);
	}
	
	private void testReshape(int inRows, int inCols, int outRows, int outCols, boolean byrow, double sparsity) {
		System.out.println("Starting testReshape:" + inRows + " " + inCols + " " + outRows + " " + outCols + " " + byrow + " " + sparsity);
		String scriptStr = "output = matrix(x, rows=" + outRows + ", cols=" + outCols +  ", byrow=" +  (byrow ? "TRUE" : "FALSE") + ");" ;
		HashMap<String, Object> inputs = new HashMap<>();
		inputs.put("x", generateInputMatrix(spark, inRows, inCols, 0, 10, sparsity, seed));
		List<String> outputs = Arrays.asList("output");
		List<Object> outCPU = runOnCPU(spark, scriptStr, inputs, outputs);
		List<Object> outGPU = runOnGPU(spark, scriptStr, inputs, outputs);
		assertHeavyHitterPresent("gpu_rshape");
		assertEqualObjects(outCPU.get(0), outGPU.get(0));
	}
}
