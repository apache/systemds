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
 * Tests channel sums rewrite
 */
public class ChannelSumsTest extends GPUTests {

	private final static String TEST_NAME = "ChannelSumsTest";
	private final int seed = 42;

	@Override
	public void setUp() {
		super.setUp();
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_DIR, TEST_NAME);
		getAndLoadTestConfiguration(TEST_NAME);
	}

	@Test
	public void testChannelSumsTest() {
		int imgSize = 32; 
		int numChannels = 10;
		double sparsity = 0.9;
		String scriptStr = "output = rowSums(matrix(colSums(x), rows=" + numChannels + ", cols=" + imgSize*imgSize + "));";
		HashMap<String, Object> inputs = new HashMap<>();
		inputs.put("x", generateInputMatrix(spark, 32, numChannels*imgSize*imgSize, 0, 100, sparsity, seed));
		List<String> outputs = Arrays.asList("output");
		List<Object> outCPU = runOnCPU(spark, scriptStr, inputs, outputs);
		List<Object> outGPU = runOnGPU(spark, scriptStr, inputs, outputs);
		assertHeavyHitterPresent("gpu_channel_sums");
		for(int i = 0; i < outputs.size(); i++) {
			assertEqualObjects(outCPU.get(i), outGPU.get(i));
		}
	}
}
