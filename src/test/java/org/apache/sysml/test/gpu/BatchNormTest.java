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
 * Tests batchnorm rewrite
 */
public class BatchNormTest extends GPUTests {

	private final static String TEST_NAME = "BatchNormTests";
	private final int seed = 42;

	@Override
	public void setUp() {
		super.setUp();
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_DIR, TEST_NAME);
		getAndLoadTestConfiguration(TEST_NAME);
	}

	@Test
	public void testBatchNormForwardTest() {
		testBatchNormForward("test");
	}
	
	@Test
	public void testBatchNormForwardTrain() {
		testBatchNormForward("train");
	}
	
	private void testBatchNormForward(String mode) {
		int imgSize = 32; 
		int numChannels = 3;
		double sparsity = 0.9;
		String scriptStr = "source(\"nn/layers/batch_norm2d_old.dml\") as batch_norm2d_old;\n"
				+ "[output, ema_mean_upd, ema_var_upd, cache_mean, cache_var] = batch_norm2d_old::forward(x, gamma, beta, " + numChannels + ", " + imgSize + ", " + imgSize + ", \"" + mode + "\", ema_mean, ema_var, 0.9, 1e-3)";
		HashMap<String, Object> inputs = new HashMap<>();
		inputs.put("x", generateInputMatrix(spark, 32, numChannels*imgSize*imgSize, sparsity, seed));
		inputs.put("gamma", generateInputMatrix(spark, numChannels, 1, sparsity, seed));
		inputs.put("beta", generateInputMatrix(spark, numChannels, 1, sparsity, seed));
		inputs.put("ema_mean", generateInputMatrix(spark, numChannels, 1, sparsity, seed));
		inputs.put("ema_var", generateInputMatrix(spark, numChannels, 1, sparsity, seed));
		List<String> outputs = Arrays.asList("output", "ema_mean_upd", "ema_var_upd", "cache_mean", "cache_var");
		List<Object> outCPU = runOnCPU(spark, scriptStr, inputs, outputs);
		List<Object> outGPU = runOnGPU(spark, scriptStr, inputs, outputs);
		// assertHeavyHitterPresent("gpu_batch_norm_test");
		for(int i = 0; i < outputs.size(); i++) {
			assertEqualObjects(outCPU.get(i), outGPU.get(i));
		}
	}
}
