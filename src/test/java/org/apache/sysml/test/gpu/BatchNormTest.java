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
	public void testBatchNormForwardTest1() {
		testBatchNormForward("test", 5, 3);
	}
	
	@Test
	public void testBatchNormForwardTrain1() {
		testBatchNormForward("train", 5, 3);
	}
	
	@Test
	public void testBatchNormForwardTest2() {
		testBatchNormForward("test", 5, 1);
	}
	
	@Test
	public void testBatchNormForwardTrain2() {
		testBatchNormForward("train", 5, 1);
	}
	
	@Test
	public void testBatchNormForwardTest3() {
		testBatchNormForward("test", 1, 3);
	}
	
	@Test
	public void testBatchNormForwardTrain3() {
		testBatchNormForward("train", 1, 3);
	}
	
	@Test
	public void testBatchNormForwardTest4() {
		testBatchNormForward("test", 1, 1);
	}
	
	@Test
	public void testBatchNormForwardTrain4() {
		testBatchNormForward("train", 1, 1);
	}
	
	@Test
	public void testBatchNormBackwardTest1() {
		testBatchNormBackward(1, 1);
	}
	
	@Test
	public void testBatchNormBackwardTest2() {
		testBatchNormBackward(1, 3);
	}
	
	@Test
	public void testBatchNormBackwardTest3() {
		testBatchNormBackward(5, 1);
	}
	
	@Test
	public void testBatchNormBackwardTest4() {
		testBatchNormBackward(5, 3);
	}
	
	private String addSmallConst(String scriptStr, List<String> outputs) {
		String ret = scriptStr;
		for(String var : outputs) {
			ret += " " + var + " = " + var + " + 1e-5; ";
		}
		return ret;
	}
	
	private void testBatchNormForward(String mode, int numImages, int numChannels) {
		int imgSize = 32;
		double sparsity = 0.9;
		String scriptStr = "source(\"nn/layers/batch_norm2d.dml\") as batch_norm2d;\n "
				+ "[output, ema_mean_upd, ema_var_upd, cache_mean, cache_var] = batch_norm2d::forward(x, gamma, beta, " + numChannels + ", " + 
				imgSize + ", " + imgSize + ", \"" + mode + "\", ema_mean, ema_var, 0.9, 1e-3); "
				+ "output = output + 1e-5; ema_mean_upd = ema_mean_upd + 1e-5; ema_var_upd = ema_var_upd + 1e-5; "
				+ "cache_mean =  cache_mean + 1e-5; cache_var = cache_var + 1e-5";
		HashMap<String, Object> inputs = new HashMap<>();
		inputs.put("x", generateInputMatrix(spark, numImages, numChannels*imgSize*imgSize, 0, 10, sparsity, seed));
        inputs.put("beta", generateInputMatrix(spark, numChannels, 1, 0, 5, sparsity, seed));
        inputs.put("gamma", generateInputMatrix(spark, numChannels, 1, 0, 5, sparsity, seed));
        inputs.put("ema_mean", generateInputMatrix(spark, numChannels, 1, 3, 7, sparsity, seed));
        inputs.put("ema_var", generateInputMatrix(spark, numChannels, 1, 0, 2, sparsity, seed));
		List<String> outputs = Arrays.asList("output", "ema_mean_upd", "ema_var_upd", "cache_mean", "cache_var");
		scriptStr = addSmallConst(scriptStr, outputs);
		List<Object> outCPU = runOnCPU(spark, scriptStr, inputs, outputs);
		List<Object> outGPU = runOnGPU(spark, scriptStr, inputs, outputs);
		if(mode.equals("test")) {
			assertHeavyHitterPresent("gpu_batch_norm2d_test");
			for(int i = 0; i < outputs.size(); i++) {
				assertEqualObjects(outCPU.get(i), outGPU.get(i));
			}
		}
		else {
			assertHeavyHitterPresent("gpu_batch_norm2d_train");
			assertEqualObjects(outCPU.get(0), outGPU.get(0));
			// Account for CuDNN's variation in output
            assertEqualObjects(outCPU.get(1), outGPU.get(1), 1e-3);
            assertEqualObjects(outCPU.get(2), outGPU.get(2), 1e-3);
            assertEqualObjects(outCPU.get(3), outGPU.get(3));
            assertEqualObjects(outCPU.get(4), outGPU.get(4));
		}
	}
	
	private void testBatchNormBackward(int numImages, int numChannels) {
		int imgSize = 32;
		double sparsity = 0.9;
		String scriptStr = "source(\"nn/layers/batch_norm2d.dml\") as batch_norm2d;\n "
				+ "[output, ema_mean, ema_var, cache_mean, cache_inv_var] = batch_norm2d::forward(x, gamma, beta, " + numChannels 
				+ ", " + imgSize + ", " + imgSize + ", \"train\", ema_mean, ema_var, 0.9, 1e-3)"
				+ "[dX, dgamma, dbeta] = batch_norm2d::backward(dout, cache_mean, cache_inv_var, x, gamma, "
				+ numChannels + "," + imgSize + "," + imgSize + ", 1e-3)";
		HashMap<String, Object> inputs = new HashMap<>();
		inputs.put("x", generateInputMatrix(spark, numImages, numChannels*imgSize*imgSize, 0, 10, sparsity, seed));
        inputs.put("dout", generateInputMatrix(spark, numImages, numChannels*imgSize*imgSize, 4, 9, sparsity, seed));
        inputs.put("beta", generateInputMatrix(spark, numChannels, 1, 0, 5, sparsity, seed));
        inputs.put("gamma", generateInputMatrix(spark, numChannels, 1, 0, 5, sparsity, seed));
        inputs.put("ema_mean", generateInputMatrix(spark, numChannels, 1, 3, 7, sparsity, seed));
        inputs.put("ema_var", generateInputMatrix(spark, numChannels, 1, 0, 2, sparsity, seed));
		List<String> outputs = Arrays.asList("dX", "dgamma", "dbeta");
		scriptStr = addSmallConst(scriptStr, outputs);
		List<Object> outCPU = runOnCPU(spark, scriptStr, inputs, outputs);
		List<Object> outGPU = runOnGPU(spark, scriptStr, inputs, outputs);
		//System.out.println("Backward:" + org.apache.sysml.utils.Statistics.getCPHeavyHitterOpCodes());
		assertHeavyHitterPresent("gpu_batch_norm2d_backward");
		assertEqualObjects(outCPU.get(0), outGPU.get(0));
        assertEqualObjects(outCPU.get(1), outGPU.get(1));
        assertEqualObjects(outCPU.get(2), outGPU.get(2));
	}
}
