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

package org.apache.sysds.test.gpu.nn;

import jcuda.CudaException;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaError;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Assume;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.HashMap;

public class ResNet18GPUTest extends AutomatedTestBase {

	private static final String TEST_NAME = "ResNet18GPU";
	private static final String TEST_DIR = "gpu/nn/";
	private static final String TEST_CLASS_DIR = TEST_DIR + ResNet18GPUTest.class.getSimpleName() + "/";

	private static final double eps = Math.pow(10, -10);

	@BeforeClass
	public static void checkGPU() {
		boolean gpuAvailable = false;
		try {
			// Ask JCuda to throw Java exceptions (much nicer than error codes)
			JCuda.setExceptionsEnabled(true);

			// How many devices does the runtime see?
			int[] devCount = {0};
			int status = JCuda.cudaGetDeviceCount(devCount);

			gpuAvailable = (status == cudaError.cudaSuccess) && (devCount[0] > 0);
		}
		catch(UnsatisfiedLinkError | CudaException ex) {
			// - native JCuda libs not on the class-path
			// - or they were built for the wrong CUDA version
			gpuAvailable = false;
		}

		Assume.assumeTrue("Skipping GPU test: no compatible CUDA device " + "or JCuda native libraries not available.",
			gpuAvailable);
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"R"}));
	}

	@Test
	public void testResnet18GPU() {
		runResNet18GPU();
	}

	private void runResNet18GPU() {

		TestConfiguration config = getTestConfiguration(TEST_NAME);
		loadTestConfiguration(config);

		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[] {"-stats", "-gpu", "-args", output("R")};

		runTest(true, false, null, -1);
		HashMap<MatrixValue.CellIndex, Double> out = readDMLMatrixFromOutputDir("R");

		double v1 = out.get(new MatrixValue.CellIndex(1, 1));
		double v2 = out.get(new MatrixValue.CellIndex(1, 2));
		double v3 = out.get(new MatrixValue.CellIndex(1, 3));

		Assert.assertTrue(v1 == 612 || v1 == 640);
		Assert.assertEquals(192, v2, 0.0);
		Assert.assertEquals(192, v3, 0.0);

		Assert.assertTrue(heavyHittersContainsAllString("gpu_conv2d_bias_add", "gpu_batch_norm2d", "gpu_softmax"));

	}
}
