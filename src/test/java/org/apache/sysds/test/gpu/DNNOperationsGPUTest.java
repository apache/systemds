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

package org.apache.sysds.test.gpu;

import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.functions.dnn.Conv1DTest;
import org.apache.sysds.test.functions.dnn.Conv2DTest;
import org.apache.sysds.test.functions.dnn.Conv2DBackwardTest;
import org.apache.sysds.test.functions.dnn.Conv2DBackwardDataTest;
import org.apache.sysds.test.functions.dnn.PoolTest;
import org.apache.sysds.test.functions.dnn.PoolBackwardTest;
import org.apache.sysds.test.functions.dnn.ReluBackwardTest;
import org.junit.Assert;
import org.junit.Test;

public class DNNOperationsGPUTest extends AutomatedTestBase {

	@Override
	public void setUp() {
		TEST_GPU = true;
		VERBOSE_STATS = true;
	}

	@Test
	public void Conv1DGPUTest() {
		Conv1DTest dmlTestCase = new Conv1DTest();
		dmlTestCase.setUpBase();
		dmlTestCase.setUp();
		dmlTestCase.testSimpleConv1DDenseSingleBatchSingleChannelSingleFilter();
		Assert.assertTrue(heavyHittersContainsString("gpu_conv2d"));
		dmlTestCase.testConv1DDense1();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_conv2d", "gpu_append"));
		dmlTestCase.testConv1DDense2();
		Assert.assertTrue(heavyHittersContainsString("gpu_conv2d"));
		dmlTestCase.testConv1DDense3();
		Assert.assertTrue(heavyHittersContainsString("gpu_conv2d"));
		dmlTestCase.testConv1DDense4();
		Assert.assertTrue(heavyHittersContainsString("gpu_conv2d"));
		dmlTestCase.testConv1DDense5();
		Assert.assertTrue(heavyHittersContainsString("gpu_conv2d"));
		dmlTestCase.testConv1DDense6();
		Assert.assertTrue(heavyHittersContainsString("gpu_conv2d"));
		dmlTestCase.testConv1DDense7();
		Assert.assertTrue(heavyHittersContainsString("gpu_conv2d"));
		dmlTestCase.testConv1DBackwardDataDense1();
		Assert.assertTrue(heavyHittersContainsString("gpu_conv2d_backward_data"));
		dmlTestCase.testConv1DBackwardFilterDense1();
		Assert.assertTrue(heavyHittersContainsString("gpu_conv2d_backward_filter"));
		dmlTestCase.testConv1DBackwardFilterDense2();
		Assert.assertTrue(heavyHittersContainsString("gpu_conv2d_backward_filter"));
	}

	@Test
	public void Conv2DGPUTest() {
		Conv2DTest dmlTestCase = new Conv2DTest();
		dmlTestCase.setUpBase();
		dmlTestCase.setUp();
		dmlTestCase.testConv2DDense1();
		Assert.assertTrue(heavyHittersContainsString("gpu_conv2d_bias_add"));
		dmlTestCase.testConv2DDense2();
		Assert.assertTrue(heavyHittersContainsString("gpu_conv2d_bias_add"));
		dmlTestCase.testConv2DDense3();
		Assert.assertTrue(heavyHittersContainsString("gpu_conv2d_bias_add"));
		dmlTestCase.testConv2DDense4();
		Assert.assertTrue(heavyHittersContainsString("gpu_conv2d_bias_add"));
		dmlTestCase.testConv2DDense5();
		Assert.assertTrue(heavyHittersContainsString("gpu_conv2d_bias_add"));
		dmlTestCase.testConv2DDense6();
		Assert.assertTrue(heavyHittersContainsString("gpu_conv2d_bias_add"));
		dmlTestCase.testConv2DDense7();
		Assert.assertTrue(heavyHittersContainsString("gpu_conv2d_bias_add"));
		dmlTestCase.testConv2DSparse1a();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_conv2d_bias_add", "gpu_*", "gpu_>"));
		dmlTestCase.testConv2DSparse2a();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_conv2d_bias_add", "gpu_*", "gpu_>"));
		dmlTestCase.testConv2DSparse3a();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_conv2d_bias_add", "gpu_*", "gpu_>"));
		dmlTestCase.testConv2DSparse4a();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_conv2d_bias_add", "gpu_*", "gpu_>"));
		dmlTestCase.testConv2DSparse5a();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_conv2d_bias_add", "gpu_*", "gpu_>"));
		dmlTestCase.testConv2DSparse6a();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_conv2d_bias_add", "gpu_*", "gpu_>"));
		dmlTestCase.testConv2DSparse7a();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_conv2d_bias_add", "gpu_*", "gpu_>"));
		dmlTestCase.testConv2DSparse1b();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_conv2d_bias_add", "gpu_*", "gpu_>"));
		dmlTestCase.testConv2DSparse2b();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_conv2d_bias_add", "gpu_*", "gpu_>"));
		dmlTestCase.testConv2DSparse3b();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_conv2d_bias_add", "gpu_*", "gpu_>"));
		dmlTestCase.testConv2DSparse4b();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_conv2d_bias_add", "gpu_*", "gpu_>"));
		dmlTestCase.testConv2DSparse5b();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_conv2d_bias_add", "gpu_*", "gpu_>"));
		dmlTestCase.testConv2DSparse6b();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_conv2d_bias_add", "gpu_*", "gpu_>"));
		dmlTestCase.testConv2DSparse7b();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_conv2d_bias_add", "gpu_*", "gpu_>"));
	}

	@Test
	public void Conv2DBackwardGPUTest() {
		Conv2DBackwardTest dmlTestCase = new Conv2DBackwardTest();
		dmlTestCase.setUpBase();
		dmlTestCase.setUp();
		dmlTestCase.testConv2DBackwardFilterDense1();
		Assert.assertTrue(heavyHittersContainsString("gpu_conv2d_backward_filter"));
		dmlTestCase.testConv2DBackwardFilterDense2();
		Assert.assertTrue(heavyHittersContainsString("gpu_conv2d_backward_filter"));
		dmlTestCase.testConv2DBackwardFilterDense3();
		Assert.assertTrue(heavyHittersContainsString("gpu_conv2d_backward_filter"));
		dmlTestCase.testConv2DBackwardFilterDense4();
		Assert.assertTrue(heavyHittersContainsString("gpu_conv2d_backward_filter"));
		dmlTestCase.testConv2DBackwardFilterDense5();
		Assert.assertTrue(heavyHittersContainsString("gpu_conv2d_backward_filter"));
		dmlTestCase.testConv2DBackwardFilterSparse1();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_conv2d_backward_filter", "gpu_>"));
		dmlTestCase.testConv2DBackwardFilterSparse2();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_conv2d_backward_filter", "gpu_>"));
		dmlTestCase.testConv2DBackwardFilterSparse3();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_conv2d_backward_filter", "gpu_>"));
		dmlTestCase.testConv2DBackwardFilterSparse4();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_conv2d_backward_filter", "gpu_>"));
		dmlTestCase.testConv2DBackwardFilterSparse5();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_conv2d_backward_filter", "gpu_>"));
		dmlTestCase.testConv2DBackwardFilterSparse6();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_conv2d_backward_filter", "gpu_>"));
		dmlTestCase.testConv2DBackwardFilterSparse7();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_conv2d_backward_filter", "gpu_>"));
		dmlTestCase.testConv2DBackwardFilterSparse8();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_conv2d_backward_filter", "gpu_>"));
		dmlTestCase.testConv2DBackwardFilterSparse9();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_conv2d_backward_filter", "gpu_>"));
		dmlTestCase.testConv2DBackwardFilterSparse10();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_conv2d_backward_filter", "gpu_>"));
		dmlTestCase.testConv2DBackwardFilterSparse11();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_conv2d_backward_filter", "gpu_>"));
		dmlTestCase.testConv2DBackwardFilterSparse12();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_conv2d_backward_filter", "gpu_>"));
		dmlTestCase.testConv2DBackwardFilterSparse13();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_conv2d_backward_filter", "gpu_>"));
		dmlTestCase.testConv2DBackwardFilterSparse14();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_conv2d_backward_filter", "gpu_>"));
		dmlTestCase.testConv2DBackwardFilterSparse15();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_conv2d_backward_filter", "gpu_>"));
		dmlTestCase.testConv2DBackwardFilterSparse16();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_conv2d_backward_filter", "gpu_>"));
		dmlTestCase.testConv2DBackwardFilterSparse17();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_conv2d_backward_filter", "gpu_>"));
	}

	@Test
	public void Conv2DBackwardDataGPUTest() {
		Conv2DBackwardDataTest dmlTestCase = new Conv2DBackwardDataTest();
		dmlTestCase.setUpBase();
		dmlTestCase.setUp();
		dmlTestCase.testConv2DBwdDataDense1();
		Assert.assertTrue(heavyHittersContainsString("gpu_conv2d_backward_data"));
		dmlTestCase.testConv2DDense2();
		Assert.assertTrue(heavyHittersContainsString("gpu_conv2d_backward_data"));
		dmlTestCase.testConv2DDense3();
		Assert.assertTrue(heavyHittersContainsString("gpu_conv2d_backward_data"));
		dmlTestCase.testConv2DBwdDataDense4();
		Assert.assertTrue(heavyHittersContainsString("gpu_conv2d_backward_data"));
		dmlTestCase.testConv2DBwdDataSparse1();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_conv2d_backward_data", "gpu_>"));
		dmlTestCase.testConv2DBwdDataSparse2();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_conv2d_backward_data"));
		dmlTestCase.testConv2DBwdDataSparse3();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_conv2d_backward_data", "gpu_>"));
		dmlTestCase.testConv2DBwdDataSparse4();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_conv2d_backward_data", "gpu_>"));
		dmlTestCase.testConv2DBwdDataSparse5();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_conv2d_backward_data", "gpu_>"));
		dmlTestCase.testConv2DBwdDataSparse6();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_conv2d_backward_data", "gpu_>"));
		dmlTestCase.testConv2DBwdDataSparse7();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_conv2d_backward_data", "gpu_>"));
	}

	@Test
	public void PoolGPUTest() {
		PoolTest dmlTestCase = new PoolTest();
		dmlTestCase.setUpBase();
		dmlTestCase.setUp();
		dmlTestCase.testMaxPool2DDense1();
		Assert.assertTrue(heavyHittersContainsString("gpu_maxpooling"));
		dmlTestCase.testMaxPool2DDense2();
		Assert.assertTrue(heavyHittersContainsString("gpu_maxpooling"));
		dmlTestCase.testMaxPool2DDense3();
		Assert.assertTrue(heavyHittersContainsString("gpu_maxpooling"));
		dmlTestCase.testMaxPool2DDense4();
		Assert.assertTrue(heavyHittersContainsString("gpu_maxpooling"));
		dmlTestCase.testMaxPool2DDense5();
		Assert.assertTrue(heavyHittersContainsString("gpu_maxpooling"));
		dmlTestCase.testMaxPool2DSparse1();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_maxpooling", "gpu_>"));
		dmlTestCase.testMaxPool2DSparse2();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_maxpooling", "gpu_>"));
		dmlTestCase.testMaxPool2DSparse3();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_maxpooling", "gpu_>"));
		dmlTestCase.testMaxPool2DSparse4();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_maxpooling", "gpu_>"));
		dmlTestCase.testMaxPool2DSparse5();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_maxpooling", "gpu_>"));
	}

	@Test
	public void PoolBackwardGPUTest() {
		PoolBackwardTest dmlTestCase = new PoolBackwardTest();
		dmlTestCase.setUpBase();
		dmlTestCase.setUp();
		dmlTestCase.testMaxPool2DBackwardDense1();
		Assert.assertTrue(heavyHittersContainsString("gpu_maxpooling_backward"));
		dmlTestCase.testMaxPool2DBackwardDense2();
		Assert.assertTrue(heavyHittersContainsString("gpu_maxpooling_backward"));
		dmlTestCase.testMaxPool2DBackwardDense3();
		Assert.assertTrue(heavyHittersContainsString("gpu_maxpooling_backward"));
		dmlTestCase.testMaxPool2DBackwardSparse1();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_maxpooling_backward", "gpu_>"));
		dmlTestCase.testMaxPool2DBackwardSparse2();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_maxpooling_backward", "gpu_>"));
		dmlTestCase.testMaxPool2DBackwardSparse3();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_maxpooling_backward", "gpu_>"));
		dmlTestCase.testMaxPool2DBackwardSparse4();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_maxpooling_backward", "gpu_>"));
		dmlTestCase.testMaxPool2DBackwardSparse5();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_maxpooling_backward", "gpu_>"));
		dmlTestCase.testMaxPool2DBackwardSparse6();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_maxpooling_backward", "gpu_>"));
		dmlTestCase.testMaxPool2DBackwardSparse7();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_maxpooling_backward", "gpu_>"));
		dmlTestCase.testMaxPool2DBackwardSparse8();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_maxpooling_backward", "gpu_>"));
		dmlTestCase.testMaxPool2DBackwardSparse9();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_maxpooling_backward", "gpu_>"));
		dmlTestCase.testMaxPool2DBackwardSparse10();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_maxpooling_backward", "gpu_>"));
		dmlTestCase.testMaxPool2DBackwardSparse11();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_maxpooling_backward", "gpu_>"));
		dmlTestCase.testMaxPool2DBackwardSparse12();
		Assert.assertTrue(heavyHittersContainsAllString("gpu_maxpooling_backward", "gpu_>"));
	}

	@Test
	public void ReluBackwardGPUTest() {
		ReluBackwardTest dmlTestCase = new ReluBackwardTest();
		dmlTestCase.setUpBase();
		dmlTestCase.setUp();
		dmlTestCase.testReluBackwardDense1();
		Assert.assertTrue(heavyHittersContainsString("gpu_relu_backward"));
		dmlTestCase.testReluBackwardDense2();
		Assert.assertTrue(heavyHittersContainsString("gpu_relu_backward"));
		dmlTestCase.testReluBackwardDense3();
		Assert.assertTrue(heavyHittersContainsString("gpu_relu_backward"));
	}

}
