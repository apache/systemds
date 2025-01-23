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

package org.apache.sysds.test.applications.nn;

import org.junit.Test;

public class NNComponentTest extends TestFolder {

	@Test
	public void batch_norm1d() {
		run("batch_norm1d.dml");
	}

	@Test
	public void batch_norm2d() {
		run("batch_norm2d.dml");
	}

	@Test
	public void conv2d() {
		run("conv2d.dml");
	}

	@Test
	public void conv2d_depthwise() {
		run("conv2d_depthwise.dml");
	}

	@Test
	public void conv2d_transpose() {
		run("conv2d_transpose.dml");
	}

	@Test
	public void conv2d_transpose_depthwise() {
		run("conv2d_transpose_depthwise.dml");
	}

	@Test
	public void cross_entropy_loss() {
		run("cross_entropy_loss.dml");
	}

	@Test
	public void cross_entropy_loss2d() {
		run("cross_entropy_loss2d.dml");
	}

	@Test
	public void im2col() {
		run("im2col.dml");
	}

	@Test
	public void max_pool2d() {
		run("max_pool2d.dml");
	}

	@Test
	public void tanh() {
		run("tanh.dml");
	}

	@Test
	public void elu() {
		run("elu.dml");
	}

	@Test
	public void threshold() {
		run("threshold.dml");
	}

	@Test
	public void softmax2d() {
		run("softmax2d.dml");
	}

	@Test
	public void top_k() {
		run("top_k.dml");
	}

	@Test
	public void padding() {
		run("padding.dml");
	}

	@Test
	public void transpose_NCHW_to_CNHW() {
		run("transpose_NCHW_to_CNHW.dml");
	}

	@Test
	public void transpose_ABCD_to_ACBD() {
		run("transpose_ABCD_to_ACBD.dml");
	}

	@Test 
	public void logcosh(){
		run("logcosh.dml");
	}

	@Test
	public void resnet() {
		run("resnet_basic.dml");
		run("resnet_bottleneck.dml");
	}

	@Test
	public void gelu() {
		run("gelu.dml");
	}

	@Override
	protected void run(String name) {
		super.run("component/" + name);
	}

	@Override
	protected void run(String name, String[] var, Object[] val) {
		super.run("component/" + name, var, val);
	}
}
