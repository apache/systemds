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
 * Unit tests for Unary ops on GPU
 */
public class UnaryOpTests extends UnaryOpTestsBase {

	private final static String TEST_NAME = "UnaryOpTests";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_DIR, TEST_NAME);
		getAndLoadTestConfiguration(TEST_NAME);
	}

	@Test
	public void testSin() throws Exception {
		testSimpleUnaryOpMatrixOutput("sin", "gpu_sin");
	}

	@Test
	public void testCos() throws Exception {
		testSimpleUnaryOpMatrixOutput("cos", "gpu_cos");
	}

	@Test
	public void testTan() throws Exception {
		testSimpleUnaryOpMatrixOutput("tan", "gpu_tan");
	}

	@Test
	public void testAsin() throws Exception {
		testSimpleUnaryOpMatrixOutput("asin", "gpu_asin");
	}

	@Test
	public void testAcos() throws Exception {
		testSimpleUnaryOpMatrixOutput("acos", "gpu_acos");
	}

	@Test
	public void testAtan() throws Exception {
		testSimpleUnaryOpMatrixOutput("atan", "gpu_atan");
	}

	@Test
	public void testExp() throws Exception {
		testSimpleUnaryOpMatrixOutput("exp", "gpu_exp");
	}

	@Test
	public void testLog() throws Exception {
		testSimpleUnaryOpMatrixOutput("log", "gpu_log");
	}

	@Test
	public void testSqrt() throws Exception {
		testSimpleUnaryOpMatrixOutput("sqrt", "gpu_sqrt");
	}

	@Test
	public void testAbs() throws Exception {
		testSimpleUnaryOpMatrixOutput("abs", "gpu_abs");
	}

	@Test
	public void testRound() throws Exception {
		testSimpleUnaryOpMatrixOutput("round", "gpu_round");
	}

	@Test
	public void testFloor() throws Exception {
		testSimpleUnaryOpMatrixOutput("sqrt", "gpu_floor");
	}

	@Test
	public void testCeil() throws Exception {
		testSimpleUnaryOpMatrixOutput("ceil", "gpu_ceil");
	}

	@Test
	public void testSign() throws Exception {
		testSimpleUnaryOpMatrixOutput("sign", "gpu_sign");
	}

	@Test
	public void testSelp() throws Exception {
		testUnaryOpMatrixOutput("out = max(in1, 0)", "gpu_selp", "in1", "out");
	}
}