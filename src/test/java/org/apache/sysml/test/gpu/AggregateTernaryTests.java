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
 * Tests Ternary Aggregate ops
 */
public class AggregateTernaryTests extends UnaryOpTestsBase {

	private final static String TEST_NAME = "AggregateTernaryTests";

	@Override
	public void setUp() {
		super.setUp();
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_DIR, TEST_NAME);
		getAndLoadTestConfiguration(TEST_NAME);
	}
	
	@Test
	public void ternaryAgg1() {
		testTernaryUnaryOpMatrixOutput("out = sum(in1*in2*in3)", "gpu_tak+*", "in1", "in2", "in3",  "out", 30, 40, 0.9);
	}
	@Test
	public void ternaryAgg2() {
		testTernaryUnaryOpMatrixOutput("out = colSums(in1*in2*in3)", "gpu_tack+*", "in1", "in2", "in3",  "out", 30, 40, 0.9);
	}
	
	@Test
	public void ternaryAgg3() {
		testTernaryUnaryOpMatrixOutput("out = sum(in1*in2*in3)", "gpu_tak+*", "in1", "in2", "in3",  "out", 30, 40, 0.2);
	}
	@Test
	public void ternaryAgg4() {
		testTernaryUnaryOpMatrixOutput("out = colSums(in1*in2*in3)", "gpu_tack+*", "in1", "in2", "in3",  "out", 30, 40, 0.2);
	}
}
