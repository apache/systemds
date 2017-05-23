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
 * Tests Aggregate Unary ops
 */
public class AggregateUnaryOpTests extends UnaryOpTestsBase {

	private final static String TEST_NAME = "AggregateUnaryOpTests";

	@Override public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_DIR, TEST_NAME);
		getAndLoadTestConfiguration(TEST_NAME);
	}

	@Test public void sum() {
		testUnaryOpMatrixOutput("sum", "gpu_uak+", unaryOpRowSizes, unaryOpColSizes, unaryOpSparsities, unaryOpSeed);
	}

	@Test public void colSums() {
		testUnaryOpMatrixOutput("colSums", "gpu_uack+", unaryOpRowSizes, unaryOpColSizes, unaryOpSparsities,
				unaryOpSeed);
	}

	@Test public void rowSums() {
		testUnaryOpMatrixOutput("rowSums", "gpu_uark+", unaryOpRowSizes, unaryOpColSizes, unaryOpSparsities,
				unaryOpSeed);
	}

	@Test public void mult() {
		testUnaryOpMatrixOutput("prod", "gpu_ua*", unaryOpRowSizes, unaryOpColSizes, unaryOpSparsities, unaryOpSeed);
	}

	@Test public void mean() {
		testUnaryOpMatrixOutput("mean", "gpu_uamean", unaryOpRowSizes, unaryOpColSizes, unaryOpSparsities, unaryOpSeed);
	}

	@Test public void colMeans() {
		testUnaryOpMatrixOutput("colMeans", "gpu_uacmean", unaryOpRowSizes, unaryOpColSizes, unaryOpSparsities,
				unaryOpSeed);
	}

	@Test public void rowMeans() {
		testUnaryOpMatrixOutput("rowMeans", "gpu_uarmean", unaryOpRowSizes, unaryOpColSizes, unaryOpSparsities,
				unaryOpSeed);
	}

	@Test public void max() {
		testUnaryOpMatrixOutput("max", "gpu_uamax", unaryOpRowSizes, unaryOpColSizes, unaryOpSparsities, unaryOpSeed);
	}

	@Test public void rowMaxs() {
		testUnaryOpMatrixOutput("rowMaxs", "gpu_uarmax", unaryOpRowSizes, unaryOpColSizes, unaryOpSparsities,
				unaryOpSeed);
	}

	@Test public void colMaxs() {
		testUnaryOpMatrixOutput("colMaxs", "gpu_uacmax", unaryOpRowSizes, unaryOpColSizes, unaryOpSparsities,
				unaryOpSeed);
	}

	@Test public void min() {
		testUnaryOpMatrixOutput("min", "gpu_uamin", unaryOpRowSizes, unaryOpColSizes, unaryOpSparsities, unaryOpSeed);
	}

	@Test public void rowMins() {
		testUnaryOpMatrixOutput("rowMins", "gpu_uarmin", unaryOpRowSizes, unaryOpColSizes, unaryOpSparsities,
				unaryOpSeed);
	}

	@Test public void colMins() {
		testUnaryOpMatrixOutput("colMins", "gpu_uacmin", unaryOpRowSizes, unaryOpColSizes, unaryOpSparsities,
				unaryOpSeed);
	}

	@Test public void var() {
		testUnaryOpMatrixOutput("var", "gpu_uavar", unaryOpRowSizes, unaryOpColSizes, unaryOpSparsities, unaryOpSeed);
	}

	@Test public void colVars() {
		testUnaryOpMatrixOutput("colVars", "gpu_uacvar", unaryOpRowSizes, unaryOpColSizes, unaryOpSparsities,
				unaryOpSeed);
	}

	@Test public void rowVars() {
		testUnaryOpMatrixOutput("rowVars", "gpu_uarvar", unaryOpRowSizes, unaryOpColSizes, unaryOpSparsities,
				unaryOpSeed);
	}
}
