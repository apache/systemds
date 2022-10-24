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

package org.apache.sysds.test.functions.countDistinctApprox;

import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.test.functions.countDistinct.CountDistinctRowColBase;
import org.junit.Test;

public class CountDistinctApproxRowColParameterized extends CountDistinctRowColBase {

	private final static String TEST_NAME = "countDistinctApproxRowColParameterized";
	private final static String TEST_DIR = "functions/countDistinctApprox/";
	private final static String TEST_CLASS_DIR = TEST_DIR + CountDistinctApproxRowColParameterized.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		super.addTestConfiguration();
		super.percentTolerance = 0.2;
	}

	@Test
	public void testCPSparseLarge() {
		ExecType ex = ExecType.CP;
		double tolerance = 9000 * percentTolerance;
		countDistinctScalarTest(9000, 10000, 5000, 0.1, ex, tolerance);
	}

	@Test
	public void testSparkSparseLarge() {
		ExecType ex = ExecType.SPARK;
		double tolerance = 9000 * percentTolerance;
		countDistinctScalarTest(9000, 10000, 5000, 0.1, ex, tolerance);
	}

	@Test
	public void testCPSparseSmall() {
		ExecType ex = ExecType.CP;
		double tolerance = 9000 * percentTolerance;
		countDistinctScalarTest(9000, 999, 999, 0.1, ex, tolerance);
	}

	@Test
	public void testSparkSparseSmall() {
		ExecType ex = ExecType.SPARK;
		double tolerance = 9000 * percentTolerance;
		countDistinctScalarTest(9000, 999, 999, 0.1, ex, tolerance);
	}

	@Test
	public void testCPDenseXSmall() {
		ExecType ex = ExecType.CP;
		double tolerance = 5 * percentTolerance;
		countDistinctScalarTest(5, 5, 10, 1.0, ex, tolerance);
	}

	@Test
	public void testSparkDenseXSmall() {
		ExecType ex = ExecType.SPARK;
		double tolerance = 5 * percentTolerance;
		countDistinctScalarTest(5, 10, 5, 1.0, ex, tolerance);
	}

	@Test
	public void testCPEmpty() {
		ExecType ex = ExecType.CP;
		countDistinctScalarTest(1, 0, 0, 0.1, ex, 0);
	}

	@Test
	public void testSparkEmpty() {
		ExecType ex = ExecType.SPARK;
		countDistinctScalarTest(1, 0, 0, 0.1, ex, 0);
	}

	@Test
	public void testCPSingleValue() {
		ExecType ex = ExecType.CP;
		countDistinctScalarTest(1, 1, 1, 1.0, ex, 0);
	}

	@Test
	public void testSparkSingleValue() {
		ExecType ex = ExecType.SPARK;
		countDistinctScalarTest(1, 1, 1, 1.0, ex, 0);
	}

	// Corresponding execType=SPARK tests for CP tests in base class
	//
	@Test
	public void testSparkDense1Unique() {
		ExecType ex = ExecType.SPARK;
		double tolerance = 0.00001;
		countDistinctScalarTest(1, 100, 1000, 1.0, ex, tolerance);
	}

	@Test
	public void testSparkDense2Unique() {
		ExecType ex = ExecType.SPARK;
		double tolerance = 0.00001;
		countDistinctScalarTest(2, 100, 1000, 1.0, ex, tolerance);
	}

	@Test
	public void testSparkDense120Unique() {
		ExecType ex = ExecType.SPARK;
		double tolerance = 0.00001 + 120 * percentTolerance;
		countDistinctScalarTest(120, 100, 1000, 1.0, ex, tolerance);
	}

	@Override
	protected String getTestClassDir() {
		return TEST_CLASS_DIR;
	}

	@Override
	protected String getTestName() {
		return TEST_NAME;
	}

	@Override
	protected String getTestDir() {
		return TEST_DIR;
	}
}
