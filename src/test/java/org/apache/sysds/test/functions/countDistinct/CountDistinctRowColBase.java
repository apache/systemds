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

package org.apache.sysds.test.functions.countDistinct;

import org.apache.sysds.common.Types.ExecType;
import org.junit.Test;

public abstract class CountDistinctRowColBase extends CountDistinctBase {
	@Test
	public void testCPDenseSmall() {
		ExecType ex = ExecType.CP;
		double tolerance = baseTolerance + 50 * percentTolerance;
		countDistinctScalarTest(50, 50, 50, 1.0, ex, tolerance);
	}

	@Test
	public void testSparkDenseSmall() {
		ExecType ex = ExecType.SPARK;
		double tolerance = baseTolerance + 50 * percentTolerance;
		countDistinctScalarTest(50, 50, 50, 1.0, ex, tolerance);
	}

	@Test
	public void testCPDenseLarge() {
		ExecType ex = ExecType.CP;
		double tolerance = baseTolerance + 800 * percentTolerance;
		countDistinctScalarTest(800, 1000, 1000, 1.0, ex, tolerance);
	}

	@Test
	public void testSparkDenseLarge() {
		ExecType ex = ExecType.SPARK;
		double tolerance = baseTolerance + 800 * percentTolerance;
		countDistinctScalarTest(800, 1000, 1000, 1.0, ex, tolerance);
	}

	@Test
	public void testCPDenseXLarge() {
		ExecType ex = ExecType.CP;
		double tolerance = baseTolerance + 1723 * percentTolerance;
		countDistinctScalarTest(1723, 5000, 2000, 1.0, ex, tolerance);
	}

	@Test
	public void testCPDense1Unique() {
		ExecType ex = ExecType.CP;
		double tolerance = 0.00001;
		countDistinctScalarTest(1, 100, 1000, 1.0, ex, tolerance);
	}

	@Test
	public void testCPDense2Unique() {
		ExecType ex = ExecType.CP;
		double tolerance = 0.00001;
		countDistinctScalarTest(2, 100, 1000, 1.0, ex, tolerance);
	}

	@Test
	public void testCPDense120Unique() {
		ExecType ex = ExecType.CP;
		double tolerance = 0.00001 + 120 * percentTolerance;
		countDistinctScalarTest(120, 100, 1000, 1.0, ex, tolerance);
	}

	@Test
	public void testCPSparse500Unique() {
		ExecType ex = ExecType.CP;
		double tolerance = 0.00001 + 500 * percentTolerance;
		countDistinctScalarTest(500, 100, 640000, 0.1, ex, tolerance);
	}

	@Test
	public void testCPSparse120Unique() {
		ExecType ex = ExecType.CP;
		double tolerance = 0.00001 + 120 * percentTolerance;
		countDistinctScalarTest(120, 100, 64000, 0.1, ex, tolerance);
	}
}
