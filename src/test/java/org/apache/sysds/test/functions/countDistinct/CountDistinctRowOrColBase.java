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

import org.apache.sysds.common.Types;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

public abstract class CountDistinctRowOrColBase extends CountDistinctBase {

	@Override
	protected abstract String getTestClassDir();

	@Override
	protected abstract String getTestName();

	@Override
	protected abstract String getTestDir();

	protected abstract Types.Direction getDirection();

	protected void addTestConfiguration() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(getTestName(), new TestConfiguration(getTestClassDir(), getTestName(), new String[] {"A"}));

		this.percentTolerance = 0.2;
	}

	@Test
	public void testCPSparseLarge() {
		Types.ExecType ex = Types.ExecType.CP;

		int actualDistinctCount = 10;
		int rows = 10000, cols = 1000;
		double sparsity = 0.1;
		double tolerance = actualDistinctCount * this.percentTolerance;

		countDistinctMatrixTest(getDirection(), actualDistinctCount, cols, rows, sparsity, ex, tolerance);
	}

	@Test
	public void testCPDenseLarge() {
		Types.ExecType ex = Types.ExecType.CP;

		int actualDistinctCount = 100;
		int rows = 10000, cols = 1000;
		double sparsity = 0.9;
		double tolerance = actualDistinctCount * this.percentTolerance;

		countDistinctMatrixTest(getDirection(), actualDistinctCount, cols, rows, sparsity, ex, tolerance);
	}

	@Test
	public void testCPSparseSmall() {
		Types.ExecType execType = Types.ExecType.CP;

		int actualDistinctCount = 10;
		int rows = 1000, cols = 1000;
		double sparsity = 0.1;
		double tolerance = actualDistinctCount * this.percentTolerance;

		countDistinctMatrixTest(getDirection(), actualDistinctCount, cols, rows, sparsity, execType, tolerance);
	}

	@Test
	public void testCPDenseSmall() {
		Types.ExecType execType = Types.ExecType.CP;

		int actualDistinctCount = 10;
		int rows = 1000, cols = 1000;
		double sparsity = 0.9;
		double tolerance = actualDistinctCount * this.percentTolerance;

		countDistinctMatrixTest(getDirection(), actualDistinctCount, cols, rows, sparsity, execType, tolerance);
	}

	@Test
	public void testSparkSparseLargeMultiBlockAggregation() {
		Types.ExecType execType = Types.ExecType.SPARK;

		int actualDistinctCount = 10;
		int rows = 10000, cols = 1001;
		double sparsity = 0.1;
		double tolerance = actualDistinctCount * this.percentTolerance;

		countDistinctMatrixTest(getDirection(), actualDistinctCount, cols, rows, sparsity, execType, tolerance);
	}

	@Test
	public void testSparkDenseLargeMultiBlockAggregation() {
		Types.ExecType execType = Types.ExecType.SPARK;

		int actualDistinctCount = 10;
		int rows = 10000, cols = 1001;
		double sparsity = 0.9;
		double tolerance = actualDistinctCount * this.percentTolerance;

		countDistinctMatrixTest(getDirection(), actualDistinctCount, cols, rows, sparsity, execType, tolerance);
	}

	@Test
	public void testSparkSparseLargeNoneAggregation() {
		Types.ExecType execType = Types.ExecType.SPARK;

		int actualDistinctCount = 10;
		int rows = 10000, cols = 1000;
		double sparsity = 0.1;
		double tolerance = actualDistinctCount * this.percentTolerance;

		countDistinctMatrixTest(getDirection(), actualDistinctCount, cols, rows, sparsity, execType, tolerance);
	}

	@Test
	public void testSparkDenseLargeNoneAggregation() {
		Types.ExecType execType = Types.ExecType.SPARK;

		int actualDistinctCount = 10;
		int rows = 10000, cols = 1000;
		double sparsity = 0.9;
		double tolerance = actualDistinctCount * this.percentTolerance;

		countDistinctMatrixTest(getDirection(), actualDistinctCount, cols, rows, sparsity, execType, tolerance);
	}
}
