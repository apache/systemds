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
import org.apache.sysds.runtime.data.SparseBlock;
import org.apache.sysds.runtime.functionobjects.ReduceCol;
import org.apache.sysds.runtime.instructions.cp.AggregateUnaryCPInstruction;
import org.apache.sysds.runtime.matrix.data.LibMatrixCountDistinct;
import org.apache.sysds.runtime.matrix.data.MatrixBlock;
import org.apache.sysds.runtime.matrix.operators.CountDistinctOperator;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import static org.junit.Assume.assumeTrue;

public abstract class CountDistinctRowOrColBase extends CountDistinctBase {

	@Override
	protected abstract String getTestClassDir();

	@Override
	protected abstract String getTestName();

	@Override
	protected abstract String getTestDir();

	protected abstract Types.Direction getDirection();

	private boolean runSparkTests = true;

	protected void addTestConfiguration() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(getTestName(), new TestConfiguration(getTestClassDir(), getTestName(), new String[] {"A"}));

		this.percentTolerance = 0.2;
	}

	public void setRunSparkTests(boolean runSparkTests) {
		this.runSparkTests = runSparkTests;
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
		assumeTrue(runSparkTests);

		Types.ExecType execType = Types.ExecType.SPARK;

		int actualDistinctCount = 10;
		int rows = 10000, cols = 1001;
		double sparsity = 0.1;
		double tolerance = actualDistinctCount * this.percentTolerance;

		countDistinctMatrixTest(getDirection(), actualDistinctCount, cols, rows, sparsity, execType, tolerance);
	}

	@Test
	public void testSparkDenseLargeMultiBlockAggregation() {
		assumeTrue(runSparkTests);

		Types.ExecType execType = Types.ExecType.SPARK;

		int actualDistinctCount = 10;
		int rows = 10000, cols = 1001;
		double sparsity = 0.9;
		double tolerance = actualDistinctCount * this.percentTolerance;

		countDistinctMatrixTest(getDirection(), actualDistinctCount, cols, rows, sparsity, execType, tolerance);
	}

	@Test
	public void testSparkSparseLargeNoneAggregation() {
		assumeTrue(runSparkTests);

		Types.ExecType execType = Types.ExecType.SPARK;

		int actualDistinctCount = 10;
		int rows = 10000, cols = 1000;
		double sparsity = 0.1;
		double tolerance = actualDistinctCount * this.percentTolerance;

		countDistinctMatrixTest(getDirection(), actualDistinctCount, cols, rows, sparsity, execType, tolerance);
	}

	@Test
	public void testSparkDenseLargeNoneAggregation() {
		assumeTrue(runSparkTests);

		Types.ExecType execType = Types.ExecType.SPARK;

		int actualDistinctCount = 10;
		int rows = 10000, cols = 1000;
		double sparsity = 0.9;
		double tolerance = actualDistinctCount * this.percentTolerance;

		countDistinctMatrixTest(getDirection(), actualDistinctCount, cols, rows, sparsity, execType, tolerance);
	}

	protected void testCPSparseLarge(SparseBlock.Type sparseBlockType, Types.Direction direction, int rows, int cols,
									 int actualDistinctCount, double sparsity, double tolerance) {
		MatrixBlock blkIn = TestUtils.round(TestUtils.generateTestMatrixBlock(rows, cols, 0, actualDistinctCount, sparsity, 7));
		if (!blkIn.isInSparseFormat()) {
			blkIn.denseToSparse(false);
		}
		blkIn = new MatrixBlock(blkIn, sparseBlockType, true);

		CountDistinctOperator op = new CountDistinctOperator(AggregateUnaryCPInstruction.AUType.COUNT_DISTINCT_APPROX,
				direction, ReduceCol.getReduceColFnObject());

		MatrixBlock blkOut = LibMatrixCountDistinct.estimateDistinctValues(blkIn, op);
		double[][] expectedMatrix = getExpectedMatrixRowOrCol(direction, cols, rows, actualDistinctCount);

		TestUtils.compareMatrices(expectedMatrix, blkOut, tolerance, "");
	}
}
