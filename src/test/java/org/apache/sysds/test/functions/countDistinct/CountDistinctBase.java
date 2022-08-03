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
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

import static org.junit.Assert.assertTrue;

public abstract class CountDistinctBase extends AutomatedTestBase {
	protected double percentTolerance = 0.0;
	protected double baseTolerance = 0.0001;

	protected abstract String getTestClassDir();

	protected abstract String getTestName();

	protected abstract String getTestDir();

	protected void addTestConfiguration() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(getTestName(),
			new TestConfiguration(getTestClassDir(), getTestName(), new String[] {"A.scalar"}));
	}

	@Override
	public abstract void setUp();

	public void countDistinctScalarTest(long numberDistinct, int cols, int rows, double sparsity,
		Types.ExecType instType, double tolerance) {
		countDistinctTest(Types.Direction.RowCol, numberDistinct, cols, rows, sparsity, instType, tolerance);
	}

	public void countDistinctMatrixTest(Types.Direction dir, long numberDistinct, int cols, int rows, double sparsity,
		Types.ExecType instType, double tolerance) {
		countDistinctTest(dir, numberDistinct, cols, rows, sparsity, instType, tolerance);
	}

	public void countDistinctTest(Types.Direction dir, long numberDistinct, int cols, int rows, double sparsity,
		Types.ExecType instType, double tolerance) {

		Types.ExecMode platformOld = setExecMode(instType);
		try {
			loadTestConfiguration(getTestConfiguration(getTestName()));
			String HOME = SCRIPT_DIR + getTestDir();
			fullDMLScriptName = HOME + getTestName() + ".dml";
			String outputPath = output("A");

			programArgs = new String[] {"-args", String.valueOf(numberDistinct), String.valueOf(rows),
				String.valueOf(cols), String.valueOf(sparsity), outputPath};

			runTest(true, false, null, -1);

			if(dir.isRowCol()) {
				writeExpectedScalar("A", numberDistinct);
			}
			else {
				double[][] expectedMatrix = getExpectedMatrixRowOrCol(dir, cols, rows, numberDistinct);
				writeExpectedMatrix("A", expectedMatrix);
			}
			compareResults(tolerance);
		}
		catch(Exception e) {
			e.printStackTrace();
			assertTrue("Exception in execution: " + e.getMessage(), false);
		}
		finally {
			rtplatform = platformOld;
		}
	}

	protected double[][] getExpectedMatrixRowOrCol(Types.Direction dir, int cols, int rows, long expectedValue) {
		double[][] expectedResult;
		if(dir.isRow()) {
			expectedResult = new double[rows][1];
			for(int i = 0; i < rows; ++i) {
				expectedResult[i][0] = expectedValue;
			}
		}
		else {
			expectedResult = new double[1][cols];
			for(int i = 0; i < cols; ++i) {
				expectedResult[0][i] = expectedValue;
			}
		}

		return expectedResult;
	}
}
