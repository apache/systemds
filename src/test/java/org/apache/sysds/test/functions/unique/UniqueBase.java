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

package org.apache.sysds.test.functions.unique;

import org.apache.sysds.common.Types;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.stats.InfrastructureAnalyzer;

public abstract class UniqueBase extends AutomatedTestBase {

	protected abstract String getTestName();

	protected abstract String getTestDir();

	protected abstract String getTestClassDir();

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(getTestName(), new TestConfiguration(getTestClassDir(), getTestName(), new String[] {"A"}));
	}

	protected void uniqueTest(double[][] inputMatrix, double[][] expectedMatrix,
							Types.ExecType instType, double epsilon) {
		uniqueTest(inputMatrix, expectedMatrix, instType, epsilon, -1, false);
	}

	/**
	 * Runs the unique script and compares the result row by row, in order. Use this where the expected output is
	 * unambiguous, i.e. where every row or column of the result holds a single value and the iteration order of the
	 * internal hash sets cannot affect the outcome.
	 */
	protected void uniqueTestOrdered(double[][] inputMatrix, double[][] expectedMatrix, Types.ExecType instType,
		double epsilon) {
		uniqueTest(inputMatrix, expectedMatrix, instType, epsilon, -1, true);
	}

	/**
	 * Runs the unique script with a temporarily reduced local memory budget. The multi-threaded unique implementation
	 * derives its budget for thread-local deduplication from the local memory, so this makes the memory-aware batched
	 * path reachable from an end-to-end script run.
	 *
	 * @param localMaxMemory local memory budget in bytes, or -1 to keep the current one
	 */
	protected void uniqueTestConstrainedMemory(double[][] inputMatrix, double[][] expectedMatrix,
		Types.ExecType instType, double epsilon, long localMaxMemory) {
		uniqueTest(inputMatrix, expectedMatrix, instType, epsilon, localMaxMemory, false);
	}

	private void uniqueTest(double[][] inputMatrix, double[][] expectedMatrix, Types.ExecType instType, double epsilon,
		long localMaxMemory, boolean orderedComparison) {
		Types.ExecMode platformOld = setExecMode(instType);
		long localMaxMemoryOld = InfrastructureAnalyzer.getLocalMaxMemory();
		try {
			if(localMaxMemory > 0)
				InfrastructureAnalyzer.setLocalMaxMemory(localMaxMemory);

			loadTestConfiguration(getTestConfiguration(getTestName()));
			String HOME = SCRIPT_DIR + getTestDir();
			fullDMLScriptName = HOME + getTestName() + ".dml";
			programArgs = new String[]{"-args", input("I"), output("A")};

			writeInputMatrixWithMTD("I", inputMatrix, true);

			runTest(true, false, null, -1);
			writeExpectedMatrix("A", expectedMatrix);

			if(orderedComparison)
				compareResults(epsilon);
			else
				compareResultsRowsOutOfOrder(epsilon);
		}
		finally {
			InfrastructureAnalyzer.setLocalMaxMemory(localMaxMemoryOld);
			rtplatform = platformOld;
		}
	}
}
