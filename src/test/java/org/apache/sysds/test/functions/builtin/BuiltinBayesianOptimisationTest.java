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

package org.apache.sysds.test.functions.builtin;

import org.junit.Ignore;
import org.junit.Test;
import org.junit.Assert;

import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.lops.LopProperties.ExecType;

public class BuiltinBayesianOptimisationTest extends AutomatedTestBase {

	private final static String TEST_NAME = "bayesianOptimization";
	private final static String TEST_DIR = "functions/builtin/";
	private final static int rows = 300;
	private final static int cols = 200;

	@Override
	public void setUp()
	{
		addTestConfiguration(TEST_DIR, TEST_NAME);
	}

	@Test
	public void bayesianOptimisationMLMinimisationTest() {
		testBayesianOptimization("TRUE", 10, ExecType.CP);
	}

	@Test
	public void bayesianOptimisationMLMaximizationTest() {
		testBayesianOptimization("FALSE", 20, ExecType.CP);
	}

	@Ignore
	public void bayesianOptimisationMLMaximizationTestSpark() {
		testBayesianOptimization("FALSE", 20, ExecType.SPARK);
	}


	public void testBayesianOptimization(String minimize, int iter, ExecType exec) {

		ExecMode modeOld = setExecMode(exec);

		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);
			fullDMLScriptName = "./scripts/staging/bayesian_optimization/test/bayesianOptimizationMLTest.dml";
			programArgs = new String[] {"-args", input("X"), input("y"),
				String.valueOf(minimize), String.valueOf(iter), output("R")};
			double[][] X = getRandomMatrix(rows, cols, 0, 1, 0.7, 1);
			double[][] y = getRandomMatrix(rows, 1, 0, 1, 1, 2);
			writeInputMatrixWithMTD("X", X, true);
			writeInputMatrixWithMTD("y", y, true);

			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
			Assert.assertTrue(TestUtils.readDMLBoolean(output("R")));
		}
		finally {
			resetExecMode(modeOld);
		}
	}
}

