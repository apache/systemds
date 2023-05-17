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

package org.apache.sysds.test.functions.async;

import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class CostBasedOrderTest extends AutomatedTestBase {

	protected static final String TEST_DIR = "functions/async/";
	protected static final String TEST_NAME = "CostBasedOrder";
	protected static final int TEST_VARIANTS = 1;
	protected static String TEST_CLASS_DIR = TEST_DIR + CostBasedOrderTest.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		for(int i=1; i<=TEST_VARIANTS; i++)
			addTestConfiguration(TEST_NAME+i, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME+i));
	}

	@Test
	public void testlmds() {
		runTest(TEST_NAME+"1");
	}

	public void runTest(String testname) {
		getAndLoadTestConfiguration(testname);
		fullDMLScriptName = getScript();

		List<String> proArgs = new ArrayList<>();

		proArgs.add("-explain");
		proArgs.add("-stats");
		proArgs.add("-args");
		proArgs.add(output("R"));
		programArgs = proArgs.toArray(new String[proArgs.size()]);

		runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
		HashMap<MatrixValue.CellIndex, Double> R = readDMLScalarFromOutputDir("R");

		OptimizerUtils.COST_BASED_ORDERING = true;
		runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
		HashMap<MatrixValue.CellIndex, Double> R_mp = readDMLScalarFromOutputDir("R");
		OptimizerUtils.COST_BASED_ORDERING = false;

		//compare matrices
		boolean matchVal = TestUtils.compareMatrices(R, R_mp, 1e-6, "Origin", "withMaxParallelize");
		if (!matchVal)
			System.out.println("Value w/ depth first"+R+" w/ cost-based"+R_mp);
	}

}
