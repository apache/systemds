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

package org.apache.sysds.test.functions.builtin.part1;

import java.util.HashMap;
import java.util.Random;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

@net.jcip.annotations.NotThreadSafe
public class BuiltinHMMPredictTest extends AutomatedTestBase
{
	private final static String TEST_NAME = "hmmPredict";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinHMMPredictTest.class.getSimpleName() + "/";
	private final static double eps = 1e-3;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"C"}));
	}
    	
    @Test    
	public void testHMMPredictCP() {
		runHMMPredictTest(ExecType.CP);
	}

	@Test
	public void testHMMPredictSpark() {
		runHMMPredictTest(ExecType.SPARK);
	}
	
	private void runHMMPredictTest(ExecType instType)
	{
		Types.ExecMode platformOld = setExecMode(instType);
		try
		{
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;

			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{
					"-nvargs", "X=" + input("X"), "k=" + input("k"), "outputs=" + output("outputs")};

			//Output generated from https://en.wikipedia.org/wiki/Hidden_Markov_model#Weather_guessing_game
			double[][] X = {{2, 1, 2, 1, 2, 2, 3, 3, 1, 1, 1, 1, 2, 2, 3, 2, 1, 2, 3, 2, 
						    1, 1, 1, 3, 2, 2, 2, 2, 3, 3, 1, 2, 3, 1, 3, 3, 2, 3, 1, 2, 1, 
							1, 2, 3, 1, 1, 1, 3, 2, 3, 2, 1, 1, 3, 3, 3, 1, 2, 3, 3, 1, 3, 
							3, 3, 2, 2, 2, 3, 3, 3, 3, 3, 1, 1, 3, 2, 2, 3, 2, 2, 1, 3, 2, 
							1, 3, 2, 2, 2, 1, 2, 1, 2, 2, 1, 3, 2, 2, 2, 3, 2}};
			int k = 10;
			
			writeInputMatrixWithMTD("X", X, true);
			runTest(true, false, null, -1);

			HashMap<MatrixValue.CellIndex, Double> test_output = readDMLMatrixFromOutputDir("outputs");
			HashMap<MatrixValue.CellIndex, Double> expected_output = new HashMap<>();

			expected_output.put(new MatrixValue.CellIndex(1, 1), 1.0);
			expected_output.put(new MatrixValue.CellIndex(2, 1), 3.0);
			expected_output.put(new MatrixValue.CellIndex(3, 1), 3.0);
			expected_output.put(new MatrixValue.CellIndex(4, 1), 2.0);
			expected_output.put(new MatrixValue.CellIndex(5, 1), 1.0);
			expected_output.put(new MatrixValue.CellIndex(6, 1), 2.0);
			expected_output.put(new MatrixValue.CellIndex(7, 1), 3.0);
			expected_output.put(new MatrixValue.CellIndex(8, 1), 1.0);
			expected_output.put(new MatrixValue.CellIndex(9, 1), 3.0);
			expected_output.put(new MatrixValue.CellIndex(10, 1), 1.0);

			TestUtils.compareMatrices(expected_output, test_output, eps, "Expected-DML", "Actual-DML");
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
