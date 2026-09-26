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
public class BuiltinHMMTest extends AutomatedTestBase
{
	private final static String TEST_NAME = "hmm";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinHMMPredictTest.class.getSimpleName() + "/";
	private final static double eps = 1e-3;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"C"}));
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
		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{
					"-nvargs", "X=" + input("X"), "ip=" + output("ip"), "A=" + output("A"), "B=" + output("B")};

			//Output generated from https://en.wikipedia.org/wiki/Hidden_Markov_model#Weather_guessing_game
			double[][] X = {{2, 1, 2, 1, 2, 2, 3, 3, 1, 1, 1, 1, 2, 2, 3, 2, 1, 2, 3, 2, 
						    1, 1, 1, 3, 2, 2, 2, 2, 3, 3, 1, 2, 3, 1, 3, 3, 2, 3, 1, 2, 1, 
							1, 2, 3, 1, 1, 1, 3, 2, 3, 2, 1, 1, 3, 3, 3, 1, 2, 3, 3, 1, 3, 
							3, 3, 2, 2, 2, 3, 3, 3, 3, 3, 1, 1, 3, 2, 2, 3, 2, 2, 1, 3, 2, 
							1, 3, 2, 2, 2, 1, 2, 1, 2, 2, 1, 3, 2, 2, 2, 3, 2}};
			
			writeInputMatrixWithMTD("X", X, true);
			runTest(true, false, null, -1);

			HashMap<MatrixValue.CellIndex, Double> out_ip = readDMLMatrixFromOutputDir("ip");
			HashMap<MatrixValue.CellIndex, Double> out_A = readDMLMatrixFromOutputDir("A");
			HashMap<MatrixValue.CellIndex, Double> out_B = readDMLMatrixFromOutputDir("B");
			HashMap<MatrixValue.CellIndex, Double> expected_ip = new HashMap<>();
			HashMap<MatrixValue.CellIndex, Double> expected_A = new HashMap<>();
			HashMap<MatrixValue.CellIndex, Double> expected_B = new HashMap<>();

			expected_ip.put(new MatrixValue.CellIndex(1, 1), 0.731);
			expected_ip.put(new MatrixValue.CellIndex(2, 1), 0.000);
			expected_ip.put(new MatrixValue.CellIndex(3, 1), 0.269);

			expected_A.put(new MatrixValue.CellIndex(1, 1), 0.194);
			expected_A.put(new MatrixValue.CellIndex(1, 2), 0.421);
			expected_A.put(new MatrixValue.CellIndex(1, 3), 0.424);
			expected_A.put(new MatrixValue.CellIndex(2, 1), 0.189);
			expected_A.put(new MatrixValue.CellIndex(2, 2), 0.432);
			expected_A.put(new MatrixValue.CellIndex(2, 3), 0.383);
			expected_A.put(new MatrixValue.CellIndex(3, 1), 0.374);
			expected_A.put(new MatrixValue.CellIndex(3, 2), 0.334);
			expected_A.put(new MatrixValue.CellIndex(3, 3), 0.257);

			expected_B.put(new MatrixValue.CellIndex(1, 1), 0.383);
			expected_B.put(new MatrixValue.CellIndex(1, 2), 0.609);
			expected_B.put(new MatrixValue.CellIndex(1, 3), 0.007);
			expected_B.put(new MatrixValue.CellIndex(2, 1), 0.152);
			expected_B.put(new MatrixValue.CellIndex(2, 2), 0.012);
			expected_B.put(new MatrixValue.CellIndex(2, 3), 0.837);
			expected_B.put(new MatrixValue.CellIndex(3, 1), 0.373);
			expected_B.put(new MatrixValue.CellIndex(3, 2), 0.615);
			expected_B.put(new MatrixValue.CellIndex(3, 3), 0.012);

			TestUtils.compareMatrices(expected_ip, out_ip, eps, "Expected-DML", "Actual-DML");
			TestUtils.compareMatrices(expected_A, out_A, eps, "Expected-DML", "Actual-DML");
			TestUtils.compareMatrices(expected_B, out_B, eps, "Expected-DML", "Actual-DML");
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
