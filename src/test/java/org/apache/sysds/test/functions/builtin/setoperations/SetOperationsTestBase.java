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

package org.apache.sysds.test.functions.builtin.setoperations;

import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.*;

@RunWith(Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public abstract class SetOperationsTestBase extends AutomatedTestBase {
	private final String TEST_NAME;
	private final String TEST_DIR ;
	private final String TEST_CLASS_DIR;

	private final ExecType _execType;

	public SetOperationsTestBase(String test_name, String test_dir, String test_class_dir, Types.ExecType execType){
		TEST_NAME = test_name;
		TEST_DIR = test_dir;
		TEST_CLASS_DIR = test_class_dir;
		_execType = execType;
	}

	@Parameterized.Parameters
	public static Collection<Object[]> types(){
		return Arrays.asList(new Object[][]{
				{Types.ExecType.CP},
				{Types.ExecType.SPARK}
		});
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[]{"R"}));
	}

	@Test
	public void testPosNumbersAscending() {
		double[][] X = {{1}, {2}, {3}};
		double[][] Y = {{2}, {3}, {4}};

		runUnitTest(X, Y, _execType);
	}

	@Test
	public void testPosNumbersRandomOrder() {
		double[][] X = {{9}, {2}, {3}};
		double[][] Y = {{2}, {3}, {4}};

		runUnitTest(X, Y, _execType);
	}

	@Test
	public void testComplexPosNumbers() {
		double[][] X =  {{12},{22},{13},{4},{6},{7},{8},{9},{12},{12}};
		double[][] Y = {{1},{2},{11},{12},{13},{18},{20},{21},{12}};
		runUnitTest(X, Y, _execType);
	}

	@Test
	public void testNegNumbers() {
		double[][] X = {{-10},{-5},{2}};
		double[][] Y = {{2},{-3}};
		runUnitTest(X, Y, _execType);
	}

	@Test
	public void testFloatingPNumbers() {
		double[][] X = {{2},{2.5},{4}};
		double[][] Y = {{2.4},{2}};
		runUnitTest(X, Y, _execType);
	}

	@Test
	public void testNegAndFloating() {
		double[][] X =  {{1.4}, {-1.3}, {10}, {4}};
		double[][] Y = {{1.3},{-1.4},{10},{9}};
		runUnitTest(X, Y, _execType);
	}

	@Test
	public void testMinValue() {
		double[][] X =  {{Double.MIN_VALUE}, {2},{4}};
		double[][] Y = {{2},{15}};
		runUnitTest(X, Y, _execType);
	}

	@Test
	public void testCombined() {
		double[][] X =  {{Double.MIN_VALUE}, {4}, {-1.3}, {10}, {4}};
		double[][] Y = {{Double.MIN_VALUE},{15},{-1.2},{-25.3}};
		runUnitTest(X, Y, _execType);
	}

	@Test
	public void testYSuperSetOfX() {
		double[][] X = TestUtils.seq(2, 200, 4);
		double[][] Y = TestUtils.seq(2, 200, 2);
		runUnitTest(X, Y, _execType);
	}

	@Test
	public void testXSuperSetOfY() {
		double[][] X = TestUtils.seq(2, 200, 2);
		double[][] Y = TestUtils.seq(2, 200, 4);
		runUnitTest(X, Y, _execType);
	}

	private void runUnitTest(double[][] X, double[][]Y, Types.ExecType instType) {
		Types.ExecMode platformOld = setExecMode(instType);
		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{ "-args", input("X"),input("Y"), output("R")};
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();

			writeInputMatrixWithMTD("X", X, true);
			writeInputMatrixWithMTD("Y", Y, true);

			runTest(true, false, null, -1);
			runRScript(true);

			HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");
			HashMap<MatrixValue.CellIndex, Double> rfile  = readRMatrixFromExpectedDir("R");

			ArrayList<Double> dml_values = new ArrayList<>(dmlfile.values());
			ArrayList<Double> r_values = new ArrayList<>(rfile.values());
			Collections.sort(dml_values);
			Collections.sort(r_values);

			Assert.assertEquals(dml_values.size(), r_values.size());
			Assert.assertEquals(dml_values, r_values);

			//Junit way collection equal ignore order.
			//Assert.assertTrue(dml_values.size() == r_values.size() && dml_values.containsAll(r_values) && r_values.containsAll(dml_values));
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
