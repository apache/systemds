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

import org.junit.Test;
import org.apache.sysds.common.Types;
import org.apache.sysds.lops.LopProperties;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

import java.util.HashMap;

public class BuiltinIntersectionTest extends AutomatedTestBase
{
	private final static String TEST_NAME = "intersection";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinIntersectionTest.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"C"}));
	}

	@Test
	public void testIntersect1CP() {
		double[][] X =  {{12},{22},{13},{4},{6},{7},{8},{9},{12},{12}};
		double[][] Y = {{1},{2},{11},{12},{13},{18},{20},{21},{12}};
		double[][] expected = {{12},{13}};
		runIntersectTest(X, Y, expected, LopProperties.ExecType.CP);
	}

	@Test
	public void testIntersect1Spark() {
		double[][] X = {{12},{22},{13},{4},{6},{7},{8},{9},{12},{12}};
		double[][] Y = {{1},{2},{11},{12},{13},{18},{20},{21},{12}};
		double[][] expected = {{12},{13}};
		runIntersectTest(X, Y, expected, LopProperties.ExecType.SPARK);
	}

	@Test
	public void testIntersect2CP() {
		double[][] X = TestUtils.seq(2, 200, 4);
		double[][] Y = TestUtils.seq(2, 100, 2);
		double[][] expected = TestUtils.seq(2, 100, 4);
		runIntersectTest(X, Y, expected, LopProperties.ExecType.CP);
	}

	@Test
	public void testIntersect2Spark() {
		double[][] X = TestUtils.seq(2, 200, 4);
		double[][] Y = TestUtils.seq(2, 100, 2);
		double[][] expected = TestUtils.seq(2, 100, 4);
		runIntersectTest(X, Y, expected, LopProperties.ExecType.SPARK);
	}

	private void runIntersectTest(double X[][], double Y[][], double[][] expected, LopProperties.ExecType instType)
	{
		Types.ExecMode platformOld = setExecMode(instType);
		
		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{ "-args", input("X"), input("Y"), output("C"), output("X")};

			//generate actual datasets
			writeInputMatrixWithMTD("X", X, true);
			writeInputMatrixWithMTD("Y", Y, true);
			
			//run test
			runTest(true, false, null, -1);
		
			//compare expected results
			HashMap<CellIndex, Double> R = new HashMap<>();
			for(int i=0; i<expected.length; i++)
				R.put(new CellIndex(i+1,1), expected[i][0]);
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("C");
			TestUtils.compareMatrices(dmlfile, R, 1e-10, "dml", "expected");
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
