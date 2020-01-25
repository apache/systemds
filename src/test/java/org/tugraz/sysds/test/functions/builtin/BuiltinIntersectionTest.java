/*
 * Copyright 2020 Graz University of Technology
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tugraz.sysds.test.functions.builtin;

import org.junit.Assert;
import org.junit.Test;
import org.tugraz.sysds.common.Types;
import org.tugraz.sysds.lops.LopProperties;
import org.tugraz.sysds.runtime.matrix.data.MatrixValue;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestConfiguration;
import org.tugraz.sysds.test.TestUtils;

import java.util.Arrays;
import java.util.HashMap;

public class BuiltinIntersectionTest extends AutomatedTestBase {
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
		double[][] X =  {{12},{22},{13},{4},{6},{7},{8},{9}, {12}, {12}};
		double[][] Y = {{1},{2},{11},{12},{13}, {18},{20},{21},{12}};
		X = TestUtils.round(X);
		Y = TestUtils.round(Y);
		runIntersectTest(X, Y, LopProperties.ExecType.CP);
	}

	@Test
	public void testIntersect1Spark() {
		double[][] X =  {{12},{22},{13},{4},{6},{7},{8},{9}, {12}, {12}};
		double[][] Y = {{1},{2},{11},{12},{13}, {18},{20},{21},{12}};
		X = TestUtils.round(X);
		Y = TestUtils.round(Y);
		runIntersectTest(X, Y, LopProperties.ExecType.SPARK);
	}

	@Test
	public void testIntersect2CP() {
		double[][] X =  new double[50][1];
		double[][] Y = new double[50][1];
		for(int i =0, j=2; i<50; i++, j+=4)
			X[i][0] = j;
		for(int i =0, j=2; i<50; i++, j+=2)
			Y[i][0] = j;
		runIntersectTest(X, Y, LopProperties.ExecType.CP);
		HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("C");
		Object[]  out = dmlfile.values().toArray();
		Arrays.sort(out);
		for(int i = 0; i< out.length; i++)
			Assert.assertEquals(out[i], Double.valueOf(X[i][0]));
	}

	@Test
	public void testIntersect2Spark() {
		double[][] X =  new double[50][1];
		double[][] Y = new double[50][1];
		for(int i =0, j=2; i<50; i++, j+=4)
			X[i][0] = j;
		for(int i =0, j=2; i<50; i++, j+=2)
			Y[i][0] = j;
		runIntersectTest(X, Y, LopProperties.ExecType.SPARK);
		HashMap<MatrixValue.CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("C");
		Object[]  out = dmlfile.values().toArray();
		Arrays.sort(out);
		for(int i = 0; i< out.length; i++)
			Assert.assertEquals(out[i], Double.valueOf(X[i][0]));
	}



	private void runIntersectTest(double X[][], double Y[][], LopProperties.ExecType instType)
	{
		Types.ExecMode platformOld = setExecMode(instType);
		//TODO compare with R instead of custom output comparison
		
		try
		{
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{ "-args", input("X"), input("Y"), output("C"), output("X")};

			//generate actual datasets
			writeInputMatrixWithMTD("X", X, true);
			writeInputMatrixWithMTD("Y", Y, true);

			runTest(true, false, null, -1);
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
