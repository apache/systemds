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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameters;

import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.TestUtils;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class BuiltinCsplineTest extends AutomatedTestBase {
	protected final static String TEST_NAME = "cspline";
	protected final static String TEST_DIR = "functions/builtin/";
	protected String TEST_CLASS_DIR = TEST_DIR + BuiltinCsplineTest.class.getSimpleName() + "/";
	
	protected int numRecords;
	private final static int numDim = 1;

	public BuiltinCsplineTest(int rows, int cols) {
		numRecords = rows;
	}

	@Parameters
	public static Collection<Object[]> data() {
		Object[][] data = new Object[][] {{10, 1}, {100, 1}, {1000, 1}};
		return Arrays.asList(data);
	}

	@Override
	public void setUp() {
		addTestConfiguration(TEST_CLASS_DIR, TEST_NAME);
	}

	@Test
	public void testCsplineDS() {	
		int rows = numRecords;
		int cols = numDim;

		getAndLoadTestConfiguration(TEST_NAME);

		List<String> proArgs = new ArrayList<>();
		proArgs.add("-args");
		proArgs.add(input("X"));
		proArgs.add(input("Y"));
		proArgs.add(Double.toString(4.5));
		proArgs.add(output("pred_y"));
		proArgs.add("DS");
		programArgs = proArgs.toArray(new String[proArgs.size()]);
		
		fullDMLScriptName = getScript();
		
		rCmd = getRCmd(input("X.mtx"), input("Y.mtx"), Double.toString(4.5), expected("pred_y"));

		double[][] X = new double[rows][cols];

		// X axis is given in the increasing order
		for (int rid = 0; rid < rows; rid++) {
			for (int cid = 0; cid < cols; cid++) {
				X[rid][cid] = rid+1;
			}
		}

		double[][] Y = getRandomMatrix(rows, cols, 0, 5, 1.0, -1);

		writeInputMatrixWithMTD("X", X, true);
		writeInputMatrixWithMTD("Y", Y, true);

		runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
		runRScript(true);

		HashMap<MatrixValue.CellIndex, Double> priorR = readRMatrixFromExpectedDir("pred_y");
		HashMap<MatrixValue.CellIndex, Double> priorSYSTEMDS= readDMLMatrixFromOutputDir("pred_y");

		double[][] from_R = TestUtils.convertHashMapToDoubleArray(priorR);
		double[][] from_DML = TestUtils.convertHashMapToDoubleArray(priorSYSTEMDS);

		TestUtils.compareMatrices(from_R, from_DML, Math.pow(10, -12));
	}

	@Test
	public void testCsplineCG() {
		int rows = numRecords;
		int cols = numDim;
		int numIter = rows;

		getAndLoadTestConfiguration(TEST_NAME);

		List<String> proArgs = new ArrayList<>();
		proArgs.add("-args");
		proArgs.add(input("X"));
		proArgs.add(input("Y"));
		proArgs.add(Double.toString(4.5));
		proArgs.add(output("pred_y"));
		proArgs.add("CG");
		proArgs.add(Double.toString(0.000001));
		proArgs.add(Double.toString(numIter));
		programArgs = proArgs.toArray(new String[proArgs.size()]);
		
		fullDMLScriptName = getScript();
		
		rCmd = getRCmd(input("X.mtx"), input("Y.mtx"), Double.toString(4.5), expected("pred_y"));

		double[][] X = new double[rows][cols];

		// X axis is given in the increasing order
		for (int rid = 0; rid < rows; rid++) {
			for (int cid = 0; cid < cols; cid++) {
				X[rid][cid] = rid+1;
			}
		}

		double[][] Y = getRandomMatrix(rows, cols, 0, 5, 1.0, -1);

		writeInputMatrixWithMTD("X", X, true);
		writeInputMatrixWithMTD("Y", Y, true);

		runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
		runRScript(true);

		HashMap<MatrixValue.CellIndex, Double> priorR = readRMatrixFromExpectedDir("pred_y");
		HashMap<MatrixValue.CellIndex, Double> priorSYSTEMDS= readDMLMatrixFromOutputDir("pred_y");
		
		double[][] from_R = TestUtils.convertHashMapToDoubleArray(priorR);
		double[][] from_DML = TestUtils.convertHashMapToDoubleArray(priorSYSTEMDS);

		TestUtils.compareMatrices(from_R, from_DML, Math.pow(10, -5));
	}
}
