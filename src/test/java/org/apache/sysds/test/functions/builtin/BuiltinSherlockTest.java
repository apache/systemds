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

import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class BuiltinSherlockTest extends AutomatedTestBase {
	private final static String TEST_NAME = "sherlock";
	private final static String TEST_DIR = "functions/builtin/";
	private final static String TEST_CLASS_DIR = TEST_DIR + BuiltinSherlockTest.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"B"}));
	}

	@Test
	public void testSherlock() {
		runtestSherlock();
	}

	@SuppressWarnings("unused")
	private void runtestSherlock() {
		loadTestConfiguration(getTestConfiguration(TEST_NAME));
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";

		List<String> proArgs = new ArrayList<>();
		proArgs.add("-exec");
		proArgs.add(" singlenode");
		proArgs.add("-nvargs");
		proArgs.add("X=" + input("X"));
		proArgs.add("Y=" + input("Y"));
		
		proArgs.add("cW1=" + output("cW1"));
		proArgs.add("cb1=" + output("cb1"));
		proArgs.add("cW2=" + output("cW2"));
		proArgs.add("cb2=" + output("cb2"));
		proArgs.add("cW3=" + output("cW3"));
		proArgs.add("cb3=" + output("cb3"));
		proArgs.add("wW1=" + output("wW1"));
		proArgs.add("wb1=" + output("wb1"));
		proArgs.add("wW2=" + output("wW2"));
		proArgs.add("wb2=" + output("wb2"));
		proArgs.add("wW3=" + output("wW3"));
		proArgs.add("wb3=" + output("wb3"));
		proArgs.add("pW1=" + output("pW1"));
		proArgs.add("pb1=" + output("pb1"));
		proArgs.add("pW2=" + output("pW2"));
		proArgs.add("pb2=" + output("pb2"));
		proArgs.add("pW3=" + output("pW3"));
		proArgs.add("pb3=" + output("pb3"));
		proArgs.add("sW1=" + output("sW1"));
		proArgs.add("sb1=" + output("sb1"));
		proArgs.add("sW2=" + output("sW2"));
		proArgs.add("sb2=" + output("sb2"));
		proArgs.add("sW3=" + output("sW3"));
		proArgs.add("sb3=" + output("sb3"));
		proArgs.add("fW1=" + output("fW1"));
		proArgs.add("fb1=" + output("fb1"));
		proArgs.add("fW2=" + output("fW2"));
		proArgs.add("fb2=" + output("fb2"));
		proArgs.add("fW3=" + output("fW3"));
		proArgs.add("fb3=" + output("fb3"));
		programArgs = proArgs.toArray(new String[proArgs.size()]);
		
		double[][] X = getRandomMatrix(256, 1588, 0, 3, 0.9, 7);
		double[][] Y = getRandomMatrix(256, 78, 0, 1, 0.9, 7);
		
		writeInputMatrixWithMTD("X", X, true);
		writeInputMatrixWithMTD("Y", Y, true);
		runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
		
		//compare expected results
		HashMap<MatrixValue.CellIndex, Double> cW1 = readDMLMatrixFromOutputDir("cW1");
		HashMap<MatrixValue.CellIndex, Double> cb1 = readDMLMatrixFromOutputDir("cb1");
		HashMap<MatrixValue.CellIndex, Double> cW2 = readDMLMatrixFromOutputDir("cW2");
		HashMap<MatrixValue.CellIndex, Double> cb2 = readDMLMatrixFromOutputDir("cb2");
		HashMap<MatrixValue.CellIndex, Double> cW3 = readDMLMatrixFromOutputDir("cW3");
		HashMap<MatrixValue.CellIndex, Double> cb3 = readDMLMatrixFromOutputDir("cb3");
		HashMap<MatrixValue.CellIndex, Double> wW1 = readDMLMatrixFromOutputDir("wW1");
		HashMap<MatrixValue.CellIndex, Double> wb1 = readDMLMatrixFromOutputDir("wb1");
		HashMap<MatrixValue.CellIndex, Double> wW2 = readDMLMatrixFromOutputDir("wW2");
		HashMap<MatrixValue.CellIndex, Double> wb2 = readDMLMatrixFromOutputDir("wb2");
		HashMap<MatrixValue.CellIndex, Double> wW3 = readDMLMatrixFromOutputDir("wW3");
		HashMap<MatrixValue.CellIndex, Double> wb3 = readDMLMatrixFromOutputDir("wb3");
		HashMap<MatrixValue.CellIndex, Double> pW1 = readDMLMatrixFromOutputDir("pW1");
		HashMap<MatrixValue.CellIndex, Double> pb1 = readDMLMatrixFromOutputDir("pb1");
		HashMap<MatrixValue.CellIndex, Double> pW2 = readDMLMatrixFromOutputDir("pW2");
		HashMap<MatrixValue.CellIndex, Double> pb2 = readDMLMatrixFromOutputDir("pb2");
		HashMap<MatrixValue.CellIndex, Double> pW3 = readDMLMatrixFromOutputDir("pW3");
		HashMap<MatrixValue.CellIndex, Double> pb3 = readDMLMatrixFromOutputDir("pb3");
		HashMap<MatrixValue.CellIndex, Double> sW1 = readDMLMatrixFromOutputDir("sW1");
		HashMap<MatrixValue.CellIndex, Double> sb1 = readDMLMatrixFromOutputDir("sb1");
		HashMap<MatrixValue.CellIndex, Double> sW2 = readDMLMatrixFromOutputDir("sW2");
		HashMap<MatrixValue.CellIndex, Double> sb2 = readDMLMatrixFromOutputDir("sb2");
		HashMap<MatrixValue.CellIndex, Double> sW3 = readDMLMatrixFromOutputDir("sW3");
		HashMap<MatrixValue.CellIndex, Double> sb3 = readDMLMatrixFromOutputDir("sb3");
		HashMap<MatrixValue.CellIndex, Double> fW1 = readDMLMatrixFromOutputDir("fW1");
		HashMap<MatrixValue.CellIndex, Double> fb1 = readDMLMatrixFromOutputDir("fb1");
		HashMap<MatrixValue.CellIndex, Double> fW2 = readDMLMatrixFromOutputDir("fW2");
		HashMap<MatrixValue.CellIndex, Double> fb2 = readDMLMatrixFromOutputDir("fb2");
		HashMap<MatrixValue.CellIndex, Double> fW3 = readDMLMatrixFromOutputDir("fW3");
		HashMap<MatrixValue.CellIndex, Double> fb3 = readDMLMatrixFromOutputDir("fb3");
	}
}
