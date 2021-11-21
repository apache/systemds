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

import org.apache.sysds.common.Types;
import org.apache.sysds.runtime.matrix.data.MatrixValue;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class BuiltinHospitalResidencyMatchTest extends AutomatedTestBase {


	private final static String TEST_NAME = "residencymatch";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinHospitalResidencyMatchTest.class.getSimpleName() + "/";

	private final static double eps = 0.0001;

	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"RM"}));
	}

	@Test
	public void testResidencyMatch1() {
		double[][] R = {
				{2,3,1},{1,3,2},{3,1,3}};
		double[][] H = {
				{1,2,0},{3,1,2},{0,1,2}};
		double[][] C = {
				{2},{3},{2}};
		double[][]EM = { // this is an expected matrix
				{0,1,0},{1,0,0},{0,0,1}};
		runtestResidencyMatchTest(R, H, C, EM, Types.ExecType.CP);
	}

	@Test
	public void testResidencyMatch2() {
		double[][] R = {
				{2,1,3},{1,2,3},{1,3,2}};
		double[][] H = {
				{3,1,2},{2,1,3},{3,2,1}};
		double[][] C = {
				{1},{1},{1}};
		double[][]EM = { // this is an expected matrix
				{0,0,3},{0,2,0},{1,0,0}};
		runtestResidencyMatchTest(R, H, C, EM, Types.ExecType.CP);
	}

	@Test
	public void testResidencyMatch3() {
		double[][] R = {
				{1,2},{2,1},{1,2},{1,2}};
		double[][] H = {
				{3,2,1,4},{2,1,3,0}};
		double[][] C = {
				{4},{3}};
		double[][]EM = { // this is an expected matrix
				{1,0},{0,1},{1,0},{1,0}};
		runtestResidencyMatchTest(R, H, C, EM, Types.ExecType.CP);
	}
	@Test
	public void testResidencyMatch4() {
		double[][] R = {
				{1,2},{2,1},{1,2},{1,2}};
		double[][] H = {
				{3,2,1,4},{2,1,3,0}};
		double[][] C = {
				{4},{3}};
		double[][]EM = { // this is an expected matrix
				{1,0},{0,1},{1,0},{1,0}};
		runtestResidencyMatchTest(R, H, C, EM, Types.ExecType.SPARK);
	}

	private void runtestResidencyMatchTest(double[][] R, double[][] H, double[][] C, double[][] EM,
		Types.ExecType instType) {

		Types.ExecMode platformOld = setExecMode(instType);
		try {
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			List<String> proArgs = new ArrayList<>();
			proArgs.add("-args");
			proArgs.add(input("R"));
			proArgs.add(input("H"));
			proArgs.add(input("C"));
			proArgs.add(output("RM"));

			programArgs = proArgs.toArray(new String[proArgs.size()]);
			// defining Residents Matrix

			writeInputMatrixWithMTD("R", R, true);
			writeInputMatrixWithMTD("H", H, true);
			writeInputMatrixWithMTD("C", C, true);


			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);

			//compare expected results
			HashMap<MatrixValue.CellIndex, Double> matrixU = readDMLMatrixFromOutputDir("RM");
			double[][] OUT = TestUtils.convertHashMapToDoubleArray(matrixU);
			TestUtils.compareMatrices(EM, OUT, eps);
		}
		finally {
			rtplatform = platformOld;
		}
	}
}

