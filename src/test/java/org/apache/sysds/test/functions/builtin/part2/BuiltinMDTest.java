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


package org.apache.sysds.test.functions.builtin.part2;

import java.util.Arrays;
import java.util.Collection;

import org.apache.sysds.common.Types;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

@RunWith(value = Parameterized.class)
@net.jcip.annotations.NotThreadSafe
public class BuiltinMDTest extends AutomatedTestBase {
	private final static String TEST_NAME = "matching_dependency";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinMDTest.class.getSimpleName() + "/";

	@Parameterized.Parameter()
	public double[][] LHSf;

	@Parameterized.Parameter(1)
	public double[][] LHSt;

	@Parameterized.Parameter(2)
	public double[][] RHSf;

	@Parameterized.Parameter(3)
	public double[][] RHSt;

	@Parameterized.Parameters
	public static Collection<Object[]> data() {
		return Arrays.asList(new Object[][] {
			{new double[][] {{1}}, new double[][] {{0.95}},
				new double[][] {{5}}, new double[][] {{0.65}}},

			{new double[][] {{1,3}}, new double[][] {{0.7,0.8}},
				new double[][] {{5}}, new double[][] {{0.8}}},

			{new double[][] {{1,4,5}}, new double[][] {{0.9,0.9,0.9}},
				new double[][] {{6}}, new double[][] {{0.9}}},

			{new double[][] {{1,4,5}}, new double[][] {{0.75,0.6,0.9}},
				new double[][] {{3}}, new double[][] {{0.8}}},
		});
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"D"}));
		if (TEST_CACHE_ENABLED) {
			setOutAndExpectedDeletionDisabled(true);
		}
	}

	@Test
	public void testMDCP() {
		double[][] D =  {
			{7567, 231, 1231, 1232, 122, 321},
			{5321, 23123, 122, 123, 1232, 11},
			{7267, 3, 223, 432, 1132, 0},
			{7267, 3, 223, 432, 1132, 500},
			{7254, 3, 223, 432, 1132, 0},
		};
		runMDTests(D, LHSf, LHSt, RHSf, RHSt, ExecType.CP);
	}

	@Test
	//@Ignore
	// https://issues.apache.org/jira/browse/SYSTEMDS-3716
	public void testMDSP() {
		double[][] D =  {
			{7567, 231, 1231, 1232, 122, 321},
			{5321, 23123, 122, 123, 1232, 11},
			{7267, 3, 223, 432, 1132, 0},
			{7267, 3, 223, 432, 1132, 500},
			{7254, 3, 223, 432, 1132, 0},
		};
		runMDTests(D, LHSf, LHSt, RHSf, RHSt, ExecType.SPARK);
	}
	
	private void runMDTests(double [][] X , double[][] LHSf, double[][] LHSt, double[][] RHSf, double[][] RHSt, ExecType instType) {
		Types.ExecMode platformOld = setExecMode(instType);
		try
		{
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-stats","-args", input("X"),
				input("LHSf"), input("LHSt"), input("RHSf"), input("RHSt"), output("B")};

			double[][] A = getRandomMatrix(20, 6, 50, 500, 1, 2);
			System.arraycopy(X, 0, A, 0, X.length);

			writeInputMatrixWithMTD("X", A, false);
			writeInputMatrixWithMTD("LHSf", LHSf, true);
			writeInputMatrixWithMTD("LHSt", LHSt, true);
			writeInputMatrixWithMTD("RHSf", RHSf, true);
			writeInputMatrixWithMTD("RHSt", RHSt, true);

			runTest(true, false, null, -1);
		}
		catch(Exception e) {
			e.printStackTrace();
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
