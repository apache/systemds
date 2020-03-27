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
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;


public class BuiltinFDTest extends AutomatedTestBase {
	private final static String TEST_NAME = "functional_dependency";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinFDTest.class.getSimpleName() + "/";

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME,new TestConfiguration(TEST_CLASS_DIR, TEST_NAME,new String[]{"C"}));
	}

	@Test
	public void testFD1() {
		double[][] X =  {{7,1,1,2,2,1},{7,2,2,3,2,1},{7,3,1,4,1,1},{7,4,2,5,3,1},{7,5,3,6,5,1}, {7,6,5,1,4,1}};
		double[][] M = {{0, 1, 1, 0, 1, 1}};
		runFDTests(X, M, 1, LopProperties.ExecType.CP );
	}

	@Test
	public void testFD2() {
		double[][] X =  {{1,1,1,1,1},{2,1,2,2,1},{3,2,1,1,1},{4,2,2,2,1},{5,3,3,1,1}};
		double[][] M = {{1, 1, 1, 1, 1}};
		runFDTests(X, M, 0.8, LopProperties.ExecType.CP );
	}

	@Test
	public void testFD3() {
		double[][] X =  {{1,1},{2,1},{3,2},{2,2},{4,2}, {5,3}};
		double[][] M = {{0, 1}};
		runFDTests(X, M, 0.8, LopProperties.ExecType.CP );
	}
	
	@Test
	public void testFD4() {
		double[][] X =  {{1,1,1,1,1,1,2},{2,1,2,2,1,2,2},{3,2,1,1,1,3,2},{4,2,2,2,1,4,2},{5,3,3,1,1,5,1}};
		double[][] M = {{0, 1, 1, 1, 0, 1, 1}};
		runFDTests(X, M, 0.8, LopProperties.ExecType.CP );
	}
	
	@Test
	public void testFD5() {
		double[][] X =  {{7,1,1,2,2,1},{7,2,2,3,2,1},{7,3,1,4,1,1},{7,4,2,5,3,1},{7,5,3,6,5,1}, {7,6,5,1,4,1}};
		double[][] M = {{0, 1, 1, 0, 1, 1}};
		runFDTests(X, M, 0.8, LopProperties.ExecType.CP );
	}
	
	private void runFDTests(double [][] X , double[][] M, double threshold, LopProperties.ExecType instType) {
		Types.ExecMode platformOld = setExecMode(instType);
		try
		{
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-stats","-args", input("X"), input("M"), String.valueOf(threshold), output("B")};
			writeInputMatrixWithMTD("X", X, true);
			writeInputMatrixWithMTD("M", M, true);
			runTest(true, false, null, -1);
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
