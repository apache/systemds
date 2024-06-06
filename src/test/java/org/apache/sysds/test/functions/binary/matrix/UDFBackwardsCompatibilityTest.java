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

package org.apache.sysds.test.functions.binary.matrix;

import org.junit.Test;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class UDFBackwardsCompatibilityTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "RowClassMeetTest";
	private final static String TEST_DIR = "functions/binary/matrix/";
	private final static String TEST_CLASS_DIR = TEST_DIR + 
		UDFBackwardsCompatibilityTest.class.getSimpleName() + "/";
	
	private final static int rows = 1267;
	private final static int cols = 56;
	
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.1;
	
	
	@Override
	public void setUp() {
		addTestConfiguration( TEST_NAME1,
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "C" }) );
	}

	@Test
	public void testRowClassMeetDenseDense() {
		runUDFTest(TEST_NAME1, false, false, ExecType.CP);
	}
	
	@Test
	public void testRowClassMeetDenseSparse() {
		runUDFTest(TEST_NAME1, false, true, ExecType.CP);
	}
	
	@Test
	public void testRowClassMeetSparseDense() {
		runUDFTest(TEST_NAME1, true, false, ExecType.CP);
	}
	
	@Test
	public void testRowClassMeetSparseSparse() {
		runUDFTest(TEST_NAME1, true, true, ExecType.CP);
	}
	
	private void runUDFTest(String testname, boolean sparseM1, boolean sparseM2, ExecType instType)
	{
		ExecMode platformOld = setExecMode(instType);
		String TEST_NAME = testname;
		
		try {
			getAndLoadTestConfiguration(TEST_NAME);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain","-args", input("A"), input("B"), output("C")};
			
			//generate actual dataset
			double[][] A = TestUtils.round(
				getRandomMatrix(rows, cols, 0, 10, sparseM1?sparsity2:sparsity1, 7));
			writeInputMatrixWithMTD("A", A, false);
			double[][] B = TestUtils.round(
				getRandomMatrix(rows, cols, 0, 10, sparseM2?sparsity2:sparsity1, 3));
			writeInputMatrixWithMTD("B", B, false);
	
			//run test case
			runTest(true, false, null, -1); 
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
