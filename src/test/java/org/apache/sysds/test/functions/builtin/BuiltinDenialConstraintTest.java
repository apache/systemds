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

import org.apache.sysds.common.Types;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.io.IOException;
import java.util.HashMap;

public class BuiltinDenialConstraintTest extends AutomatedTestBase {

	private final static String TEST_NAME = "denial_constraints";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinDenialConstraintTest.class.getSimpleName() + "/";
	private final static double eps = 1e-10;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"M"}));
	}

	@Test
	public void testSpark() throws IOException {
		runConstraints_Test(ExecType.SPARK);
	}
	@Test
	public void testCP() throws IOException {
		runConstraints_Test(ExecType.CP);
	}

	
	private void runConstraints_Test(ExecType instType)
	{
		Types.ExecMode platformOld = setExecMode(instType);
		
		try
		{
			loadTestConfiguration(getTestConfiguration(TEST_NAME));
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[] {"-nvargs", "M="+ output("Mx")}; 

			runTest(true, false, null, -1);
			double[][] X = {{1, 1}, {1, 2}, {1, 5}, {2, 2}, {2, 5}, {3, 6}, {4 , 2}, {4, 5}, {5, 1}, {5, 2}, {5, 5},
				{6, 4}, {7, 2}, {8 , 2}, {9, 2}, {10, 1}, {10, 2}, {10 , 5}};
			HashMap<CellIndex, Double> dmlfile_m = readDMLMatrixFromOutputDir("Mx");
			double[][] Y = TestUtils.convertHashMapToDoubleArray(dmlfile_m);
			TestUtils.compareMatrices(X, Y, eps);

		}
		finally {
			rtplatform = platformOld;
		}
	}

}
