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

import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;

public class BuiltinSoftmaxTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "softmax1";
	private final static String TEST_NAME2 = "softmax2";
	private final static String TEST_DIR = "functions/builtin/";
	private static final String TEST_CLASS_DIR = TEST_DIR + BuiltinSoftmaxTest.class.getSimpleName() + "/";
	
	private final static double eps = 1e-6;
	private final static int rows = 1765;
	private final static double spDense = 0.99;
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[]{"B"}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[]{"B"}));
	}

	@Test
	public void testSoftmaxCP() {
		runSoftmaxTest(TEST_NAME1, ExecType.CP);
	}
	
	@Test
	public void testSoftmaxSP() {
		runSoftmaxTest(TEST_NAME1, ExecType.SPARK);
	}

//TODO add support for eval lazy loading of builtin funcitons w/ imports
//	@Test
//	public void testSoftmaxEvalCP() {
//		runSoftmaxTest(TEST_NAME2, ExecType.CP);
//	}
//	
//	@Test
//	public void testSoftmaxEvalSP() {
//		runSoftmaxTest(TEST_NAME2, ExecType.SPARK);
//	}

	private void runSoftmaxTest(String testname, ExecType instType) {
		ExecMode platformOld = setExecMode(instType);
		
		try {
			loadTestConfiguration(getTestConfiguration(testname));
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{"-args",
				input("A"), String.valueOf(eps), output("B") };
			
			//generate actual dataset 
			double[][] A = getRandomMatrix(rows, 10, -1, 1, spDense, 7);
			writeInputMatrixWithMTD("A", A, true);
			
			runTest(true, false, null, -1);
			
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("B");
			Assert.assertEquals(rows*10, dmlfile.get(new CellIndex(1,1)).intValue());
		}
		finally {
			rtplatform = platformOld;
		}
	}
}
