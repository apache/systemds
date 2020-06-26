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

package org.apache.sysds.test.functions.recompile;

import java.util.HashMap;

import org.junit.Test;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class IPAConstantPropagationFunTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "IPAFunctionArgsFor";
	private final static String TEST_NAME2 = "IPAFunctionArgsParfor";
	
	private final static String TEST_DIR = "functions/recompile/";
	private final static String TEST_CLASS_DIR = TEST_DIR + IPAConstantPropagationFunTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[]{"R"}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[]{"R"}));
	}

	@Test
	public void runIPAConstantPropagationForTest() {
		runIPAConstantPropagationTest(TEST_NAME1);
	}
	
	@Test
	public void runIPAConstantPropagationParForTest() {
		runIPAConstantPropagationTest(TEST_NAME2);
	}
	
	private void runIPAConstantPropagationTest(String testname) {
		boolean oldFlagIPA = OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS;
		
		try
		{
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{"-args", output("R") };

			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = true;
			runTest(true, false, null, -1);
			HashMap<CellIndex, Double> dmlfile1 = readDMLMatrixFromHDFS("R");
			
			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = false;
			runTest(true, false, null, -1);
			HashMap<CellIndex, Double> dmlfile2 = readDMLMatrixFromHDFS("R");
			
			//compare results with and without IPA
			TestUtils.compareMatrices(dmlfile1, dmlfile2, 1e-14, "IPA", "No IPA");
		}
		finally {
			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = oldFlagIPA;
		}
	}
}
