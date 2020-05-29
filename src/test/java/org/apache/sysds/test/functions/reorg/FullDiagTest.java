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

package org.apache.sysds.test.functions.reorg;

import java.util.HashMap;
import java.util.Random;

import org.junit.Test;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class FullDiagTest extends AutomatedTestBase
{
	private final static String TEST_DIR = "functions/reorg/";
	private final static String TEST_NAME1 = "DiagV2MTest";
	private final static String TEST_NAME2 = "DiagM2VTest";
	private static final String TEST_CLASS_DIR = TEST_DIR + FullDiagTest.class.getSimpleName() + "/";

	private final static double epsilon=0.0000000001;
	private final static int rows = 1059;
	private final static int min=0;
	private final static int max=100;
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"C"}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] {"C"}));
	}
	
	@Test
	public void testDiagV2MCP() {
		commonReorgTest(TEST_NAME1, ExecMode.SINGLE_NODE);
	}
	
	@Test
	public void testDiagV2MSP() {
		commonReorgTest(TEST_NAME1, ExecMode.SPARK);
	}
	
	@Test
	public void testDiagM2VCP() {
		commonReorgTest(TEST_NAME2, ExecMode.SINGLE_NODE);
	}
	
	@Test
	public void testDiagM2VSP() {
		commonReorgTest(TEST_NAME2, ExecMode.SPARK);
	}
	
	public void commonReorgTest(String testname, ExecMode platform)
	{
		TestConfiguration config = getTestConfiguration(testname);
		ExecMode prevPlfm = setExecMode(platform);
		
		try {
			config.addVariable("rows", rows);
			loadTestConfiguration(config);
	
			String RI_HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = RI_HOME + testname + ".dml";
			programArgs = new String[]{"-explain", "-args",  input("A"), output("C") };
			fullRScriptName = RI_HOME + testname + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();
	
			Random rand=new Random(System.currentTimeMillis());
			int cols = testname.equals(TEST_NAME1) ? 1 : rows;
			double sparsity=0.599200924665577;
			double[][] A = getRandomMatrix(rows, cols, min, max, sparsity, 1397289950533L);
			writeInputMatrixWithMTD("A", A, true);
			sparsity=rand.nextDouble();
			
			runTest(true, EXCEPTION_NOT_EXPECTED, null, -1);
			runRScript(true);
		
			for(String file: config.getOutputFiles()) {
				HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS(file);
				HashMap<CellIndex, Double> rfile = readRMatrixFromFS(file);
				TestUtils.compareMatrices(dmlfile, rfile, epsilon, file+"-DML", file+"-R");
			}
		}
		finally {
			resetExecMode(prevPlfm);
		}
	}
}
