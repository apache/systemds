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

package org.apache.sysds.test.functions.nary;

import java.util.HashMap;

import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class NaryListCBindTest extends AutomatedTestBase
{	
	private final static String TEST_NAME = "NaryListCbind";
	private final static String TEST_DIR = "functions/nary/";
	private final static String TEST_CLASS_DIR = TEST_DIR + NaryListCBindTest.class.getSimpleName() + "/";

	private final static double epsilon=0.0000000001;
	
	private final static int rows = 1101;
	private final static int cols1 = 101;
	private final static int cols2 = 79;
	private final static int cols3 = 123;
	
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.07;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] {"R"}));
	}

	@Test
	public void testNaryCbindDenseDenseDenseCP() {
		runCbindTest(false, false, false, ExecType.CP);
	}
	
	@Test
	public void testNaryCbindDenseDenseSparseCP() {
		runCbindTest(false, false, true, ExecType.CP);
	}
	
	@Test
	public void testNaryCbindDenseSparseDenseCP() {
		runCbindTest(false, true, false, ExecType.CP);
	}
	
	@Test
	public void testNaryCbindDenseSparseSparseCP() {
		runCbindTest(false, true, true, ExecType.CP);
	}
	
	@Test
	public void testNaryCbindSparseDenseDenseCP() {
		runCbindTest(true, false, false, ExecType.CP);
	}
	
	@Test
	public void testNaryCbindSparseDenseSparseCP() {
		runCbindTest(true, false, true, ExecType.CP);
	}
	
	@Test
	public void testNaryCbindSparseSparseDenseCP() {
		runCbindTest(true, true, false, ExecType.CP);
	}
	
	@Test
	public void testNaryCbindSparseSparseSparseCP() {
		runCbindTest(true, true, true, ExecType.CP);
	}
	
	public void runCbindTest(boolean sparse1, boolean sparse2, boolean sparse3, ExecType et)
	{
		ExecMode platformOld = rtplatform;
		switch( et ) {
			case SPARK: rtplatform = ExecMode.SPARK; break;
			default: rtplatform = ExecMode.HYBRID; break;
		}
		
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == ExecMode.SPARK || rtplatform == ExecMode.HYBRID )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		
		try
		{
			TestConfiguration config = getAndLoadTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);
			
			String RI_HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = RI_HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain", "-stats", "-args", input("A"), 
				input("B"), input("C"), output("R") };
			fullRScriptName = RI_HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
				inputDir() + " "+ expectedDir();
			
			//generate input data
			double sp1 = sparse1 ? sparsity2 : sparsity1; 
			double sp2 = sparse2 ? sparsity2 : sparsity1; 
			double sp3 = sparse3 ? sparsity2 : sparsity1; 
			double[][] A = getRandomMatrix(rows, cols1, -1, 1, sp1, 711);
			double[][] B = getRandomMatrix(rows, cols2, -1, 1, sp2, 722);
			double[][] C = getRandomMatrix(rows, cols3, -1, 1, sp3, 733);
			writeInputMatrixWithMTD("A", A, true);
			writeInputMatrixWithMTD("B", B, true);
			writeInputMatrixWithMTD("C", C, true);
			
			//run tests
			runTest(true, false, null, -1);
			runRScript(true);
			
			//compare result data
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			HashMap<CellIndex, Double> rfile = readRMatrixFromFS("R");
			TestUtils.compareMatrices(dmlfile, rfile, epsilon, "DML", "R");
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}
