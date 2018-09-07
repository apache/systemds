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

package org.apache.sysml.test.integration.functions.unary.matrix;

import org.junit.Test;

import java.util.HashMap;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

public class ExtractTriangularTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "ExtractLowerTri";
	private final static String TEST_NAME2 = "ExtractUpperTri";
	private final static String TEST_DIR = "functions/unary/matrix/";
	private static final String TEST_CLASS_DIR = TEST_DIR + ExtractTriangularTest.class.getSimpleName() + "/";

	private final static int _rows = 1321;
	private final static int _cols = 1123;
	private final static double _sparsityDense = 0.5;
	private final static double _sparsitySparse = 0.05;
	private final static double eps = 1e-8;
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "R" }) );
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "R" }) );
	}
	
	@Test
	public void testExtractLowerTriDenseBoolCP() {
		runExtractTriangular(TEST_NAME1, false, false, false, ExecType.CP);
	}
	
	@Test
	public void testExtractLowerTriDenseValuesCP() {
		runExtractTriangular(TEST_NAME1, false, false, true, ExecType.CP);
	}
	
	@Test
	public void testExtractLowerTriDenseDiagBoolCP() {
		runExtractTriangular(TEST_NAME1, false, true, false, ExecType.CP);
	}
	
	@Test
	public void testExtractLowerTriDenseDiagValuesCP() {
		runExtractTriangular(TEST_NAME1, false, true, true, ExecType.CP);
	}
	
	@Test
	public void testExtractLowerTriSparseBoolCP() {
		runExtractTriangular(TEST_NAME1, true, false, false, ExecType.CP);
	}
	
	@Test
	public void testExtractLowerTriSparseValuesCP() {
		runExtractTriangular(TEST_NAME1, true, false, true, ExecType.CP);
	}
	
	@Test
	public void testExtractLowerTriSparseDiagBoolCP() {
		runExtractTriangular(TEST_NAME1, true, true, false, ExecType.CP);
	}
	
	@Test
	public void testExtractLowerTriSparseDiagValuesCP() {
		runExtractTriangular(TEST_NAME1, true, true, true, ExecType.CP);
	}
	
	@Test
	public void testExtractUpperTriDenseBoolCP() {
		runExtractTriangular(TEST_NAME2, false, false, false, ExecType.CP);
	}
	
	@Test
	public void testExtractUpperTriDenseValuesCP() {
		runExtractTriangular(TEST_NAME2, false, false, true, ExecType.CP);
	}
	
	@Test
	public void testExtractUpperTriDenseDiagBoolCP() {
		runExtractTriangular(TEST_NAME2, false, true, false, ExecType.CP);
	}
	
	@Test
	public void testExtractUpperTriDenseDiagValuesCP() {
		runExtractTriangular(TEST_NAME2, false, true, true, ExecType.CP);
	}
	
	@Test
	public void testExtractUpperTriSparseBoolCP() {
		runExtractTriangular(TEST_NAME2, true, false, false, ExecType.CP);
	}
	
	@Test
	public void testExtractUpperTriSparseValuesCP() {
		runExtractTriangular(TEST_NAME2, true, false, true, ExecType.CP);
	}
	
	@Test
	public void testExtractUpperTriSparseDiagBoolCP() {
		runExtractTriangular(TEST_NAME2, true, true, false, ExecType.CP);
	}
	
	@Test
	public void testExtractUpperTriSparseDiagValuesCP() {
		runExtractTriangular(TEST_NAME2, true, true, true, ExecType.CP);
	}
	
	@Test
	public void testExtractLowerTriDenseBoolSP() {
		runExtractTriangular(TEST_NAME1, false, false, false, ExecType.SPARK);
	}
	
	@Test
	public void testExtractLowerTriDenseValuesSP() {
		runExtractTriangular(TEST_NAME1, false, false, true, ExecType.SPARK);
	}
	
	@Test
	public void testExtractLowerTriDenseDiagBoolSP() {
		runExtractTriangular(TEST_NAME1, false, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testExtractLowerTriDenseDiagValuesSP() {
		runExtractTriangular(TEST_NAME1, false, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testExtractLowerTriSparseBoolSP() {
		runExtractTriangular(TEST_NAME1, true, false, false, ExecType.SPARK);
	}
	
	@Test
	public void testExtractLowerTriSparseValuesSP() {
		runExtractTriangular(TEST_NAME1, true, false, true, ExecType.SPARK);
	}
	
	@Test
	public void testExtractLowerTriSparseDiagBoolSP() {
		runExtractTriangular(TEST_NAME1, true, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testExtractLowerTriSparseDiagValuesSP() {
		runExtractTriangular(TEST_NAME1, true, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testExtractUpperTriDenseBoolSP() {
		runExtractTriangular(TEST_NAME2, false, false, false, ExecType.SPARK);
	}
	
	@Test
	public void testExtractUpperTriDenseValuesSP() {
		runExtractTriangular(TEST_NAME2, false, false, true, ExecType.SPARK);
	}
	
	@Test
	public void testExtractUpperTriDenseDiagBoolSP() {
		runExtractTriangular(TEST_NAME2, false, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testExtractUpperTriDenseDiagValuesSP() {
		runExtractTriangular(TEST_NAME2, false, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testExtractUpperTriSparseBoolSP() {
		runExtractTriangular(TEST_NAME2, true, false, false, ExecType.SPARK);
	}
	
	@Test
	public void testExtractUpperTriSparseValuesSP() {
		runExtractTriangular(TEST_NAME2, true, false, true, ExecType.SPARK);
	}
	
	@Test
	public void testExtractUpperTriSparseDiagBoolSP() {
		runExtractTriangular(TEST_NAME2, true, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testExtractUpperTriSparseDiagValuesSP() {
		runExtractTriangular(TEST_NAME2, true, true, true, ExecType.SPARK);
	}
	
	private void runExtractTriangular( String testname, boolean sparse, boolean diag, boolean values, ExecType et)
	{
		//rtplatform for MR
		RUNTIME_PLATFORM platformOld = rtplatform;
		switch( et ){
			case MR: rtplatform = RUNTIME_PLATFORM.HADOOP; break;
			case SPARK: rtplatform = RUNTIME_PLATFORM.SPARK; break;
			default: rtplatform = RUNTIME_PLATFORM.HYBRID; break;
		}
	
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == RUNTIME_PLATFORM.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		
		try
		{
			//setup dims and sparsity
			double sparsity = sparse ? _sparsitySparse : _sparsityDense;
			
			//register test configuration
			TestConfiguration config = getTestConfiguration(testname);
			config.addVariable("rows", _rows);
			config.addVariable("cols", _cols);
			loadTestConfiguration(config);
			
			String sdiag = String.valueOf(diag).toUpperCase();
			String svalues = String.valueOf(values).toUpperCase();
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{"-explain", "-args",
				input("X"), sdiag, svalues, output("R") };
			fullRScriptName = HOME + testname + ".R";
			rCmd = "Rscript "+fullRScriptName+" "
				+inputDir()+" "+sdiag+" "+svalues+" "+expectedDir();
	
			//generate actual dataset 
			double[][] X = getRandomMatrix(_rows, _cols, -0.05, 1, sparsity, 7);
			writeInputMatrixWithMTD("X", X, true);
			
			runTest(true, false, null, -1);
			runRScript(true);
			
			//compare matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			HashMap<CellIndex, Double> rfile = readRMatrixFromFS("R");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
		}
		finally {
			//reset platform for additional tests
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}
