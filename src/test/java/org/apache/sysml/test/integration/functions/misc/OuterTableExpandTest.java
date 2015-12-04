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

package org.apache.sysml.test.integration.functions.misc;

import java.util.HashMap;

import org.junit.Test;

import org.apache.sysml.api.DMLException;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

/**
 * 
 */
public class OuterTableExpandTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "OuterExpandTest";
	private final static String TEST_NAME2 = "TableExpandTest";
	
	private final static String TEST_DIR = "functions/misc/";
	private final static double eps = 1e-8;
	
	private final static int rows = 5191;
	private final static int cols2 = 1212;
	
	private final static double sparsity1 = 1.0; //dense 
	private final static double sparsity2 = 0.1; //sparse
	
	@Override
	public void setUp() 
	{
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_DIR, TEST_NAME1, new String[] { "C" })); 
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_DIR, TEST_NAME2, new String[] { "C" })); 
	}
	
	// outer tests ----------------
	
	@Test
	public void testOuterDenseLeftCP() {
		runOuterTest(false, true, ExecType.CP);
	}
	
	@Test
	public void testOuterSparseLeftCP() {
		runOuterTest(true, true, ExecType.CP);
	}
	
	@Test
	public void testOuterDenseRightCP() {
		runOuterTest(false, false, ExecType.CP);
	}
	
	@Test
	public void testOuterSparseRightCP() {
		runOuterTest(true, false, ExecType.CP);
	}
	
	@Test
	public void testOuterDenseLeftMR() {
		runOuterTest(false, true, ExecType.MR);
	}
	
	@Test
	public void testOuterSparseLeftMR() {
		runOuterTest(true, true, ExecType.MR);
	}
	
	@Test
	public void testOuterDenseRightMR() {
		runOuterTest(false, false, ExecType.MR);
	}
	
	@Test
	public void testOuterSparseRightMR() {
		runOuterTest(true, false, ExecType.MR);
	}
	
	@Test
	public void testOuterDenseLeftSP() {
		runOuterTest(false, true, ExecType.SPARK);
	}
	
	@Test
	public void testOuterSparseLeftSP() {
		runOuterTest(true, true, ExecType.SPARK);
	}
	
	@Test
	public void testOuterDenseRightSP() {
		runOuterTest(false, false, ExecType.SPARK);
	}
	
	@Test
	public void testOuterSparseRightSP() {
		runOuterTest(true, false, ExecType.SPARK);
	}
	
	// table tests ----------------
	
	@Test
	public void testTableDenseLeftCP() {
		runTableTest(false, true, ExecType.CP);
	}
	
	@Test
	public void testTableSparseLeftCP() {
		runTableTest(true, true, ExecType.CP);
	}
	
	@Test
	public void testTableDenseRightCP() {
		runTableTest(false, false, ExecType.CP);
	}
	
	@Test
	public void testTableSparseRightCP() {
		runTableTest(true, false, ExecType.CP);
	}
	
	@Test
	public void testTableDenseLeftMR() {
		runTableTest(false, true, ExecType.MR);
	}
	
	@Test
	public void testTableSparseLeftMR() {
		runTableTest(true, true, ExecType.MR);
	}
	
	@Test
	public void testTableDenseRightMR() {
		runTableTest(false, false, ExecType.MR);
	}
	
	@Test
	public void testTableSparseRightMR() {
		runTableTest(true, false, ExecType.MR);
	}
	
	@Test
	public void testTableDenseLeftSP() {
		runTableTest(false, true, ExecType.SPARK);
	}
	
	@Test
	public void testTableSparseLeftSP() {
		runTableTest(true, true, ExecType.SPARK);
	}
	
	@Test
	public void testTableDenseRightSP() {
		runTableTest(false, false, ExecType.SPARK);
	}
	
	@Test
	public void testTableSparseRightSP() {
		runTableTest(true, false, ExecType.SPARK);
	}
	
	
	/**
	 * 
	 * @param sparse
	 * @param left
	 * @param instType
	 */
	private void runOuterTest( boolean sparse, boolean left, ExecType instType)
	{
		runOuterTableTest(TEST_NAME1, sparse, left, instType);
	}
	
	private void runTableTest( boolean sparse, boolean left, ExecType instType)
	{
		runOuterTableTest(TEST_NAME2, sparse, left, instType);
	}
	
	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
	private void runOuterTableTest( String testname, boolean sparse, boolean left, ExecType instType)
	{
		//rtplatform for MR
		RUNTIME_PLATFORM platformOld = rtplatform;
		switch( instType ){
			case MR: rtplatform = RUNTIME_PLATFORM.HADOOP; break;
			case SPARK: rtplatform = RUNTIME_PLATFORM.SPARK; break;
			default: rtplatform = RUNTIME_PLATFORM.HYBRID; break;
		}
	
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == RUNTIME_PLATFORM.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		try
		{
			String TEST_NAME = testname;
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			String HOME = SCRIPT_DIR + TEST_DIR;			
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain","-args", 
					                            HOME + INPUT_DIR + "A",
					                            String.valueOf(cols2),
					                            String.valueOf(left).toUpperCase(),
					                            HOME + OUTPUT_DIR + "C"};
			fullRScriptName = HOME + TEST_NAME +".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       HOME + INPUT_DIR + " " + cols2 + " " + String.valueOf(left).toUpperCase() + " " + HOME + EXPECTED_DIR;
			
			loadTestConfiguration(config);
	
			//generate actual datasets
			double sparsity = sparse?sparsity2:sparsity1;
			double[][] A = TestUtils.round(getRandomMatrix(rows, 1, 1, cols2, sparsity, 235));
			writeInputMatrixWithMTD("A", A, true);
			
			//run the testcase (expect exceptions for table w/ 0s)
			boolean exceptionExpected = testname.equals(TEST_NAME2) && sparsity < 1.0;
			runTest(true, exceptionExpected, DMLException.class, -1); 
			runRScript(true); 
			
			if( !exceptionExpected ) {
				//compare matrices 
				HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("C");
				HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("C");
				TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
				
				//check meta data
				checkDMLMetaDataFile("C", new MatrixCharacteristics(left?rows:cols2,left?cols2:rows,1,1));
			
				//check compiled/executed jobs
				if( rtplatform == RUNTIME_PLATFORM.HADOOP ) {
					int expectedNumCompiled = 3; //reblock+gmr+gmr if rexpand; otherwise 3/5 
					int expectedNumExecuted = expectedNumCompiled; 
					checkNumCompiledMRJobs(expectedNumCompiled); 
					checkNumExecutedMRJobs(expectedNumExecuted); 	
				}
			}
		}
		finally
		{
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}

}