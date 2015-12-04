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

package org.apache.sysml.test.integration.functions.append;

import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;
import org.apache.sysml.utils.Statistics;

public class AppendChainTest extends AutomatedTestBase
{
	
	private final static String TEST_NAME = "AppendChainTest";
	private final static String TEST_DIR = "functions/append/";
	private final static String TEST_CLASS_DIR = TEST_DIR + AppendChainTest.class.getSimpleName() + "/";

	private final static double epsilon=0.0000000001;
	private final static int min=1;
	private final static int max=100;
	
	private final static int rows = 1692;
	private final static int cols1 = 1059;
	private final static int cols2a = 1;
	private final static int cols3a = 1;
	private final static int cols2b = 1030;
	private final static int cols3b = 1770;
	
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.07;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, 
				new String[] {"C"}));
	}

	@Test
	public void testAppendChainVectorDenseCP() {
		commonAppendTest(RUNTIME_PLATFORM.HYBRID, rows, cols1, cols2a, cols3a, false);
	}
	
	@Test
	public void testAppendChainMatrixDenseCP() {
		commonAppendTest(RUNTIME_PLATFORM.HYBRID, rows, cols1, cols2b, cols3b, false);
	}
	
	// ------------------------------------------------------
	@Test
	public void testAppendChainVectorDenseSP() {
		commonAppendTest(RUNTIME_PLATFORM.SPARK, rows, cols1, cols2a, cols3a, false);
	}
	
	@Test
	public void testAppendChainMatrixDenseSP() {
		commonAppendTest(RUNTIME_PLATFORM.SPARK, rows, cols1, cols2b, cols3b, false);
	}
	
	@Test
	public void testAppendChainVectorSparseSP() {
		commonAppendTest(RUNTIME_PLATFORM.SPARK, rows, cols1, cols2a, cols3a, true);
	}
	
	@Test
	public void testAppendChainMatrixSparseSP() {
		commonAppendTest(RUNTIME_PLATFORM.SPARK, rows, cols1, cols2b, cols3b, true);
	}
	
	// ------------------------------------------------------
	
	@Test
	public void testAppendChainVectorDenseMR() {
		commonAppendTest(RUNTIME_PLATFORM.HADOOP, rows, cols1, cols2a, cols3a, false);
	}
	
	@Test
	public void testAppendChainMatrixDenseMR() {
		commonAppendTest(RUNTIME_PLATFORM.HADOOP, rows, cols1, cols2b, cols3b, false);
	}
	
	@Test
	public void testAppendChainVectorSparseCP() {
		commonAppendTest(RUNTIME_PLATFORM.HYBRID, rows, cols1, cols2a, cols3a, true);
	}
	
	@Test
	public void testAppendChainMatrixSparseCP() {
		commonAppendTest(RUNTIME_PLATFORM.HYBRID, rows, cols1, cols2b, cols3b, true);
	}
	
	@Test
	public void testAppendChainVectorSparseMR() {
		commonAppendTest(RUNTIME_PLATFORM.HADOOP, rows, cols1, cols2a, cols3a, true);
	}
	
	@Test
	public void testAppendChainMatrixSparseMR() {
		commonAppendTest(RUNTIME_PLATFORM.HADOOP, rows, cols1, cols2b, cols3b, true);
	}
			
	/**
	 * 
	 * @param platform
	 * @param rows
	 * @param cols1
	 * @param cols2
	 * @param cols3
	 */
	public void commonAppendTest(RUNTIME_PLATFORM platform, int rows, int cols1, int cols2, int cols3, boolean sparse)
	{
		TestConfiguration config = getAndLoadTestConfiguration(TEST_NAME);
	    
		RUNTIME_PLATFORM prevPlfm=rtplatform;
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		
		try
		{
		    rtplatform = platform;
		    if( rtplatform == RUNTIME_PLATFORM.SPARK )
				DMLScript.USE_LOCAL_SPARK_CONFIG = true;
			
	        config.addVariable("rows", rows);
	        config.addVariable("cols", cols1);
	          
			//This is for running the junit test the new way, i.e., construct the arguments directly
			String RI_HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = RI_HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args",  input("A"), 
					                             Long.toString(rows), 
					                             Long.toString(cols1),
								                 input("B1"),
								                 Long.toString(cols2),
								                 input("B2"),
								                 Long.toString(cols3),
		                                         output("C") };
			fullRScriptName = RI_HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       inputDir() + " "+ expectedDir();
	
			double sparsity = sparse ? sparsity2 : sparsity1; 
			double sparsity2 = 1-sparsity;
	        
			double[][] A = getRandomMatrix(rows, cols1, min, max, sparsity, 11);
	        writeInputMatrix("A", A, true);
	        double[][] B1= getRandomMatrix(rows, cols2, min, max, sparsity2, 21);
	        writeInputMatrix("B1", B1, true);
	        double[][] B2= getRandomMatrix(rows, cols2, min, max, sparsity, 31);
	        writeInputMatrix("B2", B2, true);
	        
	        boolean exceptionExpected = false;
			int expectedCompiledMRJobs = (rtplatform==RUNTIME_PLATFORM.HADOOP)? 2+((cols3>1)?1:0) : 1;
			int expectedExecutedMRJobs = (rtplatform==RUNTIME_PLATFORM.HADOOP)? 2+((cols3>1)?1:0) : 0; 
			runTest(true, exceptionExpected, null, expectedCompiledMRJobs);
			runRScript(true);
			Assert.assertEquals("Wrong number of executed MR jobs.",
					             expectedExecutedMRJobs, Statistics.getNoOfExecutedMRJobs());
			
			//compare result data
			for(String file: config.getOutputFiles())
			{
				HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS(file);
				HashMap<CellIndex, Double> rfile = readRMatrixFromFS(file);
				TestUtils.compareMatrices(dmlfile, rfile, epsilon, file+"-DML", file+"-R");
			}
		}
		finally
		{
			rtplatform = prevPlfm;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}
