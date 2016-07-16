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

package org.apache.sysml.test.integration.functions.compress;

import java.util.HashMap;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;
import org.junit.Test;

/**
 * 
 */
public class CompressedLinregCG extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "LinregCG";
	private final static String TEST_DIR = "functions/compress/";
	private final static String TEST_CONF = "SystemML-config-compress.xml";
	
	private final static double eps = 1e-4;
	
	private final static int rows = 1468;
	private final static int cols = 980;
		
	private final static double sparsity1 = 0.7; //dense
	private final static double sparsity2 = 0.1; //sparse
	
	private final static int intercept = 0;
	private final static double epsilon = 0.000000001;
	private final static double maxiter = 10;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_DIR, TEST_NAME1, new String[] { "w" })); 
	}

	@Test
	public void testGDFOLinregCGDenseCP() {
		runGDFOTest(TEST_NAME1, false, ExecType.CP);
	}
	
	@Test
	public void testGDFOLinregCGSparseCP() {
		runGDFOTest(TEST_NAME1, true, ExecType.CP);
	}
	
	@Test
	public void testGDFOLinregCGDenseSP() {
		runGDFOTest(TEST_NAME1, false, ExecType.SPARK);
	}
	
	@Test
	public void testGDFOLinregCGSparseSP() {
		runGDFOTest(TEST_NAME1, true, ExecType.SPARK);
	}
	
	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
	private void runGDFOTest( String testname,boolean sparse, ExecType instType)
	{
		//rtplatform for MR
		RUNTIME_PLATFORM platformOld = rtplatform;
		switch( instType ){
			case MR: rtplatform = RUNTIME_PLATFORM.HADOOP; break;
			case SPARK: rtplatform = RUNTIME_PLATFORM.HYBRID_SPARK; break;
			default: rtplatform = RUNTIME_PLATFORM.HYBRID; break;
		}
	
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == RUNTIME_PLATFORM.HYBRID_SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		long memOld = InfrastructureAnalyzer.getLocalMaxMemory();
		
		try
		{
			String TEST_NAME = testname;
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{ "-explain","-stats",
					                    "-config="+HOME+TEST_CONF,
					                    "-args", HOME + INPUT_DIR + "X",
					                             HOME + INPUT_DIR + "y",
					                             String.valueOf(intercept),
					                             String.valueOf(epsilon),
					                             String.valueOf(maxiter),
					                            HOME + OUTPUT_DIR + "w"};
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       HOME + INPUT_DIR + " " + 
			       String.valueOf(intercept) + " " + String.valueOf(epsilon) + " " + 
			       String.valueOf(maxiter) + " " + HOME + EXPECTED_DIR;
			
			loadTestConfiguration(config);
	
			//generate actual datasets
			double[][] X = getRandomMatrix(rows, cols, 1, 1, sparse?sparsity2:sparsity1, 7);
			writeInputMatrixWithMTD("X", X, true);
			double[][] y = getRandomMatrix(rows, 1, 0, 10, 1.0, 3);
			writeInputMatrixWithMTD("y", y, true);
			
			if( rtplatform == RUNTIME_PLATFORM.HYBRID_SPARK  )
				InfrastructureAnalyzer.setLocalMaxMemory(8*1024*1024);
			
			runTest(true, false, null, -1); 
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("w");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("w");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
		}
		finally
		{
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			InfrastructureAnalyzer.setLocalMaxMemory(memOld);		
		}
	}

}