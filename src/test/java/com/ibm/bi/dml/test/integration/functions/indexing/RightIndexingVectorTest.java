/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.test.integration.functions.indexing;

import java.util.HashMap;
import java.util.Random;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;



public class RightIndexingVectorTest extends AutomatedTestBase
{

	
	private final static String TEST_NAME = "RightIndexingVectorTest";
	private final static String TEST_DIR = "functions/indexing/";

	private final static double epsilon=0.0000000001;
	private final static int rows = 2279;
	private final static int cols = 1050;
	private final static int min=0;
	private final static int max=100;
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_DIR, TEST_NAME, 
				new String[] {"B", "C", "D"}));
	}
	
	@Test
	public void testRightIndexingCP() 
	{
		runRightIndexingTest(ExecType.CP);
	}
	
	@Test
	public void testRightIndexingSP() 
	{
		runRightIndexingTest(ExecType.SPARK);
	}
	
	@Test
	public void testRightIndexingMR() 
	{
		runRightIndexingTest(ExecType.MR);
	}
	
	public void runRightIndexingTest( ExecType et ) 
	{
		RUNTIME_PLATFORM oldRTP = rtplatform;
				
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		
		try
		{
		    TestConfiguration config = getTestConfiguration(TEST_NAME);
		    if(et == ExecType.SPARK) {
		    	rtplatform = RUNTIME_PLATFORM.SPARK;
		    }
		    else {
		    	rtplatform = (et==ExecType.MR)? RUNTIME_PLATFORM.HADOOP : RUNTIME_PLATFORM.SINGLE_NODE;
		    }
		    if( rtplatform == RUNTIME_PLATFORM.SPARK )
				DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		    
		    config.addVariable("rows", rows);
	        config.addVariable("cols", cols);
	        
	        String RI_HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = RI_HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args",  RI_HOME + INPUT_DIR + "A" , 
		               							 Long.toString(rows), 
		               							 Long.toString(cols),
		                                         RI_HOME + OUTPUT_DIR + "B" , 
		                                         RI_HOME + OUTPUT_DIR + "C" , 
		                                         RI_HOME + OUTPUT_DIR + "D" };
			fullRScriptName = RI_HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       RI_HOME + INPUT_DIR + " " + RI_HOME + EXPECTED_DIR;
	
			loadTestConfiguration(config);
			
			Random rand=new Random(System.currentTimeMillis());
	        double sparsity=rand.nextDouble();
	        double[][] A = getRandomMatrix(rows, cols, min, max, sparsity, System.currentTimeMillis());
	        writeInputMatrix("A", A, true);
	        
	        boolean exceptionExpected = false;
			int expectedNumberOfJobs = -1;
			runTest(true, exceptionExpected, null, expectedNumberOfJobs);
			
			runRScript(true);
			//disableOutAndExpectedDeletion();
		
			for(String file: config.getOutputFiles())
			{
				HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS(file);
				HashMap<CellIndex, Double> rfile = readRMatrixFromFS(file);
				TestUtils.compareMatrices(dmlfile, rfile, epsilon, file+"-DML", file+"-R");
			}
		}
		finally
		{
			rtplatform = oldRTP;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
}
