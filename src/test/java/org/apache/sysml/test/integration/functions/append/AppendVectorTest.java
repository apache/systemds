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

package org.apache.sysml.test.integration.functions.append;

import java.util.HashMap;
import java.util.Random;

import org.junit.Assert;
import org.junit.Test;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.parser.DMLTranslator;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;
import org.apache.sysml.utils.Statistics;

public class AppendVectorTest extends AutomatedTestBase
{

	
	private final static String TEST_NAME = "AppendVectorTest";
	private final static String TEST_DIR = "functions/append/";
	private final static String TEST_CLASS_DIR = TEST_DIR + AppendVectorTest.class.getSimpleName() + "/";

	private final static double epsilon=0.0000000001;
	private final static int rows1 = 1279;
	private final static int cols1 = 1059;
	private final static int rows2 = 2021;
	private final static int cols2 = DMLTranslator.DMLBlockSize;
	private final static int min=0;
	private final static int max=100;
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, 
				new String[] {"C"}));
	}
	
	@Test
	public void testAppendInBlockSP() {
		commonAppendTest(RUNTIME_PLATFORM.SPARK, rows1, cols1);
	}
	@Test
	public void testAppendOutBlockSP() {
		commonAppendTest(RUNTIME_PLATFORM.SPARK, rows2, cols2);
	}
	

	@Test
	public void testAppendInBlockCP() {
		commonAppendTest(RUNTIME_PLATFORM.SINGLE_NODE, rows1, cols1);
	}
	
	@Test
	public void testAppendOutBlockCP() {
		commonAppendTest(RUNTIME_PLATFORM.SINGLE_NODE, rows2, cols2);
	}	

	@Test
	public void testAppendInBlockMR() {
		commonAppendTest(RUNTIME_PLATFORM.HADOOP, rows1, cols1);
	}   
	
	@Test
	public void testAppendOutBlockMR() {
		commonAppendTest(RUNTIME_PLATFORM.HADOOP, rows2, cols2);
	}   

	
	public void commonAppendTest(RUNTIME_PLATFORM platform, int rows, int cols)
	{
		TestConfiguration config = getAndLoadTestConfiguration(TEST_NAME);
	    
		RUNTIME_PLATFORM prevPlfm=rtplatform;
		
	    rtplatform = platform;
	    boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
	    if( rtplatform == RUNTIME_PLATFORM.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

	    try {
	        config.addVariable("rows", rows);
	        config.addVariable("cols", cols);
	          
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String RI_HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = RI_HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain", "-args",  input("A"), 
					Long.toString(rows), Long.toString(cols),
								input("B"),
		                        output("C") };
			fullRScriptName = RI_HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       inputDir() + " "+ expectedDir();
	
			Random rand=new Random(System.currentTimeMillis());
			double sparsity=rand.nextDouble();
			double[][] A = getRandomMatrix(rows, cols, min, max, sparsity, System.currentTimeMillis());
	        writeInputMatrix("A", A, true);
	        sparsity=rand.nextDouble();
	        double[][] B= getRandomMatrix(rows, 1, min, max, sparsity, System.currentTimeMillis());
	        writeInputMatrix("B", B, true);
	        
	        boolean exceptionExpected = false;
	        int expectedCompiledMRJobs = (rtplatform==RUNTIME_PLATFORM.HADOOP)? 2 : 1;
			int expectedExecutedMRJobs = (rtplatform==RUNTIME_PLATFORM.HADOOP)? 2 : 0;
			runTest(true, exceptionExpected, null, expectedCompiledMRJobs);
			Assert.assertEquals("Wrong number of executed MR jobs.",
					             expectedExecutedMRJobs, Statistics.getNoOfExecutedMRJobs());
		
			runRScript(true);
			//disableOutAndExpectedDeletion();
		
			for(String file: config.getOutputFiles())
			{
				HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS(file);
				HashMap<CellIndex, Double> rfile = readRMatrixFromFS(file);
			//	System.out.println(file+"-DML: "+dmlfile);
			//	System.out.println(file+"-R: "+rfile);
				TestUtils.compareMatrices(dmlfile, rfile, epsilon, file+"-DML", file+"-R");
			}
	    }
	    finally {
			rtplatform = prevPlfm;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
	    }
	}
   
}
