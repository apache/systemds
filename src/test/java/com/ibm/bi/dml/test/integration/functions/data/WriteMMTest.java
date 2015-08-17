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

package com.ibm.bi.dml.test.integration.functions.data;


import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;



/**
 * <p><b>Positive tests:</b></p>
 * <ul>
 * 	<li>text</li>
 * 	<li>binary</li>
 * 	<li>write a matrix two times</li>
 * </ul>
 * <p><b>Negative tests:</b></p>
 * 
 * 
 */
public class WriteMMTest extends AutomatedTestBase 
{
	
	private final static String TEST_NAME1 = "WriteMMTest";
	private final static String TEST_NAME2 = "WriteMMComplexTest";
	private final static String TEST_DIR = "functions/data/";
	
	//for CP
	private final static int rows1 = 30;
	private final static int cols1 = 10;
	//for MR
	private final static int rows2 = 700;  
	private final static int cols2 = 100;
	
		
	@Override
	public void setUp() 
	{
		TestUtils.clearAssertionInformation();
		addTestConfiguration(
				TEST_NAME1, 
				new TestConfiguration(TEST_DIR, TEST_NAME1, 
				new String[] { "B" })   ); 
		addTestConfiguration(
				TEST_NAME2, 
				new TestConfiguration(TEST_DIR, TEST_NAME2, 
				new String[] { "B" })   ); 
	}

	@Test
	public void testWriteMMCP() 
	{
		runWriteMMTest(ExecType.CP, TEST_NAME1);
	}
	
	@Test
	public void testWriteMMSP() 
	{
		runWriteMMTest(ExecType.SPARK, TEST_NAME1);
	}
	
	@Test
	public void testWriteMMMR() 
	{
		runWriteMMTest(ExecType.MR, TEST_NAME1);
	}
	
	@Test
	public void testWriteMMMRMerge()
	{
		runWriteMMTest(ExecType.MR, TEST_NAME2);
	}
	
	private void runWriteMMTest( ExecType instType, String TEST_NAME )
	{
		//setup exec type, rows, cols
		int rows = -1, cols = -1;
		
		
		if( instType == ExecType.CP ) {
				rows = rows1;
				cols = cols1;
		}
		else { //if type MR
				rows = rows2;
				cols = cols2;
		}
			

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
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain", "-args", HOME + INPUT_DIR + "A" ,
					                        Integer.toString(rows),
					                        Integer.toString(cols),
					                        HOME + OUTPUT_DIR + "B"  };
			
			loadTestConfiguration(config);
	
			//generate actual dataset
			double[][] A = getRandomMatrix(rows, cols, -1, 1, 1, System.currentTimeMillis()); 
			writeInputMatrixWithMTD("A", A, false, new MatrixCharacteristics(rows,cols, 1000, 1000));
			writeExpectedMatrixMarket("B", A);
	
			runTest(true, false, null, -1);
			compareResultsWithMM();
		}
		finally
		{
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
	
}
