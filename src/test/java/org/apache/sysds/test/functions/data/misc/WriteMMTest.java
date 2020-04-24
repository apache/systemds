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

package org.apache.sysds.test.functions.data.misc;


import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;



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
	private final static String TEST_CLASS_DIR = TEST_DIR + WriteMMTest.class.getSimpleName() + "/";
	
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
		addTestConfiguration(TEST_NAME1,
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "B" }) ); 
		addTestConfiguration(TEST_NAME2,
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "B" }) ); 
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
		ExecMode platformOld = rtplatform;
		switch( instType ){
			case SPARK: rtplatform = ExecMode.SPARK; break;
			default: rtplatform = ExecMode.HYBRID; break;
		}
		
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == ExecMode.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		
	
		try
		{
			getAndLoadTestConfiguration(TEST_NAME);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain", "-args",
				input("A"), Integer.toString(rows), Integer.toString(cols), output("B") };
	
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
