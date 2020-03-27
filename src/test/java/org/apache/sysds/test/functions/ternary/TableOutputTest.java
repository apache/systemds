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

package org.apache.sysds.test.functions.ternary;

import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

public class TableOutputTest extends AutomatedTestBase 
{
	
	private final static String TEST_NAME = "TableOutputTest";
	
	private final static String TEST_DIR = "functions/ternary/";
	private final static String TEST_CLASS_DIR = TEST_DIR + TableOutputTest.class.getSimpleName() + "/";
	
	private final static int rows = 50000;
	private final static int maxVal1 = 7, maxVal2 = 15; 
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "F" }) );
	}

	@Test
	public void testTableOutputSP1() 
	{
		runTableOutputTest(ExecType.SPARK, 0);
	}
	
	@Test
	public void testTableOutputSP2() 
	{
		runTableOutputTest(ExecType.SPARK, 5);
	}
	
	@Test
	public void testTableOutputSP3() 
	{
		runTableOutputTest(ExecType.SPARK, -5);
	}
	
	
	@Test
	public void testTableOutputCP1() 
	{
		runTableOutputTest(ExecType.CP, 0);
	}
	
	@Test
	public void testTableOutputCP2() 
	{
		runTableOutputTest(ExecType.CP, 5);
	}
	
	@Test
	public void testTableOutputCP3() 
	{
		runTableOutputTest(ExecType.CP, -5);
	}
	
	private void runTableOutputTest( ExecType et, int delta)
	{
		//rtplatform for MR
		ExecMode platformOld = rtplatform;

		switch( et ){
			case SPARK: rtplatform = ExecMode.SPARK; break;
			default: rtplatform = ExecMode.HYBRID; break;
		}
	
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == ExecMode.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);

			int dim1 = maxVal1 + delta;
			int dim2 = maxVal2 + delta;

			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain","-args", 
				input("A"), input("B"),
				Integer.toString(dim1), Integer.toString(dim2), output("F")};
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();
			
			//generate actual dataset (always dense because values <=0 invalid)
			double[][] A = TestUtils.floor(getRandomMatrix(rows, 1, 1, maxVal1, 1.0, -1)); 
			writeInputMatrixWithMTD("A", A, true);
			double[][] B = TestUtils.floor(getRandomMatrix(rows, 1, 1, maxVal2, 1.0, -1)); 
			writeInputMatrixWithMTD("B", B, true);
			
			runTest(true, false, null, -1); 
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("F");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("F");
			
			CellIndex tmp = new CellIndex(-1, -1);
			double dmlVal, rVal;
			int numErrors = 0;
			
			try {
			for(int i=1; i<=Math.min(dim1, maxVal1); i++) {
				for(int j=1; j<=Math.min(dim2, maxVal2); j++) {
					tmp.set(i, j);
					dmlVal = ( dmlfile.get(tmp) == null ? 0 : dmlfile.get(tmp) );
					rVal = ( rfile.get(tmp) == null ? 0 : rfile.get(tmp) );
					if ( dmlVal != rVal ) {
						System.err.println("  (" + i+","+j+ ") " + dmlVal + " != " + rVal);
						numErrors++;
					}
				}
			}
			} catch(Exception e) {
				e.printStackTrace();
			}
			Assert.assertEquals(0, numErrors);

			numErrors = 0;
			if ( delta > 0 ) {
				// check for correct padding in dmlfile
				for(int i=1; i<= delta; i++) {
					for(int j=1; j<=delta; j++) {
						tmp.set(maxVal1+i, maxVal2+j);
						dmlVal = ( dmlfile.get(tmp) == null ? 0 : dmlfile.get(tmp) );
						if(dmlVal != 0) {
							System.err.println("  Padding: (" + i+","+j+ ") " + dmlVal + " != 0");
							numErrors++;
						}
					}
				}
			}
			Assert.assertEquals(0, numErrors);
		}
		finally
		{
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
} 