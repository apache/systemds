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

package org.apache.sysds.test.functions.vect;

import java.util.HashMap;

import org.junit.Test;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;

/**
 *   
 */
public class AutoVectorizationTest extends AutomatedTestBase
{
	
	private final static String TEST_DIR = "functions/vect/";
	private final static String TEST_CLASS_DIR = TEST_DIR + AutoVectorizationTest.class.getSimpleName() + "/";

	private final static String TEST_NAME1 = "VectorizeRixRowPos";
	private final static String TEST_NAME2 = "VectorizeRixRowNeg";
	private final static String TEST_NAME3 = "VectorizeRixColPos";
	private final static String TEST_NAME4 = "VectorizeRixColNeg";
	private final static String TEST_NAME5 = "VectorizeLixRowPos";
	private final static String TEST_NAME6 = "VectorizeLixRowNeg";
	private final static String TEST_NAME7 = "VectorizeLixColPos";
	private final static String TEST_NAME8 = "VectorizeLixColNeg";
	private final static String TEST_NAME9 = "VectorizeForLoopLeftScalarRowPos";
	private final static String TEST_NAME10 = "VectorizeForLoopLeftScalarRowNeg";
	private final static String TEST_NAME11 = "VectorizeForLoopLeftScalarColPos";
	private final static String TEST_NAME12 = "VectorizeForLoopLeftScalarColNeg";
	private final static String TEST_NAME13 = "VectorizeForLoopRightScalarRowPos";
	private final static String TEST_NAME14 = "VectorizeForLoopRightScalarRowNeg";
	private final static String TEST_NAME15 = "VectorizeForLoopRightScalarColPos";
	private final static String TEST_NAME16 = "VectorizeForLoopRightScalarColNeg";
	private final static String TEST_NAME17 = "VectorizeForLoopUnaryRowPos";
	private final static String TEST_NAME18 = "VectorizeForLoopUnaryRowNeg";
	private final static String TEST_NAME19 = "VectorizeForLoopUnaryColPos";
	private final static String TEST_NAME20 = "VectorizeForLoopUnaryColNeg";
	private final static String TEST_NAME21 = "VectorizeForLoopBinaryRowPos";
	private final static String TEST_NAME22 = "VectorizeForLoopBinaryRowNeg";
	private final static String TEST_NAME23 = "VectorizeForLoopBinaryColPos";
	private final static String TEST_NAME24 = "VectorizeForLoopBinaryColNeg";
	
	private final static int rows = 20;
	private final static int cols = 15;
	
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"R"}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] {"R"}));
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] {"R"}));
		addTestConfiguration(TEST_NAME4, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4, new String[] {"R"}));
		addTestConfiguration(TEST_NAME5, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME5, new String[] {"R"}));
		addTestConfiguration(TEST_NAME6, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME6, new String[] {"R"}));
		addTestConfiguration(TEST_NAME7, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME7, new String[] {"R"}));
		addTestConfiguration(TEST_NAME8, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME8, new String[] {"R"}));
		addTestConfiguration(TEST_NAME9, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME9, new String[] {"R"}));
		addTestConfiguration(TEST_NAME10, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME10, new String[] {"R"}));
		addTestConfiguration(TEST_NAME11, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME11, new String[] {"R"}));
		addTestConfiguration(TEST_NAME12, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME12, new String[] {"R"}));
		addTestConfiguration(TEST_NAME13, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME13, new String[] {"R"}));
		addTestConfiguration(TEST_NAME14, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME14, new String[] {"R"}));
		addTestConfiguration(TEST_NAME15, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME15, new String[] {"R"}));
		addTestConfiguration(TEST_NAME16, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME16, new String[] {"R"}));
		addTestConfiguration(TEST_NAME17, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME17, new String[] {"R"}));
		addTestConfiguration(TEST_NAME18, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME18, new String[] {"R"}));
		addTestConfiguration(TEST_NAME19, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME19, new String[] {"R"}));
		addTestConfiguration(TEST_NAME20, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME20, new String[] {"R"}));
		addTestConfiguration(TEST_NAME21, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME21, new String[] {"R"}));
		addTestConfiguration(TEST_NAME22, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME22, new String[] {"R"}));
		addTestConfiguration(TEST_NAME23, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME23, new String[] {"R"}));
		addTestConfiguration(TEST_NAME24, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME24, new String[] {"R"}));
	}
	
	@Test
	public void testVectorizeRixRowPos() { 
		runVectorizationTest( TEST_NAME1 ); 
	}
	
	@Test
	public void testVectorizeRixRowNeg() { 
		runVectorizationTest( TEST_NAME2 ); 
	}
	
	@Test
	public void testVectorizeRixColPos() { 
		runVectorizationTest( TEST_NAME3 ); 
	}
	
	@Test
	public void testVectorizeRixColNeg() { 
		runVectorizationTest( TEST_NAME4 ); 
	}
	
	@Test
	public void testVectorizeLixRowPos() { 
		runVectorizationTest( TEST_NAME5 ); 
	}
	
	@Test
	public void testVectorizeLixRowNeg() { 
		runVectorizationTest( TEST_NAME6 ); 
	}
	
	@Test
	public void testVectorizeLixColPos() { 
		runVectorizationTest( TEST_NAME7 ); 
	}
	
	@Test
	public void testVectorizeLixColNeg() { 
		runVectorizationTest( TEST_NAME8 ); 
	}
	
	@Test
	public void testVectorizeForLoopLeftScalarRowPos() { 
		runVectorizationTest( TEST_NAME9 ); 
	}
	
	@Test
	public void testVectorizeForLoopLeftScalarRowNeg() { 
		runVectorizationTest( TEST_NAME10 ); 
	}
	
	@Test
	public void testVectorizeForLoopLeftScalarColPos() { 
		runVectorizationTest( TEST_NAME11 ); 
	}
	
	@Test
	public void testVectorizeForLoopLeftScalarColNeg() { 
		runVectorizationTest( TEST_NAME12 ); 
	}
	
	@Test
	public void testVectorizeForLoopRightScalarRowPos() { 
		runVectorizationTest( TEST_NAME13 ); 
	}
	
	@Test
	public void testVectorizeForLoopRightScalarRowNeg() { 
		runVectorizationTest( TEST_NAME14 ); 
	}
	
	@Test
	public void testVectorizeForLoopRightScalarColPos() { 
		runVectorizationTest( TEST_NAME15 ); 
	}
	
	@Test
	public void testVectorizeForLoopRightScalarColNeg() { 
		runVectorizationTest( TEST_NAME16 ); 
	}
	
	@Test
	public void testVectorizeForLoopUnaryRowPos() { 
		runVectorizationTest( TEST_NAME17 ); 
	}
	
	@Test
	public void testVectorizeForLoopUnaryRowNeg() { 
		runVectorizationTest( TEST_NAME18 ); 
	}
	
	@Test
	public void testVectorizeForLoopUnaryColPos() { 
		runVectorizationTest( TEST_NAME19 ); 
	}
	
	@Test
	public void testVectorizeForLoopUnaryColNeg() { 
		runVectorizationTest( TEST_NAME20 ); 
	}
	
	@Test
	public void testVectorizeForLoopBinaryRowPos() { 
		runVectorizationTest( TEST_NAME21 ); 
	}
	
	@Test
	public void testVectorizeForLoopBinaryRowNeg() { 
		runVectorizationTest( TEST_NAME22 ); 
	}
	
	@Test
	public void testVectorizeForLoopBinaryColPos() { 
		runVectorizationTest( TEST_NAME23 ); 
	}
	
	@Test
	public void testVectorizeForLoopBinaryColNeg() { 
		runVectorizationTest( TEST_NAME24 ); 
	}
	
	private void runVectorizationTest( String testName ) 
	{
		String TEST_NAME = testName;
		
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);

			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain","-args", input("A"), output("R") };
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = getRCmd(inputDir(), expectedDir());
			
			//generate input
			double[][] A = getRandomMatrix(rows, cols, 0, 1, 1.0, 7);
			writeInputMatrixWithMTD("A", A, true);	
			
			//run tests
			runTest(true, false, null, -1);
			runRScript(true);
			
			//compare results
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("R");
			TestUtils.compareMatrices(dmlfile, rfile, 1e-14, "DML", "R");		
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);
		}
	}
}
