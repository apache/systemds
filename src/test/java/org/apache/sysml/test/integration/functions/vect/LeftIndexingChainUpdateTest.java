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

package org.apache.sysml.test.integration.functions.vect;

import java.util.HashMap;

import org.junit.Test;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

/**
 *   
 */
public class LeftIndexingChainUpdateTest extends AutomatedTestBase
{	
	private final static String TEST_DIR = "functions/vect/";
	private final static String TEST_CLASS_DIR = TEST_DIR + LeftIndexingChainUpdateTest.class.getSimpleName() + "/";

	private final static String TEST_NAME1 = "VectorizeLixChainRow";
	private final static String TEST_NAME2 = "VectorizeLixChainCol";
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"R"}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] {"R"}));
	}
	
	@Test
	public void testVectorizeLixRowPos() { 
		runVectorizationTest( TEST_NAME1, true ); 
	}
	
	@Test
	public void testVectorizeLixRowNoRewrite() { 
		runVectorizationTest( TEST_NAME1, false ); 
	}
	
	@Test
	public void testVectorizeLixColPos() { 
		runVectorizationTest( TEST_NAME2, true ); 
	}
	
	@Test
	public void testVectorizeLixColNoRewrite() { 
		runVectorizationTest( TEST_NAME2, false ); 
	}
	
	/**
	 * 
	 * @param cfc
	 * @param vt
	 */
	private void runVectorizationTest( String testName, boolean rewrites ) 
	{
		String TEST_NAME = testName;
		boolean flag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		
		try
		{		
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
			
		    String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", output("R") };
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = getRCmd(inputDir(), expectedDir());		
			
			//run tests
	        runTest(true, false, null, -1);
	        runRScript(true);
	        
	        //compare results
	        HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("R");
			TestUtils.compareMatrices(dmlfile, rfile, 1e-14, "DML", "R");		
		}
		catch(Exception ex) {
			throw new RuntimeException(ex);
		}
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = flag;
		}
	}
}
