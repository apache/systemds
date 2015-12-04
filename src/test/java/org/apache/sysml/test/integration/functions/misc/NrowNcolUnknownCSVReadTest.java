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

import org.junit.Test;

import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.runtime.util.MapReduceTool;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

/**
 *  This test checks for valid meta information after csv read with unknown size
 *  in singlenode exec mode (w/o reblock).
 *  
 */
public class NrowNcolUnknownCSVReadTest extends AutomatedTestBase
{
	
	private final static String TEST_DIR = "functions/misc/";

	private final static String TEST_NAME1 = "NrowUnknownCSVTest";
	private final static String TEST_NAME2 = "NcolUnknownCSVTest";
	private final static String TEST_NAME3 = "LengthUnknownCSVTest";
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_DIR, TEST_NAME1, new String[] {}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_DIR, TEST_NAME2, new String[] {}));
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_DIR, TEST_NAME3, new String[] {}));
	}
	
	@Test
	public void testNrowUnknownCSV() 
	{ 
		runNxxUnkownCSVTest( TEST_NAME1 ); 
	}
	
	@Test
	public void testNcolUnknownCSV() 
	{ 
		runNxxUnkownCSVTest( TEST_NAME2 ); 
	}
	
	@Test
	public void testLengthUnknownCSV() 
	{ 
		runNxxUnkownCSVTest( TEST_NAME3 ); 
	}
	
	
	/**
	 * 
	 * @param cfc
	 * @param vt
	 */
	private void runNxxUnkownCSVTest( String testName ) 
	{
		String TEST_NAME = testName;
		RUNTIME_PLATFORM oldplatform = rtplatform;
		
		try
		{	
			rtplatform = RUNTIME_PLATFORM.SINGLE_NODE;
			
			//test configuration
			TestConfiguration config = getTestConfiguration(TEST_NAME);
		    String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", HOME + INPUT_DIR + "A"};
			
			loadTestConfiguration(config);
			
			//generate input data w/o mtd file (delete any mtd file from previous runs)
			int rows = 1034;
			int cols = 73;
			double[][] A = getRandomMatrix(rows, cols, 0, 1, 1.0, 7);
			MatrixBlock mb = DataConverter.convertToMatrixBlock(A);
			DataConverter.writeMatrixToHDFS(mb, HOME + INPUT_DIR + "A", OutputInfo.CSVOutputInfo, 
					                        new MatrixCharacteristics(rows,cols,-1,-1));
	        MapReduceTool.deleteFileIfExistOnHDFS(HOME + INPUT_DIR + "A.mtd");
			
			//run tests
	        runTest(true, false, null, -1);
		}
		catch(Exception ex)
		{
			throw new RuntimeException(ex);
		}
		finally
		{
			rtplatform = oldplatform;
		}
	}
}
