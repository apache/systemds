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

package org.apache.sysml.test.integration.functions.unary.matrix;

import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

public class RemoveEmptySelTest extends AutomatedTestBase 
{
	private final static String TEST_NAME = "removeEmptySel";
	private final static String TEST_DIR = "functions/unary/matrix/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RemoveEmptySelTest.class.getSimpleName() + "/";

	private final static int rows = 1007;
	private final static int cols = 1005;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME, new String[] { "R" }) );
	}
	
	@Test
	public void testRemoveEmptyRowSelCP() {
		runTestRemoveEmptySel( "rows", ExecType.CP, false );
	}
	
	@Test
	public void testRemoveEmptyRowSelEmptyCP() {
		runTestRemoveEmptySel( "rows", ExecType.CP, true );
	}
	
	@Test
	public void testRemoveEmptyColSelCP() {
		runTestRemoveEmptySel( "cols", ExecType.CP, false );
	}
	
	@Test
	public void testRemoveEmptyColSelEmptyCP() {
		runTestRemoveEmptySel( "cols", ExecType.CP, true );
	}
	
	@Test
	public void testRemoveEmptyRowSelSP() {
		runTestRemoveEmptySel( "rows", ExecType.SPARK, false );
	}
	
	@Test
	public void testRemoveEmptyRowSelEmptySP() {
		runTestRemoveEmptySel( "rows", ExecType.SPARK, true );
	}
	
	@Test
	public void testRemoveEmptyColSelSP() {
		runTestRemoveEmptySel( "cols", ExecType.SPARK, false );
	}
	
	@Test
	public void testRemoveEmptyColSelEmptySP() {
		runTestRemoveEmptySel( "cols", ExecType.SPARK, true );
	}

	@Test
	public void testRemoveEmptyRowSelMR() {
		runTestRemoveEmptySel( "rows", ExecType.MR, false );
	}
	
	@Test
	public void testRemoveEmptyRowSelEmptyMR() {
		runTestRemoveEmptySel( "rows", ExecType.MR, true );
	}
	
	@Test
	public void testRemoveEmptyColSelMR() {
		runTestRemoveEmptySel( "cols", ExecType.MR, false );
	}
	
	@Test
	public void testRemoveEmptyColSelEmptyMR() {
		runTestRemoveEmptySel( "cols", ExecType.MR, true );
	}
	
	/**
	 * 
	 * @param testname
	 * @param margin
	 * @param et
	 * @param sparse
	 * @param empty
	 */
	private void runTestRemoveEmptySel( String margin, ExecType et, boolean empty )
	{		
		//rtplatform for MR
		RUNTIME_PLATFORM platformOld = rtplatform;
		switch( et ){
			case MR: rtplatform = RUNTIME_PLATFORM.HADOOP; break;
			case SPARK: rtplatform = RUNTIME_PLATFORM.SPARK; break;
			default: rtplatform = RUNTIME_PLATFORM.HYBRID; break;
		}
	
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == RUNTIME_PLATFORM.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		
		try
		{	
			//register test configuration
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			config.addVariable("rows", rows);
			config.addVariable("cols", cols);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-stats", "-args", Integer.toString(empty?0:1), 
					Integer.toString(rows), Integer.toString(cols), margin, output("R") };
		
			//run dml test
			runTest(true, false, null, -1);
	
			//compare expected dimensions
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			Double expectedRows = (double) (margin.equals("rows") ? rows/2 : rows);
			Double expectedCols = (double) (margin.equals("cols") ? cols/2 : cols);
			Assert.assertEquals("Wrong output nrow.", expectedRows, dmlfile.get(new CellIndex(1,1)));
			Assert.assertEquals("Wrong output ncol.", expectedCols, dmlfile.get(new CellIndex(2,1)));			
		}
		finally
		{
			//reset platform for additional tests
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}	
}