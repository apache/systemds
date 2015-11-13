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


import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript;
import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;



/**
 * 
 */
public class RandVarMinMaxTest extends AutomatedTestBase 
{
	
	private final static String TEST_NAME_DML1 = "RandVarMinMax1";
	private final static String TEST_NAME_DML2 = "RandVarMinMax2";
	private final static String TEST_NAME_DML3 = "RandVarMinMax3";
	private final static String TEST_NAME_R = "RandVarMinMax";
	private final static String TEST_DIR = "functions/data/";
	private final static String TEST_CLASS_DIR = TEST_DIR + RandVarMinMaxTest.class.getSimpleName() + "/";
	
	private final static int rows = 2;
	private final static int cols = 100;
	
		
	@Override
	public void setUp() 
	{
		TestUtils.clearAssertionInformation();
		addTestConfiguration( TEST_NAME_DML1, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_DML1, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME_DML2, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_DML2, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME_DML3, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_DML3, new String[] { "R" }) );
	}

	@Test
	public void testMatrixVarMinMaxCP() {
		runRandVarMinMaxTest(TEST_NAME_DML1, ExecType.CP);
	}
	
	@Test
	public void testRandVarMinMaxCP() {
		runRandVarMinMaxTest(TEST_NAME_DML2, ExecType.CP);
	}
	
	@Test
	public void testMatrixVarExpressionCP() {
		runRandVarMinMaxTest(TEST_NAME_DML3, ExecType.CP);
	}
	
	@Test
	public void testMatrixVarMinMaxSP() {
		runRandVarMinMaxTest(TEST_NAME_DML1, ExecType.SPARK);
	}
	
	@Test
	public void testRandVarMinMaxSP() {
		runRandVarMinMaxTest(TEST_NAME_DML2, ExecType.SPARK);
	}
	
	@Test
	public void testMatrixVarExpressionSP() {
		runRandVarMinMaxTest(TEST_NAME_DML3, ExecType.SPARK);
	}
	
	@Test
	public void testMatrixVarMinMaxMR() {
		runRandVarMinMaxTest(TEST_NAME_DML1, ExecType.MR);
	}
	
	@Test
	public void testRandVarMinMaxMR() {		
		runRandVarMinMaxTest(TEST_NAME_DML2, ExecType.MR);
	}
	
	@Test
	public void testMatrixVarExpressionMR() {		
		runRandVarMinMaxTest(TEST_NAME_DML3, ExecType.MR);
	}

	/**
	 * 
	 * @param TEST_NAME
	 * @param instType
	 */
	private void runRandVarMinMaxTest( String TEST_NAME, ExecType instType )
	{
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
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullRScriptName = HOME + TEST_NAME_R + ".R";
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", 
					Integer.toString(rows), Integer.toString(cols), output("R") };
			
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
					Integer.toString(rows) + " " + Integer.toString(cols) + " " + expectedDir();
	
			runTest(true, false, null, -1);
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("R");
			TestUtils.compareMatrices(dmlfile, rfile, 0, "Stat-DML", "Stat-R");
		}
		finally
		{
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}
	
}
