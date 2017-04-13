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

package org.apache.sysml.test.integration.functions.codegen;

import java.io.File;
import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

public class CellwiseTmplTest extends AutomatedTestBase 
{	
	private static final String TEST_NAME = "cellwisetmpl";
	private static final String TEST_NAME1 = TEST_NAME+1;
	private static final String TEST_NAME2 = TEST_NAME+2;
	private static final String TEST_NAME3 = TEST_NAME+3;
	private static final String TEST_NAME4 = TEST_NAME+4;
	private static final String TEST_NAME5 = TEST_NAME+5;
	private static final String TEST_NAME6 = TEST_NAME+6;
	private static final String TEST_NAME7 = TEST_NAME+7;
	private static final String TEST_NAME8 = TEST_NAME+8;
	private static final String TEST_NAME9 = TEST_NAME+9;   //sum((X + 7 * Y)^2)
	private static final String TEST_NAME10 = TEST_NAME+10; //min/max(X + 7 * Y)
	private static final String TEST_NAME11 = TEST_NAME+11; //replace((0 / (X - 500))+1, 0/0, 7)
	private static final String TEST_NAME12 = TEST_NAME+12; //((X/3) %% 0.6) + ((X/3) %/% 0.6)
	private static final String TEST_NAME13 = TEST_NAME+13; //min(X + 7 * Y) large
	private static final String TEST_NAME14 = TEST_NAME+14; //-2 * X + t(Y); t(Y) is rowvector
	
	private static final String TEST_DIR = "functions/codegen/";
	private static final String TEST_CLASS_DIR = TEST_DIR + CellwiseTmplTest.class.getSimpleName() + "/";
	private final static String TEST_CONF6 = "SystemML-config-codegen6.xml";
	private final static String TEST_CONF7 = "SystemML-config-codegen.xml";
	private static String TEST_CONF = TEST_CONF7;
	
	private static final double eps = Math.pow(10, -10);
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		for( int i=1; i<=14; i++ ) {
			addTestConfiguration( TEST_NAME+i, new TestConfiguration(
					TEST_CLASS_DIR, TEST_NAME+i, new String[] {String.valueOf(i)}) );
		}
	}
		
	@Test
	public void testCodegenCellwiseRewrite1() {
		testCodegenIntegration( TEST_NAME1, true, ExecType.CP );
	}
		
	@Test
	public void testCodegenCellwiseRewrite2() {
		testCodegenIntegration( TEST_NAME2, true, ExecType.CP  );
	}
	
	@Test
	public void testCodegenCellwiseRewrite3() {
		testCodegenIntegration( TEST_NAME3, true, ExecType.CP  );
	}
	
	@Test
	public void testCodegenCellwiseRewrite4() 
	{
		testCodegenIntegration( TEST_NAME4, true, ExecType.CP  );
	}
	
	@Test
	public void testCodegenCellwiseRewrite5() {
		testCodegenIntegration( TEST_NAME5, true, ExecType.CP  );
	}
	
	@Test
	public void testCodegenCellwiseRewrite6() {
		testCodegenIntegration( TEST_NAME6, true, ExecType.CP  );
	}
	
	@Test
	public void testCodegenCellwiseRewrite7() {
		testCodegenIntegration( TEST_NAME7, true, ExecType.CP  );
	}
	
	@Test
	public void testCodegenCellwiseRewrite8() {
		testCodegenIntegration( TEST_NAME8, true, ExecType.CP  );
	}
	
	@Test
	public void testCodegenCellwiseRewrite9() {
		testCodegenIntegration( TEST_NAME9, true, ExecType.CP  );
	}
	
	@Test
	public void testCodegenCellwiseRewrite10() {
		testCodegenIntegration( TEST_NAME10, true, ExecType.CP  );
	}
	
	@Test
	public void testCodegenCellwiseRewrite11() {
		testCodegenIntegration( TEST_NAME11, true, ExecType.CP  );
	}
	
	@Test
	public void testCodegenCellwiseRewrite12() {
		testCodegenIntegration( TEST_NAME12, true, ExecType.CP  );
	}
	
	@Test
	public void testCodegenCellwiseRewrite13() {
		testCodegenIntegration( TEST_NAME13, true, ExecType.CP  );
	}
	
	@Test
	public void testCodegenCellwiseRewrite14() {
		testCodegenIntegration( TEST_NAME14, true, ExecType.CP  );
	}

	@Test
	public void testCodegenCellwise1() {
		testCodegenIntegration( TEST_NAME1, false, ExecType.CP );
	}
		
	@Test
	public void testCodegenCellwise2() {
		testCodegenIntegration( TEST_NAME2, false, ExecType.CP  );
	}
	
	@Test
	public void testCodegenCellwise3() {
		testCodegenIntegration( TEST_NAME3, false, ExecType.CP  );
	}
	
	@Test
	public void testCodegenCellwise4() 
	{
		testCodegenIntegration( TEST_NAME4, false, ExecType.CP  );
	}
	
	@Test
	public void testCodegenCellwise5() {
		testCodegenIntegration( TEST_NAME5, false, ExecType.CP  );
	}
	
	@Test
	public void testCodegenCellwise6() {
		testCodegenIntegration( TEST_NAME6, false, ExecType.CP  );
	}
	
	@Test
	public void testCodegenCellwise7() {
		testCodegenIntegration( TEST_NAME7, false, ExecType.CP  );
	}
	
	@Test
	public void testCodegenCellwise8() {
		testCodegenIntegration( TEST_NAME8, false, ExecType.CP  );
	}
	
	@Test
	public void testCodegenCellwise9() {
		testCodegenIntegration( TEST_NAME9, false, ExecType.CP  );
	}
	
	@Test
	public void testCodegenCellwise10() {
		testCodegenIntegration( TEST_NAME10, false, ExecType.CP  );
	}
	
	@Test
	public void testCodegenCellwise11() {
		testCodegenIntegration( TEST_NAME11, false, ExecType.CP  );
	}
	
	@Test
	public void testCodegenCellwise12() {
		testCodegenIntegration( TEST_NAME12, false, ExecType.CP  );
	}
	
	@Test
	public void testCodegenCellwise13() {
		testCodegenIntegration( TEST_NAME13, false, ExecType.CP  );
	}
	
	@Test
	public void testCodegenCellwise14() {
		testCodegenIntegration( TEST_NAME14, false, ExecType.CP  );
	}

	@Test
	public void testCodegenCellwiseRewrite1_sp() {
		testCodegenIntegration( TEST_NAME1, true, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenCellwiseRewrite7_sp() {
		testCodegenIntegration( TEST_NAME7, true, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenCellwiseRewrite8_sp() {
		testCodegenIntegration( TEST_NAME8, true, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenCellwiseRewrite9_sp() {
		testCodegenIntegration( TEST_NAME9, true, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenCellwiseRewrite10_sp() {
		testCodegenIntegration( TEST_NAME10, true, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenCellwiseRewrite11_sp() {
		testCodegenIntegration( TEST_NAME11, true, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenCellwiseRewrite12_sp() {
		testCodegenIntegration( TEST_NAME12, true, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenCellwiseRewrite13_sp() {
		testCodegenIntegration( TEST_NAME13, true, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenCellwiseRewrite14_sp() {
		testCodegenIntegration( TEST_NAME14, true, ExecType.SPARK );
	}
	
	private void testCodegenIntegration( String testname, boolean rewrites, ExecType instType )
	{			
		boolean oldRewrites = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		String oldTestConf = TEST_CONF;
		RUNTIME_PLATFORM platformOld = rtplatform;
		switch( instType ){
			case MR: rtplatform = RUNTIME_PLATFORM.HADOOP; break;
			case SPARK: rtplatform = RUNTIME_PLATFORM.SPARK; break;
			default: rtplatform = RUNTIME_PLATFORM.HYBRID_SPARK; break;
		}
	
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == RUNTIME_PLATFORM.SPARK || rtplatform == RUNTIME_PLATFORM.HYBRID_SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		
		if( testname.equals(TEST_NAME9) )
			TEST_CONF = TEST_CONF6;
		
		try
		{
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{"-explain", "runtime", "-stats", "-args", output("S") };
			
			fullRScriptName = HOME + testname + ".R";
			rCmd = getRCmd(inputDir(), expectedDir());			

			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;

			runTest(true, false, null, -1); 
			runRScript(true); 
			
			if(testname.equals(TEST_NAME6) || testname.equals(TEST_NAME7) 
				|| testname.equals(TEST_NAME9) || testname.equals(TEST_NAME10) ) {
				//compare scalars 
				HashMap<CellIndex, Double> dmlfile = readDMLScalarFromHDFS("S");
				HashMap<CellIndex, Double> rfile  = readRScalarFromFS("S");
				TestUtils.compareScalars((Double) dmlfile.values().toArray()[0], (Double) rfile.values().toArray()[0],0);
			}
			else {
				//compare matrices 
				HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("S");
				HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("S");	
				TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			}
			
			if( !(rewrites && testname.equals(TEST_NAME2)) ) //sigmoid
				Assert.assertTrue(heavyHittersContainsSubString(
						"spoofCell", "sp_spoofCell", "spoofMA", "sp_spoofMA"));
			if( testname.equals(TEST_NAME7) ) //ensure matrix mult is fused
				Assert.assertTrue(!heavyHittersContainsSubString("tsmm"));
			else if( testname.equals(TEST_NAME10) ) //ensure min/max is fused
				Assert.assertTrue(!heavyHittersContainsSubString("uamin","uamax"));
			else if( testname.equals(TEST_NAME11) ) //ensure replace is fused
				Assert.assertTrue(!heavyHittersContainsSubString("replace"));	
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldRewrites;
			OptimizerUtils.ALLOW_AUTO_VECTORIZATION = true;
			OptimizerUtils.ALLOW_OPERATOR_FUSION = true;
			TEST_CONF = oldTestConf;
		}
	}	

	/**
	 * Override default configuration with custom test configuration to ensure
	 * scratch space and local temporary directory locations are also updated.
	 */
	@Override
	protected File getConfigTemplateFile() {
		// Instrumentation in this test's output log to show custom configuration file used for template.
		File TEST_CONF_FILE = new File(SCRIPT_DIR + TEST_DIR, TEST_CONF);
		System.out.println("This test case overrides default configuration with " + TEST_CONF_FILE.getPath());
		return TEST_CONF_FILE;
	}
}
