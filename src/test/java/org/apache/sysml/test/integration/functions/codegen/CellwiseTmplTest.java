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
	private static final String TEST_NAME1 = "cellwisetmpl1";
	private static final String TEST_NAME2 = "cellwisetmpl2";
	private static final String TEST_NAME3 = "cellwisetmpl3";
	private static final String TEST_NAME4 = "cellwisetmpl4";
	private static final String TEST_NAME5 = "cellwisetmpl5";
	private static final String TEST_NAME6 = "cellwisetmpl6"; //sum

	private static final String TEST_DIR = "functions/codegen/";
	private static final String TEST_CLASS_DIR = TEST_DIR + CellwiseTmplTest.class.getSimpleName() + "/";
	private final static String TEST_CONF = "SystemML-config-codegen.xml";
	private final static File   TEST_CONF_FILE = new File(SCRIPT_DIR + TEST_DIR, TEST_CONF);
	
	private static final double eps = Math.pow(10, -10);
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "1" }) );
		addTestConfiguration( TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "2" }) );
		addTestConfiguration( TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] { "3" }) );
		addTestConfiguration( TEST_NAME4, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4, new String[] { "4" }) );
		addTestConfiguration( TEST_NAME5, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME5, new String[] { "5" }) );
		addTestConfiguration( TEST_NAME6, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME6, new String[] { "6" }) );
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
	public void testCodegenCellwiseRewrite1_sp() {
		testCodegenIntegration( TEST_NAME1, true, ExecType.SPARK );
	}
	
	private void testCodegenIntegration( String testname, boolean rewrites, ExecType instType )
	{	
		
		boolean oldRewrites = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		
		switch( instType ){
			case MR: rtplatform = RUNTIME_PLATFORM.HADOOP; break;
			case SPARK: 
				rtplatform = RUNTIME_PLATFORM.SPARK;
				DMLScript.USE_LOCAL_SPARK_CONFIG = true; 
				break;
			default: rtplatform = RUNTIME_PLATFORM.HYBRID; break;
		}
		
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
			//System.exit(1);
			if(testname.equals(TEST_NAME6)) //tak+
			{
				//compare scalars 
				HashMap<CellIndex, Double> dmlfile = readDMLScalarFromHDFS("S");
				HashMap<CellIndex, Double> rfile  = readRScalarFromFS("S");
				TestUtils.compareScalars((Double) dmlfile.values().toArray()[0], (Double) rfile.values().toArray()[0],0);
			}
			else
			{
				//compare matrices 
				HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("S");
				HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("S");	
				TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
				if( !(rewrites && testname.equals(TEST_NAME2)) ) //sigmoid
					Assert.assertTrue(heavyHittersContainsSubString("spoof") || heavyHittersContainsSubString("sp_spoof"));
			}
		}
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldRewrites;
			OptimizerUtils.ALLOW_AUTO_VECTORIZATION = true;
			OptimizerUtils.ALLOW_OPERATOR_FUSION = true;
		}
	}	

	/**
	 * Override default configuration with custom test configuration to ensure
	 * scratch space and local temporary directory locations are also updated.
	 */
	@Override
	protected File getConfigTemplateFile() {
		// Instrumentation in this test's output log to show custom configuration file used for template.
		System.out.println("This test case overrides default configuration with " + TEST_CONF_FILE.getPath());
		return TEST_CONF_FILE;
	}
}
