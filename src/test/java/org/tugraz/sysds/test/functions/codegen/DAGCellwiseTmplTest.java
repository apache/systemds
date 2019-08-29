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

package org.tugraz.sysds.test.functions.codegen;

import java.io.File;
import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;
import org.tugraz.sysds.api.DMLScript;
import org.tugraz.sysds.common.Types.ExecMode;
import org.tugraz.sysds.hops.OptimizerUtils;
import org.tugraz.sysds.lops.LopProperties.ExecType;
import org.tugraz.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestConfiguration;
import org.tugraz.sysds.test.TestUtils;

public class DAGCellwiseTmplTest extends AutomatedTestBase 
{	
	private static final String TEST_NAME1 = "DAGcellwisetmpl1";
	private static final String TEST_NAME2 = "DAGcellwisetmpl2";
	private static final String TEST_NAME3 = "DAGcellwisetmpl3";
	
	private static final String TEST_DIR = "functions/codegen/";
	private static final String TEST_CLASS_DIR = TEST_DIR + DAGCellwiseTmplTest.class.getSimpleName() + "/";
	private final static String TEST_CONF = "SystemDS-config-codegen.xml";
	private final static File   TEST_CONF_FILE = new File(SCRIPT_DIR + TEST_DIR, TEST_CONF);
	
	private static final double eps = Math.pow(10, -10);
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "S" }) );
		addTestConfiguration( TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "S" }) );
		addTestConfiguration( TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] { "S" }) );
	}
		
	@Test
	public void testDAGMatrixCellwiseRewrite1() {
		testCodegenIntegration( TEST_NAME1, true, false, ExecType.CP );
	}
		
	@Test
	public void testDAGMatrixCellwiseRewrite2() {
		testCodegenIntegration( TEST_NAME2, true, false, ExecType.CP  );
	}
	
	@Test
	public void testDAGMatrixCellwiseRewrite3() {
		testCodegenIntegration( TEST_NAME3, true, false, ExecType.CP  );
	}

	@Test
	public void testDAGMatrixCellwise1() {
		testCodegenIntegration( TEST_NAME1, false, false, ExecType.CP );
	}
		
	@Test
	public void testDAGMatrixCellwise2() {
		testCodegenIntegration( TEST_NAME2, false, false, ExecType.CP  );
	}
	
	@Test
	public void testDAGMatrixCellwise3() {
		testCodegenIntegration( TEST_NAME3, false, false, ExecType.CP  );
	}

	@Test
	public void testDAGVectorCellwiseRewrite1() {
		testCodegenIntegration( TEST_NAME1, true, true, ExecType.CP );
	}
		
	@Test
	public void testDAGVectorCellwiseRewrite2() {
		testCodegenIntegration( TEST_NAME2, true, true, ExecType.CP  );
	}
	
	@Test
	public void testDAGVectorCellwiseRewrite3() {
		testCodegenIntegration( TEST_NAME3, true, true, ExecType.CP  );
	}

	@Test
	public void testDAGVectorCellwise1() {
		testCodegenIntegration( TEST_NAME1, false, true, ExecType.CP );
	}
		
	@Test
	public void testDAGVectorCellwise2() {
		testCodegenIntegration( TEST_NAME2, false, true, ExecType.CP  );
	}
	
	@Test
	public void testDAGVectorCellwise3() {
		testCodegenIntegration( TEST_NAME3, false, true, ExecType.CP  );
	}
	
	private void testCodegenIntegration( String testname, boolean rewrites, boolean vector, ExecType instType )
	{			
		boolean oldRewrites = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		ExecMode platformOld = rtplatform;
		switch( instType ){
			case SPARK: rtplatform = ExecMode.SPARK; break;
			default: rtplatform = ExecMode.HYBRID; break;
		}
	
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == ExecMode.SPARK || rtplatform == ExecMode.HYBRID )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		
		try
		{
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			
			int cols = vector ? 1 : 50;
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{"-explain", "hops", "-stats", 
					"-args", String.valueOf(cols), output("S") };
			
			fullRScriptName = HOME + testname + ".R";
			rCmd = getRCmd(String.valueOf(cols), expectedDir());			

			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;

			runTest(true, false, null, -1); 
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("S");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("S");	
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			Assert.assertTrue(heavyHittersContainsSubString("spoofCell") 
				|| heavyHittersContainsSubString("sp_spoofCell"));
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
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
