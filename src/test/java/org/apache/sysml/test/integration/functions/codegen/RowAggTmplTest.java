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

public class RowAggTmplTest extends AutomatedTestBase 
{
	private static final String TEST_NAME = "rowAggPattern";
	private static final String TEST_NAME1 = TEST_NAME+"1"; //t(X)%*%(X%*%(lamda*v))
	private static final String TEST_NAME2 = TEST_NAME+"2"; //t(X)%*%(lamda*(X%*%v))
	private static final String TEST_NAME3 = TEST_NAME+"3"; //t(X)%*%(z+(2-(w*(X%*%v))))
	private static final String TEST_NAME4 = TEST_NAME+"4"; //colSums(X/rowSums(X))
	private static final String TEST_NAME5 = TEST_NAME+"5"; //t(X)%*%((P*(1-P))*(X%*%v));
	private static final String TEST_NAME6 = TEST_NAME+"6"; //t(X)%*%((P[,1]*(1-P[,1]))*(X%*%v));
	private static final String TEST_NAME7 = TEST_NAME+"7"; //t(X)%*%(X%*%v-y); sum((X%*%v-y)^2);
	private static final String TEST_NAME8 = TEST_NAME+"8"; //colSums((X/rowSums(X))>0.7)
	private static final String TEST_NAME9 = TEST_NAME+"9"; //t(X) %*% (v - abs(y))
	
	private static final String TEST_DIR = "functions/codegen/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RowAggTmplTest.class.getSimpleName() + "/";
	private final static String TEST_CONF = "SystemML-config-codegen.xml";
	private final static File   TEST_CONF_FILE = new File(SCRIPT_DIR + TEST_DIR, TEST_CONF);
	
	private static final double eps = Math.pow(10, -10);
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		for(int i=1; i<=9; i++)
			addTestConfiguration( TEST_NAME+i, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME+i, new String[] { String.valueOf(i) }) );
	}
	
	@Test	
	public void testCodegenRowAggRewrite1() {
		testCodegenIntegration( TEST_NAME1, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAggRewrite2() {
		testCodegenIntegration( TEST_NAME2, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAggRewrite3() {
		testCodegenIntegration( TEST_NAME3, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAggRewrite4() {
		testCodegenIntegration( TEST_NAME4, true, ExecType.CP );	
	}
	
	@Test
	public void testCodegenRowAggRewrite5() {
		testCodegenIntegration( TEST_NAME5, true, ExecType.CP );	
	}
	
	@Test
	public void testCodegenRowAggRewrite6() {
		testCodegenIntegration( TEST_NAME6, true, ExecType.CP );	
	}
	
	@Test
	public void testCodegenRowAggRewrite7() {
		testCodegenIntegration( TEST_NAME7, true, ExecType.CP );	
	}
	
	@Test
	public void testCodegenRowAggRewrite8() {
		testCodegenIntegration( TEST_NAME8, true, ExecType.CP );	
	}
	
	@Test
	public void testCodegenRowAggRewrite9() {
		testCodegenIntegration( TEST_NAME9, true, ExecType.CP );	
	}
	
	@Test	
	public void testCodegenRowAgg1() {
		testCodegenIntegration( TEST_NAME1, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg2() {
		testCodegenIntegration( TEST_NAME2, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg3() {
		testCodegenIntegration( TEST_NAME3, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowAgg4() {
		testCodegenIntegration( TEST_NAME4, false, ExecType.CP );	
	}
	
	@Test
	public void testCodegenRowAgg5() {
		testCodegenIntegration( TEST_NAME5, false, ExecType.CP );	
	}
	
	@Test
	public void testCodegenRowAgg6() {
		testCodegenIntegration( TEST_NAME6, false, ExecType.CP );	
	}
	
	@Test
	public void testCodegenRowAgg7() {
		testCodegenIntegration( TEST_NAME7, false, ExecType.CP );	
	}
	
	@Test
	public void testCodegenRowAgg8() {
		testCodegenIntegration( TEST_NAME8, false, ExecType.CP );	
	}
	
	@Test
	public void testCodegenRowAgg9() {
		testCodegenIntegration( TEST_NAME9, false, ExecType.CP );	
	}
	
	private void testCodegenIntegration( String testname, boolean rewrites, ExecType instType )
	{	
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		RUNTIME_PLATFORM platformOld = rtplatform;
		switch( instType ) {
			case MR: rtplatform = RUNTIME_PLATFORM.HADOOP; break;
			case SPARK: rtplatform = RUNTIME_PLATFORM.SPARK; break;
			default: rtplatform = RUNTIME_PLATFORM.HYBRID_SPARK; break;
		}

		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == RUNTIME_PLATFORM.SPARK || rtplatform == RUNTIME_PLATFORM.HYBRID_SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;

		try
		{
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{"-explain", "-stats", "-args", output("S") };
			
			fullRScriptName = HOME + testname + ".R";
			rCmd = getRCmd(inputDir(), expectedDir());			

			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;

			runTest(true, false, null, -1); 
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("S");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("S");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			Assert.assertTrue(heavyHittersContainsSubString("spoofRA") 
					|| heavyHittersContainsSubString("sp_spoofRA"));
		}
		finally {
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
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
