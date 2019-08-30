/*
 * Modifications Copyright 2019 Graz University of Technology
 *
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

public class MultiAggTmplTest extends AutomatedTestBase 
{
	private static final String TEST_NAME = "multiAggPattern";
	private static final String TEST_NAME1 = TEST_NAME+"1"; //min(X>7), max(X>7)
	private static final String TEST_NAME2 = TEST_NAME+"2"; //sum(X>7), sum((X>7)^2)
	private static final String TEST_NAME3 = TEST_NAME+"3"; //sum(X==7), sum(X==3)
	private static final String TEST_NAME4 = TEST_NAME+"4"; //sum(X*Y), sum(X^2), sum(Y^2)
	private static final String TEST_NAME5 = TEST_NAME+"5"; //sum(V*X), sum(Y*Z), sum(X+Y-Z)
	private static final String TEST_NAME6 = TEST_NAME+"6"; //min(X), max(X), sum(X)
	private static final String TEST_NAME7 = TEST_NAME+"7"; //t(X)%*%X, t(X)%*Y, t(Y)%*%Y
	
	private static final String TEST_DIR = "functions/codegen/";
	private static final String TEST_CLASS_DIR = TEST_DIR + MultiAggTmplTest.class.getSimpleName() + "/";
	private final static String TEST_CONF = "SystemDS-config-codegen.xml";
	private final static File   TEST_CONF_FILE = new File(SCRIPT_DIR + TEST_DIR, TEST_CONF);
	
	private static final double eps = Math.pow(10, -10);
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		for(int i=1; i<=7; i++)
			addTestConfiguration( TEST_NAME+i, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME+i, new String[] { String.valueOf(i) }) );
	}
	
	@Test	
	public void testCodegenMultiAggRewrite1CP() {
		testCodegenIntegration( TEST_NAME1, true, ExecType.CP );
	}

	@Test	
	public void testCodegenMultiAgg1CP() {
		testCodegenIntegration( TEST_NAME1, false, ExecType.CP );
	}
	
	@Test	
	public void testCodegenMultiAgg1Spark() {
		testCodegenIntegration( TEST_NAME1, false, ExecType.SPARK );
	}
	
	@Test	
	public void testCodegenMultiAggRewrite2CP() {
		testCodegenIntegration( TEST_NAME2, true, ExecType.CP );
	}

	@Test	
	public void testCodegenMultiAgg2CP() {
		testCodegenIntegration( TEST_NAME2, false, ExecType.CP );
	}
	
	@Test	
	public void testCodegenMultiAgg2Spark() {
		testCodegenIntegration( TEST_NAME2, false, ExecType.SPARK );
	}
	
	@Test	
	public void testCodegenMultiAggRewrite3CP() {
		testCodegenIntegration( TEST_NAME3, true, ExecType.CP );
	}

	@Test	
	public void testCodegenMultiAgg3CP() {
		testCodegenIntegration( TEST_NAME3, false, ExecType.CP );
	}
	
	@Test	
	public void testCodegenMultiAgg3Spark() {
		testCodegenIntegration( TEST_NAME3, false, ExecType.SPARK );
	}
	
	@Test	
	public void testCodegenMultiAggRewrite4CP() {
		testCodegenIntegration( TEST_NAME4, true, ExecType.CP );
	}

	@Test	
	public void testCodegenMultiAgg4CP() {
		testCodegenIntegration( TEST_NAME4, false, ExecType.CP );
	}
	
	@Test	
	public void testCodegenMultiAgg4Spark() {
		testCodegenIntegration( TEST_NAME4, false, ExecType.SPARK );
	}
	
	@Test	
	public void testCodegenMultiAggRewrite5CP() {
		testCodegenIntegration( TEST_NAME5, true, ExecType.CP );
	}

	@Test	
	public void testCodegenMultiAgg5CP() {
		testCodegenIntegration( TEST_NAME5, false, ExecType.CP );
	}
	
	@Test	
	public void testCodegenMultiAgg5Spark() {
		testCodegenIntegration( TEST_NAME5, false, ExecType.SPARK );
	}
	
	@Test	
	public void testCodegenMultiAggRewrite6CP() {
		testCodegenIntegration( TEST_NAME6, true, ExecType.CP );
	}

	@Test	
	public void testCodegenMultiAgg6CP() {
		testCodegenIntegration( TEST_NAME6, false, ExecType.CP );
	}
	
	@Test	
	public void testCodegenMultiAgg6Spark() {
		testCodegenIntegration( TEST_NAME6, false, ExecType.SPARK );
	}
	
	@Test	
	public void testCodegenMultiAggRewrite7CP() {
		testCodegenIntegration( TEST_NAME7, true, ExecType.CP );
	}

	@Test	
	public void testCodegenMultiAgg7CP() {
		testCodegenIntegration( TEST_NAME7, false, ExecType.CP );
	}
	
	@Test	
	public void testCodegenMultiAgg7Spark() {
		testCodegenIntegration( TEST_NAME7, false, ExecType.SPARK );
	}
	
	private void testCodegenIntegration( String testname, boolean rewrites, ExecType instType )
	{	
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		ExecMode platformOld = rtplatform;
		switch( instType ) {
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
			Assert.assertTrue(heavyHittersContainsSubString("spoofMA") 
					|| heavyHittersContainsSubString("sp_spoofMA"));
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
