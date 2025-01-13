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

package org.apache.sysds.test.functions.codegen;

import java.io.File;
import java.util.HashMap;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

public class RowVectorComparisonTest extends AutomatedTestBase 
{
	private static final Log LOG = LogFactory.getLog(RowVectorComparisonTest.class.getName());

	private static final String TEST_NAME1 = "rowComparisonEq";
	private static final String TEST_NAME2 = "rowComparisonNeq";
	private static final String TEST_NAME3 = "rowComparisonLte";
	private static final String TEST_NAME4 = "rowComparisonLt";
	private static final String TEST_NAME5 = "rowComparisonGte";
	private static final String TEST_NAME6 = "rowComparisonGt";

	private static final String TEST_DIR = "functions/codegen/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RowVectorComparisonTest.class.getSimpleName() + "/";
	private final static String TEST_CONF = "SystemDS-config-codegen.xml";
	private final static File   TEST_CONF_FILE = new File(SCRIPT_DIR + TEST_DIR, TEST_CONF);
	
	private static final double eps = Math.pow(10, -10);
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME4, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME5, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME5, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME6, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME6, new String[] { "R" }) );
	}
	
	@Test	
	public void testCodegenRowComparison1() {
		testCodegenIntegration( TEST_NAME1, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowComparison2() {
		testCodegenIntegration( TEST_NAME2, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowComparison3() {
		testCodegenIntegration( TEST_NAME3, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowComparison4() {
		testCodegenIntegration( TEST_NAME4, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowComparison5() {
		testCodegenIntegration( TEST_NAME5, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowComparison6() {
		testCodegenIntegration( TEST_NAME6, false, ExecType.CP );
	}
	
	@Test	
	public void testCodegenRowComparisonRewrite1() {
		testCodegenIntegration( TEST_NAME1, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowComparisonRewrite2() {
		testCodegenIntegration( TEST_NAME2, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowComparisonRewrite3() {
		testCodegenIntegration( TEST_NAME3, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowComparisonRewrite4() {
		testCodegenIntegration( TEST_NAME4, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowComparisonRewrite5() {
		testCodegenIntegration( TEST_NAME5, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenRowComparisonRewrite6() {
		testCodegenIntegration( TEST_NAME6, true, ExecType.CP );
	}
	
	
	private void testCodegenIntegration( String testname, boolean rewrites, ExecType instType )
	{	
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		ExecMode platformOld = setExecMode(instType);
		setOutputBuffering(true);
		
		try
		{
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{"-stats", "-args", output("R") };
			
			fullRScriptName = HOME + testname + ".R";
			rCmd = getRCmd(inputDir(), expectedDir());

			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;

			runTest(true, false, null, -1); 
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("R");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			Assert.assertTrue(heavyHittersContainsSubString("spoofRA") 
					|| heavyHittersContainsSubString("sp_spoofRA"));
		}
		finally {
			resetExecMode(platformOld);
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
		LOG.debug("This test case overrides default configuration with " + TEST_CONF_FILE.getPath());
		return TEST_CONF_FILE;
	}
}
