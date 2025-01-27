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
import org.apache.sysds.common.Opcodes;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;
import org.junit.Test;

public class OuterProdTmplTest extends AutomatedTestBase 
{
	private static final Log LOG = LogFactory.getLog(OuterProdTmplTest.class.getName());
	private static final String TEST_NAME1 = "wdivmm";
	private static final String TEST_NAME2 = "wdivmmRight";
	private static final String TEST_NAME3 = "wsigmoid";
	private static final String TEST_NAME4 = "wcemm";
	private static final String TEST_NAME5 = "wdivmmRightNotranspose";
	private static final String TEST_NAME6 = "wdivmmbasic";
	private static final String TEST_NAME7 = "wdivmmTransposeOut";
	private static final String TEST_NAME8 = "wSparseUnsafeOuterProduct";
	private static final String TEST_NAME9 = "wdivmmNeq";
	private static final String TEST_NAME10 = "rmseDist";
	private static final String TEST_NAME11 = "rmseDist2";
	
	private static final String TEST_DIR = "functions/codegen/";
	private static final String TEST_CLASS_DIR = TEST_DIR + OuterProdTmplTest.class.getSimpleName() + "/";
	private final static String TEST_CONF = "SystemDS-config-codegen.xml";
	private final static File   TEST_CONF_FILE = new File(SCRIPT_DIR + TEST_DIR, TEST_CONF);
	
	private static final double eps = Math.pow(10, -8);
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "1" }) );
		addTestConfiguration( TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "2" }) );
		addTestConfiguration( TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] { "3" }) );
		addTestConfiguration( TEST_NAME4, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4, new String[] { "4" }) );
		addTestConfiguration( TEST_NAME5, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME5, new String[] { "5" }) );
		addTestConfiguration( TEST_NAME6, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME6, new String[] { "6" }) );
		addTestConfiguration( TEST_NAME7, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME7, new String[] { "7" }) );
		addTestConfiguration( TEST_NAME8, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME8, new String[] { "8" }) );
		addTestConfiguration( TEST_NAME9, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME9, new String[] { "9" }) );
		addTestConfiguration( TEST_NAME10, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME10, new String[] { "10" }) );
		addTestConfiguration( TEST_NAME11, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME11, new String[] { "11" }) );
	}
	
	@Test
	public void testCodegenOuterProdRewrite1() {
		testCodegenIntegrationWithInput( TEST_NAME1, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenOuterProdRewrite2()  {
		testCodegenIntegration( TEST_NAME2, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenOuterProdRewrite3() {
		testCodegenIntegration( TEST_NAME3, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenOuterProdRewrite4() {
		testCodegenIntegrationWithInput( TEST_NAME4, true, ExecType.CP );
	}

	@Test
	public void testCodegenOuterProdRewrite5() {
		testCodegenIntegration( TEST_NAME5, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenOuterProdRewrite6() {
		testCodegenIntegration( TEST_NAME6, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenOuterProdRewrite7() {
		testCodegenIntegration( TEST_NAME7, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenOuterProdRewrite8() {
		testCodegenIntegration( TEST_NAME8, true, ExecType.CP );
	}

	@Test
	public void testCodegenOuterProd1() {
		testCodegenIntegrationWithInput( TEST_NAME1, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenOuterProd2()  {
		testCodegenIntegration( TEST_NAME2, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenOuterProd3() {
		testCodegenIntegration( TEST_NAME3, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenOuterProd4() {
		testCodegenIntegrationWithInput( TEST_NAME4, false, ExecType.CP );
	}

	@Test
	public void testCodegenOuterProd5() {
		testCodegenIntegration( TEST_NAME5, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenOuterProd6() {
		testCodegenIntegration( TEST_NAME6, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenOuterProd7() {
		testCodegenIntegration( TEST_NAME7, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenOuterProd8() {
		testCodegenIntegration( TEST_NAME8, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenOuterProdRewrite1_sp() {
		testCodegenIntegrationWithInput( TEST_NAME1, true, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenOuterProdRewrite2_sp() {
		testCodegenIntegration( TEST_NAME2, true, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenOuterProdRewrite3_sp() {
		testCodegenIntegration( TEST_NAME3, true, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenOuterProdRewrite4_sp() {
		testCodegenIntegrationWithInput( TEST_NAME4, true, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenOuterProdRewrite8_sp() {
		testCodegenIntegrationWithInput( TEST_NAME8, true, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenOuterProdRewrite9() {
		testCodegenIntegration( TEST_NAME9, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenOuterProd9() {
		testCodegenIntegration( TEST_NAME9, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenOuterProdRewrite9_sp() { 
		testCodegenIntegrationWithInput( TEST_NAME9, true, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenOuterProd10NoRewrite() {
		testCodegenIntegration( TEST_NAME10, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenOuterProd10NoRewriteSP() {
		testCodegenIntegrationWithInput( TEST_NAME10, false, ExecType.SPARK );
	}

	@Test
	public void testCodegenOuterProd11NoRewrite() {
		testCodegenIntegration( TEST_NAME11, false, ExecType.CP );
	}

	private void testCodegenIntegration( String testname, boolean rewrites, ExecType instType )
	{
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		ExecMode platformOld = setExecMode(instType);

		try
		{
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{"-explain", "-stats", "-args", output("S")};
			
			fullRScriptName = HOME + testname + ".R";
			rCmd = getRCmd(inputDir(), expectedDir());

			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;

			runTest(true, false, null, -1);
			runRScript(true);
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("S");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("S");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			
			if( testname.equals(TEST_NAME8) )
				Assert.assertTrue(!(heavyHittersContainsSubString("spoofOP")
						|| heavyHittersContainsSubString("sp_spoofOP")));
			else if( !rewrites ) {
				Assert.assertTrue(heavyHittersContainsSubString("spoofOP")
						|| heavyHittersContainsSubString("sp_spoofOP"));
				if( testname.equals(TEST_NAME9) )
					Assert.assertTrue(!heavyHittersContainsSubString(Opcodes.NOTEQUAL.toString()));
			}
		}
		finally {
			resetExecMode(platformOld);
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
			OptimizerUtils.ALLOW_AUTO_VECTORIZATION = true;
			OptimizerUtils.ALLOW_OPERATOR_FUSION = true;
		}
	}

	private void testCodegenIntegrationWithInput( String testname, boolean rewrites, ExecType instType )
	{
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		ExecMode platformOld = setExecMode(instType);

		try
		{
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			
			//generate actual dataset 
			double[][] A = getRandomMatrix(2000, 2000, -0.05, 1, 0.1, 6);
			writeInputMatrixWithMTD("A", A, true);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{"-stats", "-args", output("S"), input("A")};
			
			fullRScriptName = HOME + testname + ".R";
			rCmd = getRCmd(inputDir(), expectedDir());

			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;

			runTest(true, false, null, -1);
			runRScript(true);
			
			if(testname.equals(TEST_NAME4)) { //wcemm
				//compare scalars 
				HashMap<CellIndex, Double> dmlfile = readDMLScalarFromOutputDir("S");
				HashMap<CellIndex, Double> rfile  = readRScalarFromExpectedDir("S");
				TestUtils.compareScalars((Double) dmlfile.values().toArray()[0], (Double) rfile.values().toArray()[0],0.0001);
			}
			else {
				//compare matrices 
				HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("S");
				HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("S");
				TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			}
			
			if( testname.equals(TEST_NAME8) )
				Assert.assertTrue(!(heavyHittersContainsSubString("spoofOP")
					|| heavyHittersContainsSubString("sp_spoofOP")));
			else if( !rewrites )
				Assert.assertTrue(heavyHittersContainsSubString("spoofOP")
					|| heavyHittersContainsSubString("sp_spoofOP"));
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
