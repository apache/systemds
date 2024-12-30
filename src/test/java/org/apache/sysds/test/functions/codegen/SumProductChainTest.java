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

public class SumProductChainTest extends AutomatedTestBase 
{	
	private static final Log LOG = LogFactory.getLog(SumProductChainTest.class.getName());
	
	private static final String TEST_NAME1 = "SumProductChain";
	private static final String TEST_NAME2 = "SumAdditionChain";
	private static final String TEST_DIR = "functions/codegen/";
	private static final String TEST_CLASS_DIR = TEST_DIR + SumProductChainTest.class.getSimpleName() + "/";
	private final static String TEST_CONF = "SystemDS-config-codegen.xml";
	private final static File   TEST_CONF_FILE = new File(SCRIPT_DIR + TEST_DIR, TEST_CONF);
	
	private static final int rows = 1191;
	private static final int cols1 = 1;
	private static final int cols2 = 31;
	private static final double sparsity1 = 0.9;
	private static final double sparsity2 = 0.09;
	private static final double eps = Math.pow(10, -10);
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "R" }) );
	}
		
	@Test
	public void testSumProductVectorsDense() {
		testSumProductChain( TEST_NAME1, true, false, false, ExecType.CP );
	}
	
	@Test
	public void testSumProductVectorsSparse() {
		testSumProductChain( TEST_NAME1, true, true, false, ExecType.CP );
	}
	
	@Test
	public void testSumProductMatrixDense() {
		testSumProductChain( TEST_NAME1, false, false, false, ExecType.CP );
	}
	
	@Test
	public void testSumProductMatrixSparse() {
		testSumProductChain( TEST_NAME1, false, true, false, ExecType.CP );
	}
	
	@Test
	public void testSumAdditionVectorsDense() {
		testSumProductChain( TEST_NAME2, true, false, false, ExecType.CP );
	}
	
	@Test
	public void testSumAdditionVectorsSparse() {
		testSumProductChain( TEST_NAME2, true, true, false, ExecType.CP );
	}
	
	@Test
	public void testSumAdditionMatrixDense() {
		testSumProductChain( TEST_NAME2, false, false, false, ExecType.CP );
	}
	
	@Test
	public void testSumAdditionMatrixSparse() {
		testSumProductChain( TEST_NAME2, false, true, false, ExecType.CP );
	}
	
	
	private void testSumProductChain(String testname, boolean vectors, boolean sparse, boolean rewrites, ExecType instType)
	{	
		boolean oldRewrites = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		ExecMode platformOld = setExecMode(instType);
		
		try
		{
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
			
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{"-explain", "codegen",
				"-stats", "-args", input("X"), output("R") };
			
			fullRScriptName = HOME + testname + ".R";
			rCmd = getRCmd(inputDir(), expectedDir());

			//generate input data
			int cols = vectors ? cols1 : cols2;
			double sparsity = sparse ? sparsity2 : sparsity1;
			double[][] X = getRandomMatrix(rows, cols, -1, 1, sparsity, 7);
			writeInputMatrixWithMTD("X", X, true);
			
			//run tests
			runTest(true, false, null, -1);
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("R");	
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			if( vectors || !sparse  )
				Assert.assertTrue(heavyHittersContainsSubString("spoofCell") 
						|| heavyHittersContainsSubString("sp_spoofCell"));
		}
		finally {
			resetExecMode(platformOld);
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
		LOG.debug("This test case overrides default configuration with " + TEST_CONF_FILE.getPath());
		return TEST_CONF_FILE;
	}
}
