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

public class RowConv2DOperationsTest extends AutomatedTestBase
{
	private static final Log LOG = LogFactory.getLog(RowConv2DOperationsTest.class.getName());

	private final static String TEST_NAME1 = "RowConv2DTest";
	private final static String TEST_DIR = "functions/codegen/";
	private final static String TEST_CLASS_DIR = TEST_DIR + RowConv2DOperationsTest.class.getSimpleName() + "/";
	
	private final static String TEST_CONF = "SystemDS-config-codegen.xml";
	private final static File   TEST_CONF_FILE = new File(SCRIPT_DIR + TEST_DIR, TEST_CONF);

	private static final double eps = Math.pow(10, -10);

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"B"}));
	}

	@Test
	public void testConv2DDenseDenseCP() {
		runConv2DTest(TEST_NAME1, true, 16, 64, 1, 3, 2, 1, 0, false, false, ExecType.CP);
	}
	
	@Test
	public void testConv2DSparseDenseCP() {
		runConv2DTest(TEST_NAME1, true, 16, 64, 1, 3, 2, 1, 0, true, false, ExecType.CP);
	}

	@Test
	public void testConv2DDenseDenseSP() {
		runConv2DTest(TEST_NAME1, true, 16, 64, 1, 3, 2, 1, 0, false, false, ExecType.SPARK);
	}
	
	@Test
	public void testConv2DSparseDenseSP() {
		runConv2DTest(TEST_NAME1, true, 16, 64, 1, 3, 2, 1, 0, true, false, ExecType.SPARK);
	}
	
	public void runConv2DTest(String testname, boolean rewrites, int imgSize, int numImg, int numChannels,
		int numFilters, int filterSize, int stride, int pad, boolean sparse1, boolean sparse2, ExecType et)
	{
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		ExecMode platformOld = setExecMode(et);
		
		try
		{
			String sparseVal1 = String.valueOf(sparse1).toUpperCase();
			String sparseVal2 = String.valueOf(sparse2).toUpperCase();
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{"recompile_runtime", "-stats", "-args",
				String.valueOf(imgSize), String.valueOf(numImg), String.valueOf(numChannels),
				String.valueOf(numFilters), String.valueOf(filterSize), String.valueOf(stride),
				String.valueOf(pad), output("B"), sparseVal1, sparseVal2 };
			
			fullRScriptName = HOME + testname + ".R";
			rCmd = getRCmd(String.valueOf(imgSize), String.valueOf(numImg), String.valueOf(numChannels),
				String.valueOf(numFilters), String.valueOf(filterSize), String.valueOf(stride),
				String.valueOf(pad), expectedDir(), sparseVal1, sparseVal2);

			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
			
			runTest(true, false, null, -1);
			runRScript(true);
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("B");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("B");
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
