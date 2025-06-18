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

package org.apache.sysds.test.functions.einsum;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Test;

import java.io.File;
import java.util.HashMap;

public class EinsumTest extends AutomatedTestBase
{
	private static final Log LOG = LogFactory.getLog(EinsumTest.class.getName());

	private static final String TEST_NAME_EINSUM = "einsum";
	private static final String TEST_EINSUM1 = TEST_NAME_EINSUM+"1";
	private static final String TEST_EINSUM2 = TEST_NAME_EINSUM+"2";
	private static final String TEST_EINSUM3 = TEST_NAME_EINSUM+"3";
	private static final String TEST_EINSUM4 = TEST_NAME_EINSUM+"4";
	private static final String TEST_EINSUM5 = TEST_NAME_EINSUM+"5";
	private static final String TEST_EINSUM6 = TEST_NAME_EINSUM+"6";
	private static final String TEST_EINSUM7 = TEST_NAME_EINSUM+"7";
	private static final String TEST_EINSUM8 = TEST_NAME_EINSUM+"8";
	private static final String TEST_EINSUM9 = TEST_NAME_EINSUM+"9";
	private static final String TEST_EINSUM10 = TEST_NAME_EINSUM+"10";
	private static final String TEST_EINSUM11 = TEST_NAME_EINSUM+"11";
	private static final String TEST_EINSUM12 = TEST_NAME_EINSUM+"12";

	private static final String TEST_DIR = "functions/einsum/";
	private static final String TEST_CLASS_DIR = TEST_DIR + EinsumTest.class.getSimpleName() + "/";
	private final static String TEST_CONF = "SystemDS-config-codegen.xml";
	private final static File   TEST_CONF_FILE = new File(SCRIPT_DIR + TEST_DIR, TEST_CONF);

	private static double eps = Math.pow(10, -10);
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		for(int i=1; i<=12; i++)
			addTestConfiguration( TEST_NAME_EINSUM+i, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME_EINSUM+i, new String[] { String.valueOf(i) }) );
	}
	@Test
	public void testCodegenEinsum1CP() {
		testCodegenIntegration( TEST_EINSUM1, false, ExecType.CP );
	}
	@Test
	public void testCodegenEinsum2CP() {
		testCodegenIntegration( TEST_EINSUM2, false, ExecType.CP );
	}
	@Test
	public void testCodegenEinsum3CP() {
	testCodegenIntegration( TEST_EINSUM3, false, ExecType.CP );
}
	@Test
	public void testCodegenEinsum4CP() {
		testCodegenIntegration( TEST_EINSUM4, false, ExecType.CP );
	}
	@Test
	public void testCodegenEinsum5CP() {
		testCodegenIntegration( TEST_EINSUM5, false, ExecType.CP );
	}
	@Test
	public void testCodegenEinsum6CP() {
		testCodegenIntegration( TEST_EINSUM6, false, ExecType.CP );
	}
	@Test
	public void testCodegenEinsum7CP() {
		testCodegenIntegration( TEST_EINSUM7, false, ExecType.CP );
	}
	@Test
	public void testCodegenEinsum8CP() { testCodegenIntegration( TEST_EINSUM8, false, ExecType.CP ); }
	@Test
	public void testCodegenEinsum9CP() {
		testCodegenIntegration( TEST_EINSUM9, false, ExecType.CP );
	}
	@Test
	public void testCodegenEinsum10CP() {
		testCodegenIntegration( TEST_EINSUM10, false, ExecType.CP );
	}
	@Test
	public void testCodegenEinsum11CP() {
		testCodegenIntegration( TEST_EINSUM11, false, ExecType.CP );
	}
	@Test
	public void testCodegenEinsum12CP() {
		testCodegenIntegration( TEST_EINSUM12, false, ExecType.CP );
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
			programArgs = new String[]{"-stats", "-explain", "-args", output("S") };
			
			fullRScriptName = HOME + testname + ".R";
			rCmd = getRCmd(inputDir(), expectedDir());

			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
			
			runTest(true, false, null, -1);
			runRScript(true);
			
			//compare matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("S");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("S");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
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
