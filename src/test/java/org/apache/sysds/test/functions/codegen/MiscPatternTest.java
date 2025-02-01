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

public class MiscPatternTest extends AutomatedTestBase 
{
	
	private static final Log LOG = LogFactory.getLog(MiscPatternTest.class.getName());

	private static final String TEST_NAME = "miscPattern";
	private static final String TEST_NAME1 = TEST_NAME+"1"; //Y + (X * U%*%t(V)) overlapping cell-outer
	private static final String TEST_NAME2 = TEST_NAME+"2"; //multi-agg w/ large common subexpression 
	private static final String TEST_NAME3 = TEST_NAME+"3"; //sum((X!=0) * (U %*% t(V) - X)^2) 
	private static final String TEST_NAME4 = TEST_NAME+"4"; //((X!=0) * (U %*% t(V) - X)) %*% V + Y overlapping row-outer
	
	private static final String TEST_DIR = "functions/codegen/";
	private static final String TEST_CLASS_DIR = TEST_DIR + MiscPatternTest.class.getSimpleName() + "/";
	private final static String TEST_CONF = "SystemDS-config-codegen.xml";
	private final static File   TEST_CONF_FILE = new File(SCRIPT_DIR + TEST_DIR, TEST_CONF);

	private static final double eps = Math.pow(10, -10);
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		for(int i=1; i<=4; i++)
			addTestConfiguration( TEST_NAME+i, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME+i, new String[] { String.valueOf(i) }) );
	}
	
	@Test
	public void testCodegenMiscRewrite1CP() {
		testCodegenIntegration( TEST_NAME1, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenMisc1CP() {
		testCodegenIntegration( TEST_NAME1, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenMisc1SP() {
		testCodegenIntegration( TEST_NAME1, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenMiscRewrite2CP() {
		testCodegenIntegration( TEST_NAME2, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenMisc2CP() {
		testCodegenIntegration( TEST_NAME2, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenMisc2SP() {
		testCodegenIntegration( TEST_NAME2, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenMiscRewrite3CP() {
		testCodegenIntegration( TEST_NAME3, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenMisc3CP() {
		testCodegenIntegration( TEST_NAME3, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenMisc3SP() {
		testCodegenIntegration( TEST_NAME3, false, ExecType.SPARK );
	}
	
	@Test
	public void testCodegenMiscRewrite4CP() {
		testCodegenIntegration( TEST_NAME4, true, ExecType.CP );
	}
	
	@Test
	public void testCodegenMisc4CP() {
		testCodegenIntegration( TEST_NAME4, false, ExecType.CP );
	}
	
	@Test
	public void testCodegenMisc4SP() {
		testCodegenIntegration( TEST_NAME4, false, ExecType.SPARK );
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
			programArgs = new String[]{"recompile_runtime", "-stats", "-args", output("S") };
			
			fullRScriptName = HOME + testname + ".R";
			rCmd = getRCmd(expectedDir());

			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
			
			runTest(true, false, null, -1); 
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("S");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromExpectedDir("S");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			Assert.assertTrue(heavyHittersContainsSubString(Opcodes.SPOOF.toString())
					|| heavyHittersContainsSubString("sp_spoof"));
			
			//ensure correct optimizer decisions
			if( testname.equals(TEST_NAME1) )
				Assert.assertTrue(!heavyHittersContainsSubString("spoofCell")
					&& !heavyHittersContainsSubString("sp_spoofCell"));
			else if( testname.equals(TEST_NAME2) )
				Assert.assertTrue(!heavyHittersContainsSubString(Opcodes.SPOOF.toString(), 2)
					&& !heavyHittersContainsSubString("sp_spoof", 2));
			else if( testname.equals(TEST_NAME3) || testname.equals(TEST_NAME4) )
				Assert.assertTrue(heavyHittersContainsSubString("spoofOP", "sp+spoofOP")
					&& !heavyHittersContainsSubString(Opcodes.MMULT.toString()));
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
