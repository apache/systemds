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

package org.apache.sysds.test.functions.rewrite;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;

public class RewriteNaryPlusTest extends AutomatedTestBase 
{
	private static final String TEST_NAME1 = "RewriteNaryPlusPos";
	private static final String TEST_NAME2 = "RewriteNaryPlusNeg";
	
	private static final String TEST_DIR = "functions/rewrite/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RewriteNaryPlusTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[]{"R"}) );
		addTestConfiguration( TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[]{"R"}) );
	}

	@Test
	public void testRewritePosNoRewriteCP() {
		testRewritePushdownUagg(TEST_NAME1, false, ExecType.CP);
	}
	
	@Test
	public void testRewriteNegNoRewriteCP() {
		testRewritePushdownUagg(TEST_NAME2, false, ExecType.CP);
	}
	
	@Test
	public void testRewritePosRewriteCP() {
		testRewritePushdownUagg(TEST_NAME1, true, ExecType.CP);
	}
	
	@Test
	public void testRewriteNegRewriteCP() {
		testRewritePushdownUagg(TEST_NAME2, true, ExecType.CP);
	}
	
	@Test
	public void testRewritePosNoRewriteSP() {
		testRewritePushdownUagg(TEST_NAME1, false, ExecType.SPARK);
	}
	
	@Test
	public void testRewriteNegNoRewriteSP() {
		testRewritePushdownUagg(TEST_NAME2, false, ExecType.SPARK);
	}
	
	@Test
	public void testRewritePosRewriteSP() {
		testRewritePushdownUagg(TEST_NAME1, true, ExecType.SPARK);
	}
	
	@Test
	public void testRewriteNegRewriteSP() {
		testRewritePushdownUagg(TEST_NAME2, true, ExecType.SPARK);
	}

	private void testRewritePushdownUagg(String testname, boolean rewrites, ExecType et)
	{
		ExecMode oldMode = setExecMode(et);
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		
		try
		{
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{ "-stats","-args", output("R") };

			runTest(true, false, null, -1); 
			
			//compare output
			Double ret = readDMLMatrixFromHDFS("R").get(new CellIndex(1,1));
			Assert.assertEquals(new Double(100000), ret);
			
			//check for applied nary plus
			String prefix = et == ExecType.SPARK ? "sp_" : "";
			if( rewrites && testname.equals(TEST_NAME1) )
				Assert.assertTrue(Statistics.getCPHeavyHitterCount(prefix+"n+")==1);
			else
				Assert.assertTrue(Statistics.getCPHeavyHitterCount(prefix+"+")>=1);
		}
		finally {
			resetExecMode(oldMode);
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
		}
	}
}
