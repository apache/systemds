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

import org.apache.sysds.common.Opcodes;
import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;

public class RewriteFoldMinMaxTest extends AutomatedTestBase 
{
	private static final String TEST_NAME1 = "RewriteFoldMin";
	private static final String TEST_NAME2 = "RewriteFoldMax";
	
	private static final String TEST_DIR = "functions/rewrite/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RewriteFoldMinMaxTest.class.getSimpleName() + "/";
	
	private static final int rows = 1932;
	private static final int cols = 14;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "R" }) );
	}

	@Test
	public void testRewriteFoldMinNoRewrite() {
		testRewriteFoldMinMax( TEST_NAME1, false, ExecType.CP );
	}
	
	@Test
	public void testRewriteFoldMinRewrite() {
		testRewriteFoldMinMax( TEST_NAME1, true, ExecType.CP );
	}
	
	@Test
	public void testRewriteFoldMaxNoRewrite() {
		testRewriteFoldMinMax( TEST_NAME2, false, ExecType.CP );
	}
	
	@Test
	public void testRewriteFoldMaxRewrite() {
		testRewriteFoldMinMax( TEST_NAME2, true, ExecType.CP );
	}

	private void testRewriteFoldMinMax( String testname, boolean rewrites, ExecType et )
	{
		ExecMode platformOld = setExecMode(et);
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		
		try {
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{ "-stats", "-args", String.valueOf(rows), 
					String.valueOf(cols), output("R") };
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;

			//run performance tests
			runTest(true, false, null, -1); 
			
			//compare matrices 
			Double ret = readDMLMatrixFromOutputDir("R").get(new CellIndex(1,1));
			Assert.assertEquals("Wrong result", Double.valueOf(5*rows*cols), ret);
			
			//check for applied rewrites
			if( rewrites ) {
				Assert.assertTrue(!heavyHittersContainsString(Opcodes.MIN.toString()) && !heavyHittersContainsString(Opcodes.MAX.toString())
					&& (!testname.equals(TEST_NAME1) || Statistics.getCPHeavyHitterCount(Opcodes.NMIN.toString()) == 1)
					&& (!testname.equals(TEST_NAME2) || Statistics.getCPHeavyHitterCount(Opcodes.NMAX.toString()) == 1));
			}
		}
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
			resetExecMode(platformOld);
		}
	}
}

