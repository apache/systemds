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

package org.apache.sysds.test.functions.misc;

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.common.Types;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.lops.LopProperties.ExecType;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;

public class ListAppendRemove extends AutomatedTestBase 
{
	private static final String TEST_NAME1 = "ListAppendRemove";
	
	private static final String TEST_DIR = "functions/misc/";
	private static final String TEST_CLASS_DIR = TEST_DIR + ListAppendRemove.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "R" }) );
	}
	
	@Test
	public void testListAppendRemoveRewriteCP() {
		runListAppendRemove(TEST_NAME1, ExecType.CP, true, false);
	}
	
	@Test
	public void testListAppendRemoveRewriteSP() {
		runListAppendRemove(TEST_NAME1, ExecType.SPARK, true, false);
	}
	
	@Test
	public void testListAppendRemoveNoRewriteCP() {
		runListAppendRemove(TEST_NAME1, ExecType.CP, false, false);
	}
	
	@Test
	public void testListAppendRemoveNoRewriteSP() {
		runListAppendRemove(TEST_NAME1, ExecType.SPARK, false, false);
	}
	
	@Test
	public void testListAppendRemoveRewriteCondCP() {
		runListAppendRemove(TEST_NAME1, ExecType.CP, true, true);
	}
	
	@Test
	public void testListAppendRemoveRewriteCondSP() {
		runListAppendRemove(TEST_NAME1, ExecType.SPARK, true, true);
	}
	
	@Test
	public void testListAppendRemoveNoRewriteCondCP() {
		runListAppendRemove(TEST_NAME1, ExecType.CP, false, true);
	}
	
	@Test
	public void testListAppendRemoveNoRewriteCondSP() {
		runListAppendRemove(TEST_NAME1, ExecType.SPARK, false, true);
	}
	
	
	private void runListAppendRemove(String testname, ExecType type, boolean rewrites, boolean conditional)
	{
		Types.ExecMode platformOld = setExecMode(type);
		boolean rewriteOld = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		
		try {
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = true;
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{ "-stats","-explain","-args",
				String.valueOf(conditional).toUpperCase(), output("R") };
			fullRScriptName = HOME + testname + ".R";
			rCmd = getRCmd(expectedDir());
			
			//run test
			runTest(true, false, null, -1);
			
			//compare matrices 
			double[][] ret = TestUtils.convertHashMapToDoubleArray(
				readDMLMatrixFromHDFS("R"), 4, 1);
			Assert.assertEquals(new Double(ret[0][0]), new Double(0)); //empty list
			Assert.assertEquals(new Double(ret[1][0]), new Double(7)); //append list
			//Assert.assertEquals(new Double(ret[2][0]), new Double(3)); //remove list
			
			//check for properly compiled CP operations for list 
			//(but spark instructions for sum, indexing, write)
			int numExpected = (type == ExecType.CP) ? 0 :
				conditional ? 5 : 4;
			Assert.assertTrue(Statistics.getNoOfExecutedSPInst()==numExpected);
			Assert.assertTrue(Statistics.getNoOfExecutedSPInst()==numExpected);
		}
		finally {
			rtplatform = platformOld;
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewriteOld;
		}
	}
}
