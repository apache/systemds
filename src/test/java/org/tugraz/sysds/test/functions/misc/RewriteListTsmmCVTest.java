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

package org.tugraz.sysds.test.functions.misc;

import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;
import org.tugraz.sysds.common.Types.ExecMode;
import org.tugraz.sysds.hops.OptimizerUtils;
import org.tugraz.sysds.lops.LopProperties.ExecType;
import org.tugraz.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.tugraz.sysds.test.AutomatedTestBase;
import org.tugraz.sysds.test.TestConfiguration;
import org.tugraz.sysds.test.TestUtils;
import org.tugraz.sysds.utils.Statistics;

/**
 * Regression test for function recompile-once issue with literal replacement.
 * 
 */
public class RewriteListTsmmCVTest extends AutomatedTestBase 
{
	private static final String TEST_NAME1 = "RewriteListTsmmCV";
	
	private static final String TEST_DIR = "functions/misc/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RewriteListTsmmCVTest.class.getSimpleName() + "/";
	
	private static final int rows = 10001; 
	private static final int cols = 100;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "R" }) );
	}
	
	@Test
	public void testListTsmmRewriteCP() {
		testListTsmmCV(TEST_NAME1, true, false, ExecType.CP);
	}
	
	@Test
	public void testListTsmmRewriteSP() {
		testListTsmmCV(TEST_NAME1, true, false, ExecType.SPARK);
	}
	
//	@Test
//	public void testListTsmmRewriteLineageCP() {
//		testListTsmmCV(TEST_NAME1, true, true, ExecType.CP);
//	}
//	
//	@Test
//	public void testListTsmmRewriteLineageSP() {
//		testListTsmmCV(TEST_NAME1, true, true, ExecType.SPARK);
//	}
	
	private void testListTsmmCV( String testname, boolean rewrites, boolean lineage, ExecType instType )
	{
		ExecMode platformOld = setExecMode(instType);
		
		boolean rewritesOld = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
		
		try
		{
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";

			//TODO lineage list
//			ReuseCacheType ltype = lineage ? ReuseCacheType.REUSE_FULL : ReuseCacheType.NONE; 
//			programArgs = new String[]{"-explain", "-lineage", ltype.name(), "-stats","-args",
//				String.valueOf(rows), String.valueOf(cols), output("S") };

			programArgs = new String[]{"-explain", "-stats","-args",
				String.valueOf(rows), String.valueOf(cols), output("S") };
			
			fullRScriptName = HOME + testname + ".R";
			rCmd = getRCmd(inputDir(), expectedDir());
			
			runTest(true, false, null, -1);
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("S");
			Assert.assertEquals(new Double(cols*7), dmlfile.get(new CellIndex(1,1)));
			
			//check compiled instructions after rewrite
			if( instType == ExecType.CP )
				Assert.assertEquals(0, Statistics.getNoOfExecutedSPInst());
			if( rewrites ) {
				String[] codes = (instType==ExecType.CP) ?
					new String[]{"rbind","tsmm","ba+*","+"} :
					new String[]{"sp_append","sp_tsmm","sp_mapmm","sp_+"};
				Assert.assertTrue(!heavyHittersContainsString(codes[0]));
				Assert.assertEquals( (lineage ? 7 : 7*6), //per fold
					Statistics.getCPHeavyHitterCount(codes[1]));
				Assert.assertEquals( (lineage ? 7 : 7*6) + 1, //per fold
					Statistics.getCPHeavyHitterCount(codes[2]));
				Assert.assertEquals( 7*5 + (instType==ExecType.CP?7*6:0), //for intermediates
					Statistics.getCPHeavyHitterCount(codes[3]));
			}
		}
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewritesOld;
			rtplatform = platformOld;
		}
	}
}
