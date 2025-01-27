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

import java.util.HashMap;

import org.apache.sysds.common.Opcodes;
import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.common.Types.ExecType;
import org.apache.sysds.runtime.lineage.LineageCacheConfig.ReuseCacheType;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;

/**
 * Regression test for function recompile-once issue with literal replacement.
 * 
 */
public class RewriteListTsmmCVTest extends AutomatedTestBase 
{
	private static final String TEST_NAME1 = "RewriteListTsmmCV1";
	private static final String TEST_NAME2 = "RewriteListTsmmCV2";
	
	private static final String TEST_DIR = "functions/rewrite/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RewriteListTsmmCVTest.class.getSimpleName() + "/";
	
	private static final int rows = 10001; 
	private static final int cols = 100;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] {"R"}));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] {"R"}));
	}
	
	@Test
	public void testListTsmm1RewriteCP() {
		testListTsmmCV(TEST_NAME1, true, false, ExecType.CP);
	}
	
	@Test
	public void testListTsmm1RewriteSP() {
		testListTsmmCV(TEST_NAME1, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testListTsmm1RewriteLineageCP() {
		testListTsmmCV(TEST_NAME1, true, true, ExecType.CP);
	}
	
	@Test
	public void testListTsmm1RewriteLineageSP() {
		testListTsmmCV(TEST_NAME1, true, true, ExecType.SPARK);
	}
	
	@Test
	public void testListTsmm2RewriteCP() {
		testListTsmmCV(TEST_NAME2, true, false, ExecType.CP);
	}
	
	@Test
	public void testListTsmm2RewriteSP() {
		testListTsmmCV(TEST_NAME2, true, false, ExecType.SPARK);
	}
	
	@Test
	public void testListTsmm2RewriteLineageCP() {
		testListTsmmCV(TEST_NAME2, true, true, ExecType.CP);
	}
	
	@Test
	public void testListTsmm2RewriteLineageSP() {
		testListTsmmCV(TEST_NAME2, true, true, ExecType.SPARK);
	}
	
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
			
			//lineage tracing with and without reuse
			ReuseCacheType reuse = lineage ? ReuseCacheType.REUSE_FULL : ReuseCacheType.NONE;
			programArgs = new String[]{"-lineage", reuse.name().toLowerCase(),
				"-stats","-args", String.valueOf(rows), String.valueOf(cols), output("S") };
			
			fullRScriptName = HOME + testname + ".R";
			rCmd = getRCmd(inputDir(), expectedDir());
			
			runTest(true, false, null, -1);
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("S");
			Assert.assertEquals(Double.valueOf(cols*7), dmlfile.get(new CellIndex(1,1)));
			
			//check compiled instructions after rewrite
			if( instType == ExecType.CP )
				Assert.assertEquals(0, Statistics.getNoOfExecutedSPInst());
			if( rewrites ) {
				//boolean expectedReuse = lineage && instType == ExecType.CP;
				boolean expectedReuse = lineage;
				String[] codes = (instType==ExecType.CP) ?
					new String[]{Opcodes.RBIND.toString(),Opcodes.TSMM.toString(), Opcodes.MMULT.toString(), Opcodes.NP.toString()} :
					new String[]{"sp_append","sp_tsmm","sp_mapmm","sp_n+"};
				Assert.assertTrue(!heavyHittersContainsString(codes[0]));
				Assert.assertEquals( (expectedReuse ? 7 : 7*6), //per fold
					Statistics.getCPHeavyHitterCount(codes[1]));
				Assert.assertEquals( (expectedReuse ? 7 : 7*6) + 1, //per fold
					Statistics.getCPHeavyHitterCount(codes[2]));
				//for intermediates tsmm/ba+* + 7 diag (in spark sp_map+ vs sp_+)
				Assert.assertEquals( 7*2,
					Statistics.getCPHeavyHitterCount(codes[3]));
			}
		}
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewritesOld;
			rtplatform = platformOld;
		}
	}
}
