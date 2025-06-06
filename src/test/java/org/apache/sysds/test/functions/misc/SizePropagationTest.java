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

import org.junit.Test;
import org.apache.sysds.api.DMLScript;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;
import org.junit.Assert;

import java.util.HashMap;

public class SizePropagationTest extends AutomatedTestBase 
{
	private static final String TEST_NAME1 = "SizePropagationRBind";
	private static final String TEST_NAME2 = "SizePropagationLoopIx1";
	private static final String TEST_NAME3 = "SizePropagationLoopIx2";
	private static final String TEST_NAME4 = "SizePropagationLoopIx3";
	private static final String TEST_NAME5 = "SizePropagationLoopIx4";
	private static final String TEST_NAME6 = "SizePropagationUnique";
	
	private static final String TEST_DIR = "functions/misc/";
	private static final String TEST_CLASS_DIR = TEST_DIR + SizePropagationTest.class.getSimpleName() + "/";
	
	private static final int N = 100;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME4, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME5, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME5, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME6, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME6, new String[] { "R" }) );
	}

	@Test
	public void testSizePropagationRBindNoRewrites() {
		testSizePropagation( TEST_NAME1, false, 2*(N+2) );
	}
	
	@Test
	public void testSizePropagationRBindRewrites() {
		testSizePropagation( TEST_NAME1, true, 2*(N+2) );
	}
	
	@Test
	public void testSizePropagationLoopIx1NoRewrites() {
		testSizePropagation( TEST_NAME2, false, N );
	}
	
	@Test
	public void testSizePropagationLoopIx1Rewrites() {
		testSizePropagation( TEST_NAME2, true, N );
	}
	
	@Test
	public void testSizePropagationLoopIx2NoRewrites() {
		testSizePropagation( TEST_NAME3, false, N-2 );
	}
	
	@Test
	public void testSizePropagationLoopIx2Rewrites() {
		testSizePropagation( TEST_NAME3, true, N-2 );
	}
	
	@Test
	public void testSizePropagationLoopIx3NoRewrites() {
		testSizePropagation( TEST_NAME4, false, N-1 );
	}
	
	@Test
	public void testSizePropagationLoopIx3Rewrites() {
		testSizePropagation( TEST_NAME4, true, N-1 );
	}
	
	@Test
	public void testSizePropagationLoopIx4NoRewrites() {
		testSizePropagation( TEST_NAME5, false, N );
	}
	
	@Test
	public void testSizePropagationLoopIx4Rewrites() {
		testSizePropagation( TEST_NAME5, true, N );
	}
	
	@Test
	public void testSizePropagationUnique1() {
		testSizePropagation( TEST_NAME6, false, 10 );
	}
	
	@Test
	public void testSizePropagationUnique2() {
		testSizePropagation( TEST_NAME6, false, 10 );
	}
	
	private void testSizePropagation( String testname, boolean rewrites, int expect ) {
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		ExecMode oldPlatform = rtplatform;
		
		try {
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{ "hops", "-stats","-args", String.valueOf(N), output("R") };
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
			rtplatform = ExecMode.HYBRID;
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
			
			runTest(true, false, null, -1); 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromOutputDir("R");
			Assert.assertEquals(Double.valueOf(expect), dmlfile.get(new CellIndex(1,1)));
			if( testname.equals(TEST_NAME6) )
				Assert.assertEquals(0, Statistics.getNoOfCompiledSPInst());
		}
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
			DMLScript.USE_LOCAL_SPARK_CONFIG = false;
			rtplatform = oldPlatform;
		}
	}
}
