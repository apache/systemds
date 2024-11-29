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

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysds.hops.OptimizerUtils;
import org.apache.sysds.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.apache.sysds.utils.Statistics;

public class RewriteLoopVectorization extends AutomatedTestBase 
{
	private static final String TEST_NAME1 = "RewriteLoopVectorizationSum"; //amendable
	private static final String TEST_NAME2 = "RewriteLoopVectorizationSum2"; //not amendable
	private static final String TEST_NAME3 = "RewriteLoopVectorizationBinary"; //amendable
	private static final String TEST_NAME4 = "RewriteLoopVectorizationUnary"; //amendable
	private static final String TEST_NAME5 = "RewriteLoopVectorizationIndexedCopy"; //amendable
	
	private static final String TEST_DIR = "functions/rewrite/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RewriteLoopVectorization.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME4, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME5, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME5, new String[] { "R" }) );
	}
	
	@Test
	public void testLoopVectorizationSumNoRewrite() {
		testRewriteLoopVectorizationSum( TEST_NAME1, false );
	}
	
	@Test
	public void testLoopVectorizationSumRewrite() {
		testRewriteLoopVectorizationSum( TEST_NAME1, true );
	}
	
	@Test
	public void testLoopVectorizationSum2NoRewrite() {
		testRewriteLoopVectorizationSum( TEST_NAME2, false );
	}
	
	@Test
	public void testLoopVectorizationSum2Rewrite() {
		testRewriteLoopVectorizationSum( TEST_NAME2, true );
	}
	
	@Test
	public void testLoopVectorizationBinaryNoRewrite() {
		testRewriteLoopVectorizationSum( TEST_NAME3, false );
	}
	
	@Test
	public void testLoopVectorizationBinaryRewrite() {
		testRewriteLoopVectorizationSum( TEST_NAME3, true );
	}
	
	@Test
	public void testLoopVectorizationUnaryNoRewrite() {
		testRewriteLoopVectorizationSum( TEST_NAME4, false );
	}
	
	@Test
	public void testLoopVectorizationUnaryRewrite() {
		testRewriteLoopVectorizationSum( TEST_NAME4, true );
	}
	
	@Test
	public void testLoopVectorizationIndexedCopyNoRewrite() {
		testRewriteLoopVectorizationSum( TEST_NAME5, false );
	}
	
	@Test
	public void testLoopVectorizationIndexedCopyRewrite() {
		testRewriteLoopVectorizationSum( TEST_NAME5, true );
	}
	
	private void testRewriteLoopVectorizationSum( String testname, boolean rewrites )
	{	
		boolean oldFlag = OptimizerUtils.ALLOW_AUTO_VECTORIZATION;
		
		try
		{
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{ "-stats", "-args", output("Scalar") };
			
			fullRScriptName = HOME + testname + ".R";
			rCmd = getRCmd(inputDir(), expectedDir());

			OptimizerUtils.ALLOW_AUTO_VECTORIZATION = rewrites;

			runTest(true, false, null, -1); 
			runRScript(true); 
			
			//compare scalars 
			HashMap<CellIndex, Double> dmlfile = readDMLScalarFromOutputDir("Scalar");
			HashMap<CellIndex, Double> rfile  = readRScalarFromExpectedDir("Scalar");
			TestUtils.compareScalars(dmlfile.toString(), rfile.toString());
			if( !testname.equals(TEST_NAME2) && rewrites )
				Assert.assertTrue(Statistics.getCPHeavyHitterCount("rightIndex") <= 2);
		}
		finally {
			OptimizerUtils.ALLOW_AUTO_VECTORIZATION = oldFlag;
		}
	}
}
