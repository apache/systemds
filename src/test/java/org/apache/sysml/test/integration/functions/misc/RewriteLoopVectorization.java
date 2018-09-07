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

package org.apache.sysml.test.integration.functions.misc;

import java.util.HashMap;

import org.junit.Test;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

/**
 * Regression test for loop vectorization rewrite
 * for(i in 1:n) s = s + as.scalar(A[i,1]) -> s = s + sum(A[1:n,1])
 * 
 */
public class RewriteLoopVectorization extends AutomatedTestBase 
{
	private static final String TEST_NAME1 = "RewriteLoopVectorizationSum"; //amendable
	private static final String TEST_NAME2 = "RewriteLoopVectorizationSum2"; //not amendable

	private static final String TEST_DIR = "functions/misc/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RewriteLoopVectorization.class.getSimpleName() + "/";
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "R" }) );
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
	
	/**
	 * 
	 * @param testname
	 * @param rewrites
	 */
	private void testRewriteLoopVectorizationSum( String testname, boolean rewrites )
	{	
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		
		try
		{
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{ "-stats","-args", output("Scalar") };
			
			fullRScriptName = HOME + testname + ".R";
			rCmd = getRCmd(inputDir(), expectedDir());			

			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;

			runTest(true, false, null, -1); 
			runRScript(true); 
			
			//compare scalars 
			HashMap<CellIndex, Double> dmlfile = readDMLScalarFromHDFS("Scalar");
			HashMap<CellIndex, Double> rfile  = readRScalarFromFS("Scalar");
			TestUtils.compareScalars(dmlfile.toString(), rfile.toString());
		}
		finally {
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
		}	
	}	
}