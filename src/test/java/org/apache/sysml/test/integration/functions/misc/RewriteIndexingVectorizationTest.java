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

import org.junit.Assert;
import org.junit.Test;
import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;
import org.apache.sysml.utils.Statistics;

public class RewriteIndexingVectorizationTest extends AutomatedTestBase 
{
	private static final String TEST_NAME1 = "RewriteIndexingVectorizationRow"; 
	private static final String TEST_NAME2 = "RewriteIndexingVectorizationCol";

	private static final String TEST_DIR = "functions/misc/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RewriteIndexingVectorizationTest.class.getSimpleName() + "/";
	
	private static final int dim1 = 711;
	private static final int dim2 = 7;
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "R" }) );
	}
	
	@Test
	public void testIndexingVectorizationRowNoRewrite() {
		testRewriteIndexingVectorization(TEST_NAME1, false);
	}
	
	@Test
	public void testIndexingVectorizationColNoRewrite() {
		testRewriteIndexingVectorization(TEST_NAME2, false);
	}
	
	@Test
	public void testIndexingVectorizationRow() {
		testRewriteIndexingVectorization(TEST_NAME1, true);
	}
	
	@Test
	public void testIndexingVectorizationCol() {
		testRewriteIndexingVectorization(TEST_NAME2, true);
	}
	
	
	private void testRewriteIndexingVectorization(String testname, boolean vectorize)
	{
		boolean oldFlag = OptimizerUtils.ALLOW_AUTO_VECTORIZATION;
		
		try
		{
			TestConfiguration config = getTestConfiguration(testname);
			loadTestConfiguration(config);
			
			int rows = testname.equals(TEST_NAME1) ? dim2 : dim1;
			int cols = testname.equals(TEST_NAME1) ? dim1 : dim2;
			
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{ "-stats","-args", String.valueOf(rows),
				String.valueOf(cols), output("R") };
			
			OptimizerUtils.ALLOW_AUTO_VECTORIZATION = vectorize;

			runTest(true, false, null, -1);
			
			//compare output 
			double ret = readDMLMatrixFromHDFS("R").get(new CellIndex(1,1));
			Assert.assertTrue(ret == (711*5));
			
			//check for applied rewrite
			int expected = vectorize ? 1 : 5;
			Assert.assertTrue(Statistics.getCPHeavyHitterCount("rightIndex")==expected+1);
			Assert.assertTrue(Statistics.getCPHeavyHitterCount("leftIndex")==expected);
		}
		finally {
			OptimizerUtils.ALLOW_AUTO_VECTORIZATION = oldFlag;
		}
	}
}
