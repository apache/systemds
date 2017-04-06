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

import org.junit.Test;

import org.junit.Assert;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

public class RewriteCTableToRExpandTest extends AutomatedTestBase 
{
	private static final String TEST_NAME1 = "RewriteCTableToRExpandLeftPos";
	private static final String TEST_NAME2 = "RewriteCTableToRExpandRightPos"; 
	private static final String TEST_NAME3 = "RewriteCTableToRExpandLeftNeg"; 
	private static final String TEST_NAME4 = "RewriteCTableToRExpandRightNeg"; 
	
	private static final String TEST_DIR = "functions/misc/";
	private static final String TEST_CLASS_DIR = TEST_DIR + RewriteCTableToRExpandTest.class.getSimpleName() + "/";
	
	private static final int maxVal = 10;
	private static final int rows = 123;
	
	private enum CropType {
		CROP,
		PAD
	}
	
	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] { "R" }) );
		addTestConfiguration( TEST_NAME4, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4, new String[] { "R" }) );
	}

	@Test
	public void testRewriteCTableRExpandLeftPositiveDenseCrop()  {
		testRewriteCTableRExpand( TEST_NAME1, CropType.CROP );
	}
	
	@Test
	public void testRewriteCTableRExpandLeftPositiveDensePad()  {
		testRewriteCTableRExpand( TEST_NAME1, CropType.PAD );
	}
	
	@Test
	public void testRewriteCTableRExpandRightPositiveDenseCrop()  {
		testRewriteCTableRExpand( TEST_NAME2, CropType.CROP );
	}
	
	@Test
	public void testRewriteCTableRExpandRightPositiveDensePad()  {
		testRewriteCTableRExpand( TEST_NAME2, CropType.PAD );
	}
	
	@Test
	public void testRewriteCTableRExpandLeftNegativeDenseCrop()  {
		testRewriteCTableRExpand( TEST_NAME3, CropType.CROP );
	}
	
	@Test
	public void testRewriteCTableRExpandLeftNegativeDensePad()  {
		testRewriteCTableRExpand( TEST_NAME3, CropType.PAD );
	}
	
	@Test
	public void testRewriteCTableRExpandRightNegativeDenseCrop()  {
		testRewriteCTableRExpand( TEST_NAME4, CropType.CROP );
	}
	
	@Test
	public void testRewriteCTableRExpandRightNegativeDensePad()  {
		testRewriteCTableRExpand( TEST_NAME4, CropType.PAD );
	}
	
	
	private void testRewriteCTableRExpand( String testname, CropType type )
	{	
		TestConfiguration config = getTestConfiguration(testname);
		loadTestConfiguration(config);

		int outDim = maxVal + ((type==CropType.CROP) ? -7 : 7);
		
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + testname + ".dml";
		programArgs = new String[]{ "-stats","-args", 
			input("A"), String.valueOf(outDim), output("R") };
		
		fullRScriptName = HOME + testname + ".R";
		rCmd = getRCmd(inputDir(), String.valueOf(outDim), expectedDir());			

		double[][] A = getRandomMatrix(rows, 1, 1, 10, 1.0, 7);
		writeInputMatrixWithMTD("A", A, false);
		
		//run performance tests
		runTest(true, false, null, -1); 
		
		//compare output meta data
		boolean left = (testname.equals(TEST_NAME1) || testname.equals(TEST_NAME3));
		boolean pos = (testname.equals(TEST_NAME1) || testname.equals(TEST_NAME2));
		int rrows = (left && pos) ? rows : outDim;
		int rcols = (!left && pos) ? rows : outDim;
		checkDMLMetaDataFile("R", new MatrixCharacteristics(rrows, rcols, 1, 1));
		
		//check for applied rewrite
		Assert.assertEquals(new Boolean(testname.equals(TEST_NAME1)||testname.equals(TEST_NAME2)), 
				new Boolean(heavyHittersContainsSubString("rexpand")));
	}
}
