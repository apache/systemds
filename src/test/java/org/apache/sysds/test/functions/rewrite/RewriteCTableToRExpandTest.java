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
import org.junit.Test;
import org.apache.sysds.common.Types.ExecMode;
import org.apache.sysds.runtime.meta.MatrixCharacteristics;
import org.apache.sysds.test.AutomatedTestBase;
import org.apache.sysds.test.TestConfiguration;
import org.apache.sysds.test.TestUtils;
import org.junit.Assert;

public class RewriteCTableToRExpandTest extends AutomatedTestBase 
{
	private static final String[] TEST_NAMES = new String[] {
		"RewriteCTableToRExpandLeftPos",
		"RewriteCTableToRExpandRightPos",
		"RewriteCTableToRExpandLeftNeg",
		"RewriteCTableToRExpandRightNeg",
		"RewriteCTableToRExpandLeftUnknownPos",
		"RewriteCTableToRExpandRightUnknownPos",
		"RewriteCTableToRExpandRightVarMax"
	};
	
	private static final String TEST_DIR = "functions/rewrite/";
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
		for(int i=0; i<TEST_NAMES.length; i++)
			addTestConfiguration( TEST_NAMES[i],
				new TestConfiguration(TEST_CLASS_DIR, TEST_NAMES[i], new String[] { "R" }) );
	}

	@Test
	public void testRewriteCTableRExpandLeftPositiveDenseCrop()  {
		testRewriteCTableRExpand( 1, CropType.CROP );
	}
	
	@Test
	public void testRewriteCTableRExpandLeftPositiveDensePad()  {
		testRewriteCTableRExpand( 1, CropType.PAD );
	}
	
	@Test
	public void testRewriteCTableRExpandRightPositiveDenseCrop()  {
		testRewriteCTableRExpand( 2, CropType.CROP );
	}
	
	@Test
	public void testRewriteCTableRExpandRightPositiveDensePad()  {
		testRewriteCTableRExpand( 2, CropType.PAD );
	}
	
	@Test
	public void testRewriteCTableRExpandLeftNegativeDenseCrop()  {
		testRewriteCTableRExpand( 3, CropType.CROP );
	}
	
	@Test
	public void testRewriteCTableRExpandLeftNegativeDensePad()  {
		testRewriteCTableRExpand( 3, CropType.PAD );
	}
	
	@Test
	public void testRewriteCTableRExpandRightNegativeDenseCrop()  {
		testRewriteCTableRExpand( 4, CropType.CROP );
	}
	
	@Test
	public void testRewriteCTableRExpandRightNegativeDensePad()  {
		testRewriteCTableRExpand( 4, CropType.PAD );
	}
	
	@Test
	public void testRewriteCTableRExpandLeftUnknownDenseCrop()  {
		testRewriteCTableRExpand( 5, CropType.CROP );
	}
	
	@Test
	public void testRewriteCTableRExpandLeftUnknownDensePad()  {
		testRewriteCTableRExpand( 5, CropType.PAD );
	}
	
	@Test
	public void testRewriteCTableRExpandRightUnknownDenseCrop()  {
		testRewriteCTableRExpand( 6, CropType.CROP );
	}
	
	@Test
	public void testRewriteCTableRExpandRightUnknownDensePad()  {
		testRewriteCTableRExpand( 6, CropType.PAD );
	}
	
	@Test
	public void testRewriteCTableRExpandVarMaxCropCP()  {
		testRewriteCTableRExpand( 7, CropType.CROP, ExecMode.HYBRID );
	}
	
	@Test
	public void testRewriteCTableRExpandVarMaxPadCP()  {
		testRewriteCTableRExpand( 7, CropType.PAD, ExecMode.HYBRID );
	}
	
	@Test
	public void testRewriteCTableRExpandVarMaxCropSP()  {
		testRewriteCTableRExpand( 7, CropType.CROP, ExecMode.SPARK );
	}
	
	@Test
	public void testRewriteCTableRExpandVarMaxPadSP()  {
		testRewriteCTableRExpand( 7, CropType.PAD, ExecMode.SPARK );
	}
	
	private void testRewriteCTableRExpand( int test, CropType type ) {
		testRewriteCTableRExpand(test, type, ExecMode.HYBRID);
	}
	
	private void testRewriteCTableRExpand(int test, CropType type, ExecMode mode)
	{
		String testname = TEST_NAMES[test-1];
		TestConfiguration config = getTestConfiguration(testname);
		loadTestConfiguration(config);

		int outDim = maxVal + ((type==CropType.CROP) ? -7 : 7);
		boolean unknownTests = (test >= 5);
		
		ExecMode platformOld = rtplatform;
		if( unknownTests & test != 7 )
			mode = ExecMode.SINGLE_NODE;
		setExecMode(mode);
		
		try 
		{
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{ "-explain","-stats","-args", 
				input("A"), String.valueOf(outDim), output("R") };
			
			fullRScriptName = HOME + testname + ".R";
			rCmd = getRCmd(inputDir(), String.valueOf(outDim), expectedDir());
	
			double[][] A = getRandomMatrix(rows, 1, 1, 10, 1.0, 7);
			writeInputMatrixWithMTD("A", A, false);
			
			//run performance tests
			runTest(true, false, null, -1); 
			
			//compare output meta data
			boolean left = (test == 1 || test == 3 || test == 5 || test == 6 || test == 7);
			boolean pos = (test == 1 || test == 2);
			int rrows = (left && pos) ? rows : outDim;
			int rcols = (!left && pos) ? rows : outDim;
			if( !unknownTests )
				checkDMLMetaDataFile("R", new MatrixCharacteristics(rrows, rcols, 1, 1));
			
			//check for applied rewrite
			Assert.assertEquals(Boolean.valueOf(test==1 || test==2 || unknownTests),
				Boolean.valueOf(heavyHittersContainsSubString(Opcodes.REXPAND.toString())));
		}
		finally {
			resetExecMode(platformOld);
		}
	}
}
