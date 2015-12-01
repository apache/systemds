/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.test.integration.functions.recompile;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

/**
 * The main purpose of this test is to ensure that encountered and fixed
 * issues, related to remove empty rewrites (or issues, which showed up due
 * to those rewrites) will never happen again.
 * 
 */
public class RemoveEmptyPotpourriTest extends AutomatedTestBase 
{
	
	private final static String TEST_NAME1 = "remove_empty_potpourri1";
	private final static String TEST_NAME2 = "remove_empty_potpourri2";
	private final static String TEST_NAME3 = "remove_empty_potpourri3";
	private final static String TEST_NAME4 = "remove_empty_potpourri4";
	
	private final static String TEST_DIR = "functions/recompile/";
	private final static String TEST_CLASS_DIR = TEST_DIR + RemoveEmptyPotpourriTest.class.getSimpleName() + "/";
	private final static double eps = 1e-10;
	
	
	@Override
	public void setUp() 
	{
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "R" }));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "R" }));
		addTestConfiguration(TEST_NAME3, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] { "R" }));
		addTestConfiguration(TEST_NAME4, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4, new String[] { "R" }));
	}
	
	@Test
	public void testRemoveEmptySequenceReshapeNoRewrite() {
		runRemoveEmptyTest(TEST_NAME1, false);
	}
	
	@Test
	public void testRemoveEmptySequenceReshapeRewrite() {
		runRemoveEmptyTest(TEST_NAME1, true);
	}
	
	@Test
	public void testRemoveEmptySumColSumNoRewrite()  {
		runRemoveEmptyTest(TEST_NAME2, false);
	}
	
	@Test
	public void testRemoveEmptySumColSumRewrite() {
		runRemoveEmptyTest(TEST_NAME2, true);
	}
	
	@Test
	public void testRemoveEmptyComplexDagSplitNoRewrite() {
		runRemoveEmptyTest(TEST_NAME3, false);
	}
	
	@Test
	public void testRemoveEmptyComplexDagSplitRewrite() {
		runRemoveEmptyTest(TEST_NAME3, true);
	}
	
	@Test
	public void testRemoveEmptyComplexDagSplit2NoRewrite() {
		runRemoveEmptyTest(TEST_NAME4, false);
	}
	
	@Test
	public void testRemoveEmptyComplexDagSplit2Rewrite() {
		runRemoveEmptyTest(TEST_NAME4, true);
	}

	/**
	 * 
	 * @param type
	 * @param empty
	 */
	private void runRemoveEmptyTest( String TEST_NAME, boolean rewrite )
	{	
		getAndLoadTestConfiguration(TEST_NAME);
		boolean oldFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		
		try
		{
			// This is for running the junit test the new way, i.e., construct the arguments directly
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			//note: stats required for runtime check of rewrite
			programArgs = new String[]{"-explain","-args", output("R") };
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + expectedDir();
	
			runTest(true, false, null, -1); 
			runRScript(true);
					
			//compare matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("R");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "DML", "R");	
		}
		finally
		{
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlag;
		}
	}
	

}