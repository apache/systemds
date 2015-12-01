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

package org.apache.sysml.test.integration.functions.parfor;

import java.util.HashMap;

import org.junit.Assert;

import org.junit.Test;

import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

public class ParForReplaceThreadIDRecompileTest extends AutomatedTestBase 
{
	
	private final static String TEST_DIR = "functions/parfor/";
	private final static String TEST_NAME1 = "parfor_threadid_recompile1"; //for
	private final static String TEST_NAME2 = "parfor_threadid_recompile2"; //parfor
	private final static String TEST_CLASS_DIR = TEST_DIR + ParForReplaceThreadIDRecompileTest.class.getSimpleName() + "/";
	
	@Override
	public void setUp() 
	{
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "B" }));
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "B" }));
	}

	@Test
	public void testThreadIDReplaceForRecompile() 
	{
		runThreadIDReplaceTest(TEST_NAME1, true);
	}
	
	@Test
	public void testThreadIDReplaceParForRecompile() 
	{
		runThreadIDReplaceTest(TEST_NAME2, true);
	}
	
	
	/**
	 * 
	 * @param TEST_NAME
	 * @param recompile
	 */
	private void runThreadIDReplaceTest( String TEST_NAME, boolean recompile )
	{
		boolean flag = OptimizerUtils.ALLOW_DYN_RECOMPILATION;
		
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			loadTestConfiguration(config);
			
			OptimizerUtils.ALLOW_DYN_RECOMPILATION = recompile;
			
			// This is for running the junit test the new way, i.e., construct the arguments directly 
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", input("A"), output("B") };
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + inputDir() + " " + expectedDir();
			
	        double[][] A = new double[][]{{2.0},{3.0}};
			writeInputMatrixWithMTD("A", A, false);
	
			runTest(true, false, null, -1);
			
			//compare matrices
			HashMap<CellIndex, Double> dmlout = readDMLMatrixFromHDFS("B");
			Assert.assertTrue( dmlout.size()>=1 );
		}
		finally{
			OptimizerUtils.ALLOW_DYN_RECOMPILATION = flag;
		}
	}
	
}