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

package org.apache.sysml.test.integration.functions.data;

import org.junit.Assert;
import org.junit.Test;

import org.apache.sysml.api.DMLScript;
import org.apache.sysml.api.DMLScript.RUNTIME_PLATFORM;
import org.apache.sysml.lops.LopProperties.ExecType;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.test.utils.TestUtils;

/**
 * The major purpose of this test is to verify min/max value domains
 * of rand-generated values. It is important to test for both
 * dense and sparse because they rely on different code paths.
 * 
 */
public class RandTest5 extends AutomatedTestBase 
{
	
	private final static String TEST_NAME1 = "RandTest5";
	private final static String TEST_DIR = "functions/data/";
	private final static String TEST_CLASS_DIR = TEST_DIR + RandTest5.class.getSimpleName() + "/";
	
	private final static int rows = 1323;
	private final static int cols = 1156; 
	
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.01;

	private final static double minN = -25;
	private final static double maxN = -7;
	
	private final static double minP = 7;
	private final static double maxP = 25;
	
	public enum RandMinMaxType{
		POSITIVE_ONLY,
		NEGATIVE_ONLY,
		NEGATIVE_POSITIVE,
	}
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "C" })   );
		TestUtils.clearAssertionInformation();
	}

	
	@Test
	public void testRandValuesDensePositiveCP() 
	{
		runRandTest(false, RandMinMaxType.POSITIVE_ONLY, ExecType.CP);
	}
	
	@Test
	public void testRandValuesDenseNegativeCP() 
	{
		runRandTest(false, RandMinMaxType.NEGATIVE_ONLY, ExecType.CP);
	}
	
	@Test
	public void testRandValuesDenseNegativePositiveCP() 
	{
		runRandTest(false, RandMinMaxType.NEGATIVE_POSITIVE, ExecType.CP);
	}
	
	@Test
	public void testRandValuesSparsePositiveCP() 
	{
		runRandTest(true, RandMinMaxType.POSITIVE_ONLY, ExecType.CP);
	}
	
	@Test
	public void testRandValuesSparseNegativeCP() 
	{
		runRandTest(true, RandMinMaxType.NEGATIVE_ONLY, ExecType.CP);
	}
	
	@Test
	public void testRandValuesSparseNegativePositiveCP() 
	{
		runRandTest(true, RandMinMaxType.NEGATIVE_POSITIVE, ExecType.CP);
	}
	
	// -------------------------------------------------------------
	
	@Test
	public void testRandValuesDensePositiveSP() 
	{
		runRandTest(false, RandMinMaxType.POSITIVE_ONLY, ExecType.SPARK);
	}
	
	@Test
	public void testRandValuesDenseNegativeSP() 
	{
		runRandTest(false, RandMinMaxType.NEGATIVE_ONLY, ExecType.SPARK);
	}
	
	@Test
	public void testRandValuesDenseNegativePositiveSP() 
	{
		runRandTest(false, RandMinMaxType.NEGATIVE_POSITIVE, ExecType.SPARK);
	}
	
	@Test
	public void testRandValuesSparsePositiveSP() 
	{
		runRandTest(true, RandMinMaxType.POSITIVE_ONLY, ExecType.SPARK);
	}
	
	@Test
	public void testRandValuesSparseNegativeSP() 
	{
		runRandTest(true, RandMinMaxType.NEGATIVE_ONLY, ExecType.SPARK);
	}
	
	@Test
	public void testRandValuesSparseNegativePositiveSP() 
	{
		runRandTest(true, RandMinMaxType.NEGATIVE_POSITIVE, ExecType.SPARK);
	}
	
	// -------------------------------------------------------------
	
	@Test
	public void testRandValuesDensePositiveMR() 
	{
		runRandTest(false, RandMinMaxType.POSITIVE_ONLY, ExecType.MR);
	}
	
	@Test
	public void testRandValuesDenseNegativeMR() 
	{
		runRandTest(false, RandMinMaxType.NEGATIVE_ONLY, ExecType.MR);
	}
	
	@Test
	public void testRandValuesDenseNegativePositiveMR() 
	{
		runRandTest(false, RandMinMaxType.NEGATIVE_POSITIVE, ExecType.MR);
	}
	
	@Test
	public void testRandValuesSparsePositiveMR() 
	{
		runRandTest(true, RandMinMaxType.POSITIVE_ONLY, ExecType.MR);
	}
	
	@Test
	public void testRandValuesSparseNegativeMR() 
	{
		runRandTest(true, RandMinMaxType.NEGATIVE_ONLY, ExecType.MR);
	}
	
	@Test
	public void testRandValuesSparseNegativePositiveMR() 
	{
		runRandTest(true, RandMinMaxType.NEGATIVE_POSITIVE, ExecType.MR);
	}
	
	
	/**
	 * 
	 * @param sparse
	 * @param et
	 */
	private void runRandTest( boolean sparse, RandMinMaxType type, ExecType et )
	{	
		//keep old runtime 
		RUNTIME_PLATFORM platformOld = rtplatform;
		switch( et ){
			case MR: rtplatform = RUNTIME_PLATFORM.HADOOP; break;
			case SPARK: rtplatform = RUNTIME_PLATFORM.SPARK; break;
			default: rtplatform = RUNTIME_PLATFORM.HYBRID; break;
		}
		
		boolean sparkConfigOld = DMLScript.USE_LOCAL_SPARK_CONFIG;
		if( rtplatform == RUNTIME_PLATFORM.SPARK )
			DMLScript.USE_LOCAL_SPARK_CONFIG = true;
		
		//set basic parameters
		String TEST_NAME = TEST_NAME1;		 
		double sparsity = (sparse) ? sparsity2 : sparsity1;	
		double min = -1, max = -1;
		switch( type ){
			case POSITIVE_ONLY:     min = minP; max = maxP; break;
			case NEGATIVE_ONLY:     min = minN; max = maxN; break;
			case NEGATIVE_POSITIVE: min = minN; max = maxP; break;
		}
		
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		loadTestConfiguration(config);
		
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[]{"-explain", "-args", Integer.toString(rows), Integer.toString(cols),
			Double.toString(min), Double.toString(max), Double.toString(sparsity), output("C")};
		
		try 
		{
			//run tests
			runTest(true, false, null, -1);
		    
			//check validity results (rows, cols, min, max)
			checkResults(rows, cols, min, max);
		} 
		catch (Exception e) 
		{
			e.printStackTrace();
			Assert.fail();
		}
		finally
		{
			rtplatform = platformOld;
			DMLScript.USE_LOCAL_SPARK_CONFIG = sparkConfigOld;
		}
	}

}