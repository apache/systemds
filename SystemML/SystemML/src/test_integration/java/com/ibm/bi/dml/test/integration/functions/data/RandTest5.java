/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.data;

import junit.framework.Assert;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

/**
 * The major purpose of this test is to verify min/max value domains
 * of rand-generated values. It is important to test for both
 * dense and sparse because they rely on different code paths.
 * 
 */
public class RandTest5 extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME1 = "RandTest5";
	private final static String TEST_DIR = "functions/data/";
	
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
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_DIR, TEST_NAME1, new String[] { "C" })   );
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
		rtplatform = (et==ExecType.MR) ? RUNTIME_PLATFORM.HADOOP : RUNTIME_PLATFORM.HYBRID;
		
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
		
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[]{"-args", Integer.toString(rows),
											Integer.toString(cols),
											Double.toString(min),
											Double.toString(max),
											Double.toString(sparsity),
							                HOME + OUTPUT_DIR + "C"};
		
		loadTestConfiguration(config);
		
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
		}
	}

}