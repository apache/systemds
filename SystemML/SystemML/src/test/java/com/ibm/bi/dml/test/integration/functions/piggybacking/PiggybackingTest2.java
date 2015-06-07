/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.piggybacking;

import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

public class PiggybackingTest2 extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME = "Piggybacking_iqm";
	private final static String TEST_DIR = "functions/piggybacking/";

	@Override
	public void setUp() 
	{
		addTestConfiguration(
				TEST_NAME, 
				new TestConfiguration(TEST_DIR, TEST_NAME, 
				new String[] { "x", "iqm.scalar" })   ); 
	}
	
	/**
	 * Tests for a piggybacking bug.
	 * 
	 * Specific issue is that the combineunary lop gets piggybacked
	 * into GMR while it should only be piggybacked into SortMR job.
	 */
	@Test
	public void testPiggybacking_iqm()
	{		

		RUNTIME_PLATFORM rtold = rtplatform;
		
		// bug can be reproduced only when exec mode is HADOOP 
		rtplatform = RUNTIME_PLATFORM.HADOOP;
		
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[]{"-args", HOME + OUTPUT_DIR + config.getOutputFiles()[0] };

		loadTestConfiguration(config);
		
		boolean exceptionExpected = false;
		runTest(true, exceptionExpected, null, -1);
	
		HashMap<CellIndex, Double> d = TestUtils.readDMLScalarFromHDFS(HOME + OUTPUT_DIR + config.getOutputFiles()[0]);
		
		Assert.assertEquals(d.get(new CellIndex(1,1)), Double.valueOf(1.0), 1e-10);
		
		rtplatform = rtold;
	}

}