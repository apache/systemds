/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.recompile;

import java.util.HashMap;

import org.junit.Test;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue.CellIndex;
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
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME1 = "remove_empty_potpourri1";
	
	private final static String TEST_DIR = "functions/recompile/";
	private final static double eps = 1e-10;
	
	
	@Override
	public void setUp() 
	{
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_DIR, TEST_NAME1, new String[] { "R" }));
	}
	
	@Test
	public void testRemoveEmptySequenceReshape() 
	{
		runRemoveEmptyTest(TEST_NAME1);
	}

	/**
	 * 
	 * @param type
	 * @param empty
	 */
	private void runRemoveEmptyTest( String TEST_NAME )
	{	
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		
		// This is for running the junit test the new way, i.e., construct the arguments directly
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		//note: stats required for runtime check of rewrite
		programArgs = new String[]{"-args", HOME + OUTPUT_DIR + "R" };
		fullRScriptName = HOME + TEST_NAME + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + HOME + EXPECTED_DIR;
		
		loadTestConfiguration(config);

		runTest(true, false, null, -1); 
		runRScript(true);
				
		//compare matrices
		HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
		HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("R");
		TestUtils.compareMatrices(dmlfile, rfile, eps, "DML", "R");	
	}
	

}