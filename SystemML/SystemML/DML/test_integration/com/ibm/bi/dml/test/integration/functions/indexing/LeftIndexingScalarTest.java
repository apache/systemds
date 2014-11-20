/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.indexing;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

public class LeftIndexingScalarTest extends AutomatedTestBase
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_DIR = "functions/indexing/";
	private final static String TEST_NAME = "LeftIndexingScalarTest";

	
	private final static double epsilon=0.0000000001;
	private final static int rows = 1279;
	private final static int cols = 1050;

	private final static double sparsity = 0.7;
	private final static int min = 0;
	private final static int max = 100;
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_DIR, TEST_NAME, new String[] {"A"}));
	}

	@Test
	public void testLeftIndexingScalarCP() 
	{
		runLeftIndexingTest(ExecType.CP);
	}
	
	@Test
	public void testLeftIndexingScalarMR() 
	{
		runLeftIndexingTest(ExecType.MR);
	}
	
	private void runLeftIndexingTest( ExecType instType ) 
	{		
		//rtplatform for MR
		RUNTIME_PLATFORM platformOld = rtplatform;
		rtplatform = (instType==ExecType.MR) ? RUNTIME_PLATFORM.HADOOP : RUNTIME_PLATFORM.HYBRID;
	
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args",  HOME + INPUT_DIR + "A" , 
							               		Long.toString(rows), 
							               		Long.toString(cols),
							               		HOME + OUTPUT_DIR + "A"};
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       HOME + INPUT_DIR + " "+ HOME + EXPECTED_DIR;
	
			loadTestConfiguration(config);
			
	        double[][] A = getRandomMatrix(rows, cols, min, max, sparsity, System.currentTimeMillis());
	        writeInputMatrix("A", A, true);
	       
	        runTest(true, false, null, -1);		
			runRScript(true);
			
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("A");
			HashMap<CellIndex, Double> rfile = readRMatrixFromFS("A");
			TestUtils.compareMatrices(dmlfile, rfile, epsilon, "A-DML", "A-R");
		}
		finally
		{
			rtplatform = platformOld;
		}
	}
}

