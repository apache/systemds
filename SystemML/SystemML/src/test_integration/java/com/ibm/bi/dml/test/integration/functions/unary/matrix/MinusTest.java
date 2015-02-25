/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.unary.matrix;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

public class MinusTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME1 = "Minus";
	private final static String TEST_DIR = "functions/unary/matrix/";

	private final static int rows = 1501;
	private final static int cols = 2502;
	
	private final static double sparsityDense = 0.7;
	private final static double sparsitySparse = 0.07;
	
	@Override
	public void setUp() 
	{
		TestUtils.clearAssertionInformation();
		
		addTestConfiguration(
				TEST_NAME1, 
				new TestConfiguration(TEST_DIR, TEST_NAME1, 
				new String[] { "Y" })   ); 
	}
	
	@Test
	public void testMinusDenseCP() 
	{
		runTestMinus( false, ExecType.CP );
	}
	
	@Test
	public void testMinusSparseCP() 
	{
		runTestMinus( true, ExecType.CP );
	}
	
	@Test
	public void testMinusDenseMR() 
	{
		runTestMinus( false, ExecType.MR );
	}
	
	@Test
	public void testMinusSparseMR() 
	{
		runTestMinus( true, ExecType.MR );
	}
		
	
	private void runTestMinus( boolean sparse, ExecType et )
	{		
		//handle rows and cols
		RUNTIME_PLATFORM platformOld = rtplatform;
	
		try
		{
			//register test configuration
			TestConfiguration config = getTestConfiguration(TEST_NAME1);
			config.addVariable("rows", rows);
			config.addVariable("cols", cols);
			
			loadTestConfiguration(config);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME1 + ".dml";
			programArgs = new String[]{"-args", HOME + INPUT_DIR + "X" , 
					                            String.valueOf(rows),
												String.valueOf(cols),
												HOME + OUTPUT_DIR + "Y" };
			fullRScriptName = HOME + TEST_NAME1 + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR;
			
			
			double sparsity = sparse ? sparsitySparse : sparsityDense;
			double[][] X = getRandomMatrix(rows, cols, 0, 1, sparsity, 7);
			writeInputMatrix("X", X, true);
	
			runTest(true, false, null, -1);
			runRScript(true);
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("Y");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("Y");
			TestUtils.compareMatrices(dmlfile, rfile, 1e-12, "Stat-DML", "Stat-R");
		}
		finally
		{
			//reset platform for additional tests
			rtplatform = platformOld;
		}
	}
}
