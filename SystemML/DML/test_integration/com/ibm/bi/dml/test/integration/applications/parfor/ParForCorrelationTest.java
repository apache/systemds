/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.applications.parfor;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.hops.Hop;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PExecMode;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

public class ParForCorrelationTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME = "parfor_corr";
	private final static String TEST_DIR = "applications/parfor/";
	private final static double eps = 1e-10;
	
	private final static int rows1 = (int)Hop.CPThreshold;  // # of rows in each vector (for CP instructions)
	private final static int rows2 = (int)Hop.CPThreshold+1;  // # of rows in each vector (for MR instructions)
	private final static int cols1 = 20;      // # of columns in each vector  
	
	private final static double minVal=0;    // minimum value in each vector 
	private final static double maxVal=1000; // maximum value in each vector 

	
	@Override
	public void setUp() 
	{
		addTestConfiguration(
				TEST_NAME, 
				new TestConfiguration(TEST_DIR, TEST_NAME, 
				new String[] { "Rout" })   );
	}

	@Test
	public void testForCorrleationSerialSerialCP() 
	{
		runParForCorrelationTest(false, PExecMode.LOCAL, PExecMode.LOCAL, ExecType.CP, false);
	}

	@Test
	public void testForCorrleationSerialSerialMR() 
	{
		runParForCorrelationTest(false, PExecMode.LOCAL, PExecMode.LOCAL, ExecType.MR, false);
	}
	
	@Test
	public void testParForCorrleationLocalLocalCP() 
	{
		runParForCorrelationTest(true, PExecMode.LOCAL, PExecMode.LOCAL, ExecType.CP, false);
	}

	@Test
	public void testParForCorrleationLocalLocalMR() 
	{
		runParForCorrelationTest(true, PExecMode.LOCAL, PExecMode.LOCAL, ExecType.MR, false);
	}

	@Test
	public void testParForCorrleationLocalRemoteCP() 
	{
		runParForCorrelationTest(true, PExecMode.LOCAL, PExecMode.REMOTE_MR, ExecType.CP, false);
	}
	
	@Test
	public void testParForCorrleationRemoteLocalCP() 
	{
		runParForCorrelationTest(true, PExecMode.REMOTE_MR, PExecMode.LOCAL, ExecType.CP, false);
	}
	

	@Test
	public void testParForCorrleationDefaultCP() 
	{
		runParForCorrelationTest(true, null, null, ExecType.CP, false);
	}
	
	@Test
	public void testParForCorrleationDefaultMR() 
	{
		runParForCorrelationTest(true, null, null, ExecType.MR, false);
	}
	
	@Test
	public void testParForCorrleationDefaultMRWithProfile() 
	{
		runParForCorrelationTest(true, null, null, ExecType.MR, true);
	}
	
	
	/**
	 * 
	 * @param outer execution mode of outer parfor loop
	 * @param inner execution mode of inner parfor loop
	 * @param instType execution mode of instructions
	 */
	private void runParForCorrelationTest( boolean parallel, PExecMode outer, PExecMode inner, ExecType instType, boolean profile )
	{
		//inst exec type, influenced via rows
		int rows = -1;
		if( instType == ExecType.CP )
			rows = rows1;
		else //if type MR
			rows = rows2;
		int cols = cols1;
		
		//script
		int scriptNum = -1;
		if( parallel )
		{
			if( inner == PExecMode.REMOTE_MR )      scriptNum=2;
			else if( outer == PExecMode.REMOTE_MR ) scriptNum=3;
			else if( outer == PExecMode.LOCAL )		scriptNum=1;
			else if( !profile)                      scriptNum=4; //optimized
			else                                    scriptNum=5; //optimized with profile
		}
		else
		{
			scriptNum = 0;
		}
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + scriptNum + ".dml";
		programArgs = new String[]{"-args", HOME + INPUT_DIR + "V" , 
				                        Integer.toString(rows),
				                        Integer.toString(cols),
				                        HOME + OUTPUT_DIR + "PearsonR" };
		fullRScriptName = HOME + TEST_NAME + ".R";
		rCmd = "Rscript" + " " + HOME + TEST_NAME + ".R" + " " + 
		       HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR;
		
		loadTestConfiguration(config);

		long seed = System.nanoTime();
        double[][] V = getRandomMatrix(rows, cols, minVal, maxVal, 1.0, seed);
		writeInputMatrix("V", V, true);

		boolean exceptionExpected = false;
		runTest(true, exceptionExpected, null, -1);
		runRScript(true);
		
		//compare matrices
		HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("PearsonR");
		HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("Rout");
		TestUtils.compareMatrices(dmlfile, rfile, eps, "PearsonR-DML", "PearsonR-R");
		
	}
}