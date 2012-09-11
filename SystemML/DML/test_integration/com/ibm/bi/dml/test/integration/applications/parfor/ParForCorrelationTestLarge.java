package com.ibm.bi.dml.test.integration.applications.parfor;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.hops.Hops;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PExecMode;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

/**
 * Intension is to test file-based result merge with regard to its integration
 * with the different execution modes. Hence we need at least a dataset of size
 * CPThreshold^2
 * 
 * 
 */
public class ParForCorrelationTestLarge extends AutomatedTestBase 
{
	private final static String TEST_NAME = "parfor_corr_large";
	private final static String TEST_DIR = "applications/parfor/";
	private final static double eps = 1e-10;
	
	private final static int rows = (int)Hops.CPThreshold+1;  // # of rows in each vector (for MR instructions)
	private final static int cols = (int)Hops.CPThreshold+1;      // # of columns in each vector  
	
	private final static double minVal=0;    // minimum value in each vector 
	private final static double maxVal=1000; // maximum value in each vector 

	
	@Override
	public void setUp() 
	{
		addTestConfiguration(
				TEST_NAME, 
				new TestConfiguration(TEST_DIR, TEST_NAME, 
				new String[] { "Rout" })   ); //TODO this specification is not intuitive
	}

	
	@Test
	public void testParForCorrleationLargeLocalLocal() 
	{
		runParForCorrelationTest(PExecMode.LOCAL, PExecMode.LOCAL);
	}

	/*
	@Test
	public void testParForCorrleationLargeLocalRemote() 
	{
		runParForCorrelationTest(PExecMode.LOCAL, PExecMode.REMOTE_MR);
	}
	
	@Test
	public void testParForCorrleationRemoteLocalCP() 
	{
		runParForCorrelationTest(PExecMode.REMOTE_MR, PExecMode.LOCAL);
	}
	*/
	
	/**
	 * 
	 * @param outer execution mode of outer parfor loop
	 * @param inner execution mode of inner parfor loop
	 * @param instType execution mode of instructions
	 */
	private void runParForCorrelationTest( PExecMode outer, PExecMode inner )
	{
		//script
		int scriptNum = -1;
		if( inner == PExecMode.REMOTE_MR )      scriptNum=2;
		else if( outer == PExecMode.REMOTE_MR ) scriptNum=3;
		else 									scriptNum=1;
		
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String HOME = SCRIPT_DIR + TEST_DIR;
		dmlArgs = new String[]{"-f", HOME + TEST_NAME +scriptNum + ".dml",
				               "-args", HOME + INPUT_DIR + "V" , 
				                        Integer.toString(rows),
				                        Integer.toString(cols),
				                        HOME + OUTPUT_DIR + "PearsonR" };
		dmlArgsDebug = new String[]{"-f", HOME + TEST_NAME + scriptNum + ".dml", "-d",
					               "-args", HOME + INPUT_DIR + "V" , 
						                   Integer.toString(rows),
						                   Integer.toString(cols),
						                   HOME + OUTPUT_DIR + "PearsonR" };
		
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