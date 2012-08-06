package com.ibm.bi.dml.test.integration.applications.parfor;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PExecMode;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

public class ParForCorrelationTest extends AutomatedTestBase 
{
	private final static String TEST_NAME = "parfor_corr";
	private final static String TEST_DIR = "applications/parfor/";
	private final static double eps = 1e-10;
	
	private final static int rows1 = 1000;  // # of rows in each vector (for CP instructions)
	private final static int rows2 = 10000;  // # of rows in each vector (for MR instructions)
	private final static int cols = 20;      // # of columns in each vector  
	
	private final static double minVal=0;    // minimum value in each vector 
	private final static double maxVal=1000; // maximum value in each vector 

	
	@Override
	public void setUp() 
	{
		addTestConfiguration(
				TEST_NAME, 
				new TestConfiguration(TEST_DIR, TEST_NAME, 
				new String[] { "Rout" })   ); //TODO this specification is not intuitiv
	}
	
	@Test
	public void testParForCorrleationLocalLocalCP() 
	{
		runParForCorrelationTest(PExecMode.LOCAL, PExecMode.LOCAL, ExecType.CP);
	}

	/* FIXME: decomment as soon as indexing is fixed.
	@Test
	public void testParForCorrleationLocalLocalMR() 
	{
		runParForCorrelationTest(PExecMode.LOCAL, PExecMode.LOCAL, ExecType.MR);
	}*/

	@Test
	public void testParForCorrleationLocalRemoteCP() 
	{
		runParForCorrelationTest(PExecMode.LOCAL, PExecMode.REMOTE_MR, ExecType.CP);
	}
	
	@Test
	public void testParForCorrleationRemoteLocalCP() 
	{
		runParForCorrelationTest(PExecMode.REMOTE_MR, PExecMode.LOCAL, ExecType.CP);
	}
	
	/**
	 * 
	 * @param outer execution mode of outer parfor loop
	 * @param inner execution mode of inner parfor loop
	 * @param instType execution mode of instructions
	 */
	private void runParForCorrelationTest( PExecMode outer, PExecMode inner, ExecType instType )
	{
		//inst exec type, influenced via rows
		int rows = -1;
		if( instType == ExecType.CP )
			rows = rows1;
		else //if type MR
			rows = rows2;
		
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
		
		rCmd = "Rscript" + " " + HOME + TEST_NAME + ".r" + " " + 
		       HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR;
		
		loadTestConfiguration(config);

		long seed = System.nanoTime();
        double[][] V = getRandomMatrix(rows, cols, minVal, maxVal, 0.1, seed);
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