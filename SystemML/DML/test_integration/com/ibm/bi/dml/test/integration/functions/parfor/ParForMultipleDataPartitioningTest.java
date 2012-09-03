package com.ibm.bi.dml.test.integration.functions.parfor;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.hops.Hops;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

public class ParForMultipleDataPartitioningTest extends AutomatedTestBase 
{
	private final static String TEST_NAME = "parfor_mdatapartitioning";
	private final static String TEST_DIR = "functions/parfor/";
	private final static double eps = 1e-10;
	
	private final static int rows = (int)Hops.CPThreshold+1;
	private final static int cols = 10;    
	private final static double sparsity = 1.0;
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(
				TEST_NAME, 
				new TestConfiguration(TEST_DIR, TEST_NAME, 
				new String[] { "Rout" })   ); //TODO this specification is not intuitive
	}

	
	@Test
	public void testParForDataPartitioningEquivalentSchemes() 
	{
		runParForDataPartitioningTest(true);
	}

	@Test
	public void testParForDataPartitioningDifferentSchemes() 
	{
		runParForDataPartitioningTest(false);
	}
	
	
	/**
	 * 
	 * @param outer execution mode of outer parfor loop
	 * @param inner execution mode of inner parfor loop
	 * @param instType execution mode of instructions
	 */
	private void runParForDataPartitioningTest( boolean equiSchemes )
	{		
		//script
		int scriptNum = -1;
		if( equiSchemes )
			scriptNum = 1;  
		else
			scriptNum = 2;
		
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String HOME = SCRIPT_DIR + TEST_DIR;
		dmlArgs = new String[]{"-f", HOME + TEST_NAME +scriptNum + ".dml",
				               "-args", HOME + INPUT_DIR + "V" , 
				                        Integer.toString(rows),
				                        Integer.toString(cols),
				                        HOME + OUTPUT_DIR + "R" };
		dmlArgsDebug = new String[]{"-f", HOME + TEST_NAME + scriptNum + ".dml", "-d",
					               "-args", HOME + INPUT_DIR + "V" , 
						                   Integer.toString(rows),
						                   Integer.toString(cols),
						                   HOME + OUTPUT_DIR + "R" };
		
		rCmd = "Rscript" + " " + HOME + TEST_NAME + ".R" + " " + 
		       HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR + " " + scriptNum;
		
		loadTestConfiguration(config);

		long seed = System.nanoTime();
        double[][] V = getRandomMatrix(rows, cols, 0, 1, sparsity, seed);
		writeInputMatrix("V", V, true);

		boolean exceptionExpected = false;
		runTest(true, exceptionExpected, null, -1);
		runRScript(true);
		
		//compare matrices
		HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
		HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("Rout");
		TestUtils.compareMatrices(dmlfile, rfile, eps, "DML", "R");	
	}
}