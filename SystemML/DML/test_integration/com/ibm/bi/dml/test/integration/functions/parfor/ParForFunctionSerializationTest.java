package com.ibm.bi.dml.test.integration.functions.parfor;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.runtime.matrix.io.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

public class ParForFunctionSerializationTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "parfor_funct";
	private final static String TEST_NAME2 = "parfor_extfunct";
	private final static String TEST_DIR = "functions/parfor/";
	private final static double eps = 1e-10;
	
	private final static int rows = 20;
	private final static int cols = 10;    
	private final static double sparsity = 1.0;
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(
				TEST_NAME1, 
				new TestConfiguration(TEST_DIR, TEST_NAME1, 
				new String[] { "Rout" })   );
		addTestConfiguration(
				TEST_NAME2, 
				new TestConfiguration(TEST_DIR, TEST_NAME2, 
				new String[] { "Rout" })   );
	}

	@Test
	public void testParForFunctSerialization() 
	{
		runFunctionTest(1);
	}
	
	@Test
	public void testParForExtFunctSerialization() 
	{
		runFunctionTest(2);
	}

	
	private void runFunctionTest( int testNum )
	{
		String TEST_NAME = null;
		switch( testNum )
		{
			case 1: TEST_NAME = TEST_NAME1; break;
			case 2: TEST_NAME = TEST_NAME2; break;
		}
		
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[]{"-args", HOME + INPUT_DIR + "V" , 
				                        Integer.toString(rows),
				                        Integer.toString(cols),
				                        HOME + OUTPUT_DIR + "R" };
		fullRScriptName = HOME + TEST_NAME1 + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + 
		       HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR;
		
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