package com.ibm.bi.dml.test.integration.functions.recompile;

import java.util.HashMap;

import junit.framework.Assert;

import org.junit.Test;

import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;
import com.ibm.bi.dml.utils.Statistics;

public class FunctionRecompileTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "funct_recompile";
	private final static String TEST_DIR = "functions/recompile/";
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
	}

	@Test
	public void testFunctionWithoutRecompile() 
	{
		runFunctionTest(false);
	}
	
	@Test
	public void testFunctionWithRecompile() 
	{
		runFunctionTest(true);
	}


	
	private void runFunctionTest( boolean recompile )
	{	
		boolean oldFlag = OptimizerUtils.ALLOW_DYN_RECOMPILATION;
		
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME1);
			config.addVariable("rows", rows);
			config.addVariable("cols", cols);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME1 + ".dml";
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
	
			OptimizerUtils.ALLOW_DYN_RECOMPILATION = recompile;
			boolean exceptionExpected = false;
			runTest(true, exceptionExpected, null, -1); 
			runRScript(true);
			
			//check expected number of compiled and executed MR jobs
			if( recompile )
			{
				Assert.assertEquals("Unexpected number of executed MR jobs.", 
						  0, Statistics.getNoOfExecutedMRJobs()); //reblock, 10*(GMR,MMCJ,GMR), GMR write			
			}
			else
			{
				Assert.assertEquals("Unexpected number of executed MR jobs.", 
						            41, Statistics.getNoOfExecutedMRJobs()); //reblock, 10*(GMR,MMCJ,GMR), GMR write
			}
			
			//compare matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("Rout");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "DML", "R");			
		}
		finally
		{
			OptimizerUtils.ALLOW_DYN_RECOMPILATION = oldFlag;
		}
	}
	
}