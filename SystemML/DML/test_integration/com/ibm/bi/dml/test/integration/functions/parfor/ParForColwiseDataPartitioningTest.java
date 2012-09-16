package com.ibm.bi.dml.test.integration.functions.parfor;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.hops.Hops;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PDataPartitioner;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PExecMode;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

public class ParForColwiseDataPartitioningTest extends AutomatedTestBase 
{
	private final static String TEST_NAME = "parfor_cdatapartitioning";
	private final static String TEST_DIR = "functions/parfor/";
	private final static double eps = 1e-10;
	
	private final static int rows1 = (int)Hops.CPThreshold;
	private final static int rows2 = 10*(int)Hops.CPThreshold;
	private final static int cols = 50;      // # of columns 
	private final static double sparsity1 = 1.0;
	private final static double sparsity2 = 0.1d;
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(
				TEST_NAME, 
				new TestConfiguration(TEST_DIR, TEST_NAME, 
				new String[] { "Rout" })   ); //TODO this specification is not intuitive
	}

	
	@Test
	public void testParForDataPartitioningNoneLocalLargeDense() 
	{
		runParForDataPartitioningTest(PDataPartitioner.NONE, null, false, false);
	}

	@Test
	public void testParForDataPartitioningNoneLocalLargeSparse() 
	{
		runParForDataPartitioningTest(PDataPartitioner.NONE, PExecMode.LOCAL, false, true);
	}
	
	@Test
	public void testParForDataPartitioningLocalLocalLargeDense() 
	{
		runParForDataPartitioningTest(PDataPartitioner.LOCAL, PExecMode.LOCAL, false, false);
	}

	@Test
	public void testParForDataPartitioningLocalLocalLargeSparse() 
	{
		runParForDataPartitioningTest(PDataPartitioner.LOCAL, PExecMode.LOCAL, false, true);
	}
	
	@Test
	public void testParForDataPartitioningLocalRemoteLargeDense() 
	{
		runParForDataPartitioningTest(PDataPartitioner.LOCAL, PExecMode.REMOTE_MR, false, false);
	}

	@Test
	public void testParForDataPartitioningLocalRemoteLargeSparse() 
	{
		runParForDataPartitioningTest(PDataPartitioner.LOCAL, PExecMode.REMOTE_MR, false, true);
	}
	
	
	
	/**
	 * 
	 * @param outer execution mode of outer parfor loop
	 * @param inner execution mode of inner parfor loop
	 * @param instType execution mode of instructions
	 */
	private void runParForDataPartitioningTest( PDataPartitioner partitioner, PExecMode mode, boolean small, boolean sparse )
	{
		//inst exec type, influenced via rows
		int rows = rows2; //always large to test 
			
		//script
		int scriptNum = -1;
		switch( partitioner )
		{
			case NONE: 
				scriptNum=1; 
				break; 
			case LOCAL: 
				if( mode==PExecMode.LOCAL )
					scriptNum=2; 
				else
					scriptNum=3;				
				break; 
			//case REMOTE_MR: 
			//	if( mode==PExecMode.REMOTE_MR )
			//		scriptNum=4;
			//	else
			//		scriptNum=5;
			//	break; 
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
				                        HOME + OUTPUT_DIR + "R" };
		fullRScriptName = HOME + TEST_NAME + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + 
		       HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR;
		
		loadTestConfiguration(config);

		long seed = System.nanoTime();
		double sparsity = -1;
		if( sparse )
			sparsity = sparsity2;
		else
			sparsity = sparsity1;
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