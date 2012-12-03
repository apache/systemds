package com.ibm.bi.dml.test.integration.functions.recompile;

import java.io.IOException;
import java.util.HashMap;

import junit.framework.Assert;

import org.junit.Test;

import com.ibm.bi.dml.parser.DMLTranslator;
import com.ibm.bi.dml.runtime.matrix.io.InputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue.CellIndex;
import com.ibm.bi.dml.runtime.util.DataConverter;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;
import com.ibm.bi.dml.utils.Statistics;

public class ReblockRecompileTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "rblk_recompile1";
	private final static String TEST_NAME2 = "rblk_recompile2";
	private final static String TEST_NAME3 = "rblk_recompile3";
	private final static String TEST_DIR = "functions/recompile/";
	private final static double eps = 1e-10;
	
	private final static int rows = 20;   
	
	
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
		addTestConfiguration(
				TEST_NAME3, 
				new TestConfiguration(TEST_DIR, TEST_NAME3, 
				new String[] { "Rout" })   );
	}
	
	@Test
	public void testReblockPWrite() 
	{
		runReblockTest(1);
	}

	@Test
	public void testReblockCTable() 
	{
		runReblockTest(2);
	}
	
	@Test
	public void testReblockGroupedAggregate() 
	{
		runReblockTest(3);
	}
	
	private void runReblockTest(int scriptNum)
	{
		String TEST_NAME = null;
		switch(scriptNum) 
		{
			case 1: TEST_NAME = TEST_NAME1; break;
			case 2: TEST_NAME = TEST_NAME2; break;
			case 3: TEST_NAME = TEST_NAME3; break;
		}
		
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		config.addVariable("rows", rows);
		
		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME + ".dml";
		programArgs = new String[]{"-args", HOME + INPUT_DIR + "V" , 
				                        Integer.toString(rows),
				                        HOME + OUTPUT_DIR + "R" };
		fullRScriptName = HOME + TEST_NAME + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + 
		       HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR;
		
		loadTestConfiguration(config);

		long seed = System.nanoTime();
        double[][] V = getRandomMatrix(rows, 1, 1, 5, 1.0d, seed);
		writeInputMatrix("V", V, true);
		
		//cleanup previous executions
		try {
			MapReduceTool.deleteFileIfExistOnHDFS(HOME + OUTPUT_DIR + "R" );
		} catch (IOException e1){}
		
		boolean exceptionExpected = false;
		runTest(true, exceptionExpected, null, -1); //0 due to recompile 
		runRScript(true);
		
		Assert.assertEquals("Unexpected number of executed MR jobs.", 
				  			0, Statistics.getNoOfExecutedMRJobs());
		
		//compare matrices		
		try 
		{
			MatrixBlock mo = DataConverter.readMatrixFromHDFS(HOME + OUTPUT_DIR+"R", InputInfo.BinaryBlockInputInfo, rows, 1, DMLTranslator.DMLBlockSize, DMLTranslator.DMLBlockSize);
			HashMap<CellIndex, Double> dmlfile = new HashMap<CellIndex,Double>();
			for( int i=0; i<mo.getNumRows(); i++ )
				for( int j=0; j<mo.getNumColumns(); j++ )
					dmlfile.put(new CellIndex(i+1,j+1), mo.getValue(i, j));
				
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("Rout");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "DML", "R");	
		} 
		catch (IOException e) {
			Assert.fail(e.getMessage());
		}
	}
}