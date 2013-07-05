package com.ibm.bi.dml.test.integration.functions.data;


import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;



/**
 * <p><b>Positive tests:</b></p>
 * <ul>
 * 	<li>text</li>
 * 	<li>binary</li>
 * 	<li>write a matrix two times</li>
 * </ul>
 * <p><b>Negative tests:</b></p>
 * 
 * 
 */
public class WriteMMTest extends AutomatedTestBase {

	private final static String TEST_NAME1 = "WriteMMTest";
	private final static String TEST_NAME2 = "WriteMMComplexTest";
	private final static String TEST_DIR = "functions/data/";
	
	//for CP
	private final static int rows1 = 30;
	private final static int cols1 = 10;
	//for MR
	private final static int rows2 = 700;  
	private final static int cols2 = 100;
	
		
	@Override
	public void setUp() 
	{
		addTestConfiguration(
				TEST_NAME1, 
				new TestConfiguration(TEST_DIR, TEST_NAME1, 
				new String[] { "B" })   ); 
		addTestConfiguration(
				TEST_NAME2, 
				new TestConfiguration(TEST_DIR, TEST_NAME2, 
				new String[] { "B" })   ); 
	}

	@Test
	public void testWriteMMCP() 
	{
		runWriteMMTest(ExecType.CP, TEST_NAME1);
	}
	
	@Test
	public void testWriteMMMR() 
	{
		runWriteMMTest(ExecType.MR, TEST_NAME1);
	}
	
	@Test
	public void testWriteMMMRMerge()
	{
		runWriteMMTest(ExecType.MR, TEST_NAME2);
	}
	
	private void runWriteMMTest( ExecType instType, String TEST_NAME )
	{
		//setup exec type, rows, cols
		int rows = -1, cols = -1;
		
		
		if( instType == ExecType.CP ) {
				rows = rows1;
				cols = cols1;
		}
		else { //if type MR
				rows = rows2;
				cols = cols2;
		}
			

		//rtplatform for MR
		RUNTIME_PLATFORM platformOld = rtplatform;
		rtplatform = (instType==ExecType.MR) ? RUNTIME_PLATFORM.HADOOP : RUNTIME_PLATFORM.HYBRID;
	
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", HOME + INPUT_DIR + "A" ,
					                        Integer.toString(rows),
					                        Integer.toString(cols),
					                        HOME + OUTPUT_DIR + "B"  };
			
			loadTestConfiguration(config);
	
			//generate actual dataset
			double[][] A = getRandomMatrix(rows, cols, -1, 1, 1, System.currentTimeMillis()); 
			writeInputMatrixWithMTD("A", A, false, new MatrixCharacteristics(rows,cols, 1000, 1000));
			writeExpectedMatrixMarket("B", A);
	
			runTest(true, false, null, -1);
			compareResultsWithMM();
		}
		finally
		{
			rtplatform = platformOld;
		}
	}
	
}
