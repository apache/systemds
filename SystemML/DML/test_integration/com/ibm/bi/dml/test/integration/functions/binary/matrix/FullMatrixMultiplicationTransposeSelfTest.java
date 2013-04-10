package com.ibm.bi.dml.test.integration.functions.binary.matrix;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.lops.MMTSJ.MMTSJType;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

public class FullMatrixMultiplicationTransposeSelfTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "TransposeSelfMatrixMultiplication1";
	private final static String TEST_NAME2 = "TransposeSelfMatrixMultiplication2";
	private final static String TEST_DIR = "functions/binary/matrix/";
	private final static double eps = 1e-10;
	
	//for CP
	private final static int rows1 = 3500;
	private final static int cols1 = 1500;
	//for MR
	private final static int rows2 = 7000;//7000;  
	private final static int cols2 = 750;//750; 
	
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.1;
	
	
	
	
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
	public void testMMLeftDenseCP() 
	{
		runTransposeSelfMatrixMultiplicationTest(MMTSJType.LEFT, ExecType.CP, false);
	}
	
	@Test
	public void testMMRightDenseCP() 
	{
		runTransposeSelfMatrixMultiplicationTest(MMTSJType.RIGHT, ExecType.CP, false);
	}

	@Test
	public void testMMLeftSparseCP() 
	{
		runTransposeSelfMatrixMultiplicationTest(MMTSJType.LEFT, ExecType.CP, true);
	}
	
	@Test
	public void testMMRightSparseCP() 
	{
		runTransposeSelfMatrixMultiplicationTest(MMTSJType.RIGHT, ExecType.CP, true);
	}
	
	@Test
	public void testMMLeftDenseMR() 
	{
		runTransposeSelfMatrixMultiplicationTest(MMTSJType.LEFT, ExecType.MR, false);
	}
	
	@Test
	public void testMMRightDenseMR() 
	{
		runTransposeSelfMatrixMultiplicationTest(MMTSJType.RIGHT, ExecType.MR, false);
	}

	@Test
	public void testMMLeftSparseMR() 
	{
		runTransposeSelfMatrixMultiplicationTest(MMTSJType.LEFT, ExecType.MR, true);
	}
	
	@Test
	public void testMMRightSparseMR() 
	{
		runTransposeSelfMatrixMultiplicationTest(MMTSJType.RIGHT, ExecType.MR, true);
	}	
	
	@Test
	public void testVVLeftDenseCP() 
	{
		runTransposeSelfMatrixMultiplicationTest(MMTSJType.LEFT, ExecType.CP, false);
	}
	
	@Test
	public void testVRightDenseCP() 
	{
		runTransposeSelfMatrixMultiplicationTest(MMTSJType.RIGHT, ExecType.CP, false);
	}

	@Test
	public void testVVLeftSparseCP() 
	{
		runTransposeSelfMatrixMultiplicationTest(MMTSJType.LEFT, ExecType.CP, true);
	}
	
	@Test
	public void testVVRightSparseCP() 
	{
		runTransposeSelfMatrixMultiplicationTest(MMTSJType.RIGHT, ExecType.CP, true);
	}
	
	@Test
	public void testVVLeftDenseMR() 
	{
		runTransposeSelfMatrixMultiplicationTest(MMTSJType.LEFT, ExecType.MR, false);
	}
	
	@Test
	public void testVVRightDenseMR() 
	{
		runTransposeSelfMatrixMultiplicationTest(MMTSJType.RIGHT, ExecType.MR, false);
	}

	@Test
	public void testVVLeftSparseMR() 
	{
		runTransposeSelfMatrixMultiplicationTest(MMTSJType.LEFT, ExecType.MR, true);
	}
	
	@Test
	public void testVVRightSparseMR() 
	{
		runTransposeSelfMatrixMultiplicationTest(MMTSJType.RIGHT, ExecType.MR, true);
	}

	/**
	 * 
	 * @param type
	 * @param instType
	 * @param sparse
	 */
	private void runTransposeSelfMatrixMultiplicationTest( MMTSJType type, ExecType instType, boolean sparse )
	{
		//setup exec type, rows, cols
		int rows = -1, cols = -1;
		String TEST_NAME = null;
		if( type == MMTSJType.LEFT ) {
			if( instType == ExecType.CP ) {
				rows = rows1;
				cols = cols1;
			}
			else { //if type MR
				rows = rows2;
				cols = cols2;
			}
			TEST_NAME = TEST_NAME1;
		}
		else {
			if( instType == ExecType.CP ) {
				rows = cols1;
				cols = rows1;
			}
			else { //if type MR
				rows = cols2;
				cols = rows2;
			}
			TEST_NAME = TEST_NAME2;
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
					                        HOME + OUTPUT_DIR + "B"    };
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR;
			
			loadTestConfiguration(config);
	
			//generate actual dataset
			double[][] A = getRandomMatrix(rows, cols, 0, 1, sparse?sparsity2:sparsity1, 7); 
			writeInputMatrix("A", A, true);
			
	
			boolean exceptionExpected = false;
			runTest(true, exceptionExpected, null, -1); 
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("B");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("B");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
		}
		finally
		{
			rtplatform = platformOld;
		}
	}
	
	/**
	 * 
	 * @param type
	 * @param instType
	 * @param sparse
	 */
	private void runTransposeSelfVectorMultiplicationTest( MMTSJType type, ExecType instType, boolean sparse )
	{
		//setup exec type, rows, cols
		int rows = -1, cols = -1;
		String TEST_NAME = null;
		if( type == MMTSJType.LEFT ) {
			if( instType == ExecType.CP ) {
				rows = rows1;
				cols = 1;
			}
			else { //if type MR
				rows = rows2;
				cols = 1;
			}
			TEST_NAME = TEST_NAME1;
		}
		else {
			if( instType == ExecType.CP ) {
				rows = 1;
				cols = rows1;
			}
			else { //if type MR
				rows = 1;
				cols = rows2;
			}
			TEST_NAME = TEST_NAME2;
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
					                        HOME + OUTPUT_DIR + "B"    };
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR;
			
			loadTestConfiguration(config);
	
			//generate actual dataset
			double[][] A = getRandomMatrix(rows, cols, 0, 1, sparse?sparsity2:sparsity1, 7); 
			writeInputMatrix("A", A, true);
			
	
			boolean exceptionExpected = false;
			runTest(true, exceptionExpected, null, -1); 
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("B");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("B");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
		}
		finally
		{
			rtplatform = platformOld;
		}
	}	
	
}