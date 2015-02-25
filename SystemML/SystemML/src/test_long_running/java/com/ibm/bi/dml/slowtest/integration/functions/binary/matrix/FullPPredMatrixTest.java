/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.slowtest.integration.functions.binary.matrix;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

/**
 * The main purpose of this test is to verify various input combinations for
 * matrix-matrix ppred operations that internally translate to binary operations.
 * 
 */
public class FullPPredMatrixTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME1 = "PPredMatrixTest";
	private final static String TEST_DIR = "functions/binary/matrix/";
	private final static double eps = 1e-10;
	
	private final static int rows1 = 1383;
	private final static int cols1 = 1432;
	
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.01;
	
	public enum Type{
		GREATER,
		LESS,
		EQUALS,
		NOT_EQUALS,
		GREATER_EQUALS,
		LESS_EQUALS,
	}
		
	@Override
	public void setUp() 
	{
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_DIR, TEST_NAME1, new String[] { "C" }) ); 
		TestUtils.clearAssertionInformation();
	}
	
	@Test
	public void testPPredGreaterDenseDenseCP() 
	{
		runPPredTest(Type.GREATER, false, false, ExecType.CP);
	}
	
	@Test
	public void testPPredGreaterDenseSparseCP() 
	{
		runPPredTest(Type.GREATER, false, true, ExecType.CP);
	}
	
	@Test
	public void testPPredGreaterSparseDenseCP() 
	{
		runPPredTest(Type.GREATER, true, false, ExecType.CP);
	}
	
	@Test
	public void testPPredGreaterSparseSparseCP() 
	{
		runPPredTest(Type.GREATER, true, true, ExecType.CP);
	}
	
	@Test
	public void testPPredGreaterEqualsDenseDenseCP() 
	{
		runPPredTest(Type.GREATER_EQUALS, false, false, ExecType.CP);
	}
	
	@Test
	public void testPPredGreaterEqualsDenseSparseCP() 
	{
		runPPredTest(Type.GREATER_EQUALS, false, true, ExecType.CP);
	}
	
	@Test
	public void testPPredGreaterEqualsSparseDenseCP() 
	{
		runPPredTest(Type.GREATER_EQUALS, true, false, ExecType.CP);
	}
	
	@Test
	public void testPPredGreaterEqualsSparseSparseCP() 
	{
		runPPredTest(Type.GREATER_EQUALS, true, true, ExecType.CP);
	}
	
	@Test
	public void testPPredEqualsDenseDenseCP() 
	{
		runPPredTest(Type.EQUALS, false, false, ExecType.CP);
	}
	
	@Test
	public void testPPredEqualsDenseSparseCP() 
	{
		runPPredTest(Type.EQUALS, false, true, ExecType.CP);
	}
	
	@Test
	public void testPPredEqualsSparseDenseCP() 
	{
		runPPredTest(Type.EQUALS, true, false, ExecType.CP);
	}
	
	@Test
	public void testPPredEqualsSparseSparseCP() 
	{
		runPPredTest(Type.EQUALS, true, true, ExecType.CP);
	}
	
	@Test
	public void testPPredNotEqualsDenseDenseCP() 
	{
		runPPredTest(Type.NOT_EQUALS, false, false, ExecType.CP);
	}
	
	@Test
	public void testPPredNotEqualsDenseSparseCP() 
	{
		runPPredTest(Type.NOT_EQUALS, false, true, ExecType.CP);
	}
	
	@Test
	public void testPPredNotEqualsSparseDenseCP() 
	{
		runPPredTest(Type.NOT_EQUALS, true, false, ExecType.CP);
	}
	
	@Test
	public void testPPredNotEqualsSparseSparseCP() 
	{
		runPPredTest(Type.NOT_EQUALS, true, true, ExecType.CP);
	}
	
	@Test
	public void testPPredLessDenseDenseCP() 
	{
		runPPredTest(Type.LESS, false, false, ExecType.CP);
	}
	
	@Test
	public void testPPredLessDenseSparseCP() 
	{
		runPPredTest(Type.LESS, false, true, ExecType.CP);
	}
	
	@Test
	public void testPPredLessSparseDenseCP() 
	{
		runPPredTest(Type.LESS, true, false, ExecType.CP);
	}
	
	@Test
	public void testPPredLessSparseSparseCP() 
	{
		runPPredTest(Type.LESS, true, true, ExecType.CP);
	}
	
	@Test
	public void testPPredLessEqualsDenseDenseCP() 
	{
		runPPredTest(Type.LESS_EQUALS, false, false, ExecType.CP);
	}
	
	@Test
	public void testPPredLessEqualsDenseSparseCP() 
	{
		runPPredTest(Type.LESS_EQUALS, false, true, ExecType.CP);
	}
	
	@Test
	public void testPPredLessEqualsSparseDenseCP() 
	{
		runPPredTest(Type.LESS_EQUALS, true, false, ExecType.CP);
	}
	
	@Test
	public void testPPredLessEqualsSparseSparseCP() 
	{
		runPPredTest(Type.LESS_EQUALS, true, true, ExecType.CP);
	}
	
	
//TODO
	
	@Test
	public void testPPredGreaterDenseDenseMR() 
	{
		runPPredTest(Type.GREATER, false, false, ExecType.MR);
	}
	
	@Test
	public void testPPredGreaterDenseSparseMR() 
	{
		runPPredTest(Type.GREATER, false, true, ExecType.MR);
	}
	
	@Test
	public void testPPredGreaterSparseDenseMR() 
	{
		runPPredTest(Type.GREATER, true, false, ExecType.MR);
	}
	
	@Test
	public void testPPredGreaterSparseSparseMR() 
	{
		runPPredTest(Type.GREATER, true, true, ExecType.MR);
	}
	
	@Test
	public void testPPredGreaterEqualsDenseDenseMR() 
	{
		runPPredTest(Type.GREATER_EQUALS, false, false, ExecType.MR);
	}
	
	@Test
	public void testPPredGreaterEqualsDenseSparseMR() 
	{
		runPPredTest(Type.GREATER_EQUALS, false, true, ExecType.MR);
	}
	
	@Test
	public void testPPredGreaterEqualsSparseDenseMR() 
	{
		runPPredTest(Type.GREATER_EQUALS, true, false, ExecType.MR);
	}
	
	@Test
	public void testPPredGreaterEqualsSparseSparseMR() 
	{
		runPPredTest(Type.GREATER_EQUALS, true, true, ExecType.MR);
	}
	
	@Test
	public void testPPredEqualsDenseDenseMR() 
	{
		runPPredTest(Type.EQUALS, false, false, ExecType.MR);
	}
	
	@Test
	public void testPPredEqualsDenseSparseMR() 
	{
		runPPredTest(Type.EQUALS, false, true, ExecType.MR);
	}
	
	@Test
	public void testPPredEqualsSparseDenseMR() 
	{
		runPPredTest(Type.EQUALS, true, false, ExecType.MR);
	}
	
	@Test
	public void testPPredEqualsSparseSparseMR() 
	{
		runPPredTest(Type.EQUALS, true, true, ExecType.MR);
	}
	
	@Test
	public void testPPredNotEqualsDenseDenseMR() 
	{
		runPPredTest(Type.NOT_EQUALS, false, false, ExecType.MR);
	}
	
	@Test
	public void testPPredNotEqualsDenseSparseMR() 
	{
		runPPredTest(Type.NOT_EQUALS, false, true, ExecType.MR);
	}
	
	@Test
	public void testPPredNotEqualsSparseDenseMR() 
	{
		runPPredTest(Type.NOT_EQUALS, true, false, ExecType.MR);
	}
	
	@Test
	public void testPPredNotEqualsSparseSparseMR() 
	{
		runPPredTest(Type.NOT_EQUALS, true, true, ExecType.MR);
	}
	
	@Test
	public void testPPredLessDenseDenseMR() 
	{
		runPPredTest(Type.LESS, false, false, ExecType.MR);
	}
	
	@Test
	public void testPPredLessDenseSparseMR() 
	{
		runPPredTest(Type.LESS, false, true, ExecType.MR);
	}
	
	@Test
	public void testPPredLessSparseDenseMR() 
	{
		runPPredTest(Type.LESS, true, false, ExecType.MR);
	}
	
	@Test
	public void testPPredLessSparseSparseMR() 
	{
		runPPredTest(Type.LESS, true, true, ExecType.MR);
	}
	
	@Test
	public void testPPredLessEqualsDenseDenseMR() 
	{
		runPPredTest(Type.LESS_EQUALS, false, false, ExecType.MR);
	}
	
	@Test
	public void testPPredLessEqualsDenseSparseMR() 
	{
		runPPredTest(Type.LESS_EQUALS, false, true, ExecType.MR);
	}
	
	@Test
	public void testPPredLessEqualsSparseDenseMR() 
	{
		runPPredTest(Type.LESS_EQUALS, true, false, ExecType.MR);
	}
	
	@Test
	public void testPPredLessEqualsSparseSparseMR() 
	{
		runPPredTest(Type.LESS_EQUALS, true, true, ExecType.MR);
	}
	
	
	/**
	 * 
	 * @param type
	 * @param instType
	 * @param sparse
	 */
	private void runPPredTest( Type type, boolean sp1, boolean sp2, ExecType et )
	{
		String TEST_NAME = TEST_NAME1;
		int rows = rows1;
		int cols = cols1;
		
		//rtplatform for MR
		RUNTIME_PLATFORM platformOld = rtplatform;
		rtplatform = (et==ExecType.MR) ? RUNTIME_PLATFORM.HADOOP : RUNTIME_PLATFORM.HYBRID;
	
		double sparsityLeft = sp1 ? sparsity2 : sparsity1;
		double sparsityRight = sp2 ? sparsity2 : sparsity1;
		
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", HOME + INPUT_DIR + "A",
					                            HOME + INPUT_DIR + "B",
					                            Integer.toString(type.ordinal()),
					                            HOME + OUTPUT_DIR + "C"    };
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       HOME + INPUT_DIR + " " + type.ordinal() + " " + HOME + EXPECTED_DIR;
			
			loadTestConfiguration(config);
	
			//generate actual dataset
			double[][] A = getRandomMatrix(rows, cols, -10, 10, sparsityLeft, 7); 
			writeInputMatrixWithMTD("A", A, true);
			double[][] B = getRandomMatrix(rows, cols, -15, 15, sparsityRight, 3); 
			writeInputMatrixWithMTD("B", B, true);
			
			//run tests
			runTest(true, false, null, -1); 
			runRScript(true); 
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("C");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("C");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
		}
		finally
		{
			rtplatform = platformOld;
		}
	}
}