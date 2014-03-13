/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.binary.matrix;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

public class FullScalarPPredTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME1 = "ScalarPPredTest";
	private final static String TEST_DIR = "functions/binary/matrix/";
	private final static double eps = 1e-10;
	
	private final static int rows1 = 1072;
	private final static int cols1 = 1009;
	
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.1;
	
	public enum Type{
		GREATER,
		LESS,
		EQUALS,
		NOT_EQAULS,
		GREATER_EQUALS,
		LESS_EQUALS,
	}
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(
				TEST_NAME1, 
				new TestConfiguration(TEST_DIR, TEST_NAME1, 
				new String[] { "B" })   ); 
	}

	
	@Test
	public void testPPredGreaterZeroDenseCP() 
	{
		runPPredTest(Type.GREATER, true, false, ExecType.CP);
	}
	
	@Test
	public void testPPredLessZeroDenseCP() 
	{
		runPPredTest(Type.LESS, true, false, ExecType.CP);
	}
	
	@Test
	public void testPPredEqualsZeroDenseCP() 
	{
		runPPredTest(Type.EQUALS, true, false, ExecType.CP);
	}
	
	@Test
	public void testPPredNotEqualsZeroDenseCP() 
	{
		runPPredTest(Type.NOT_EQAULS, true, false, ExecType.CP);
	}
	
	@Test
	public void testPPredGreaterEqualsZeroDenseCP() 
	{
		runPPredTest(Type.GREATER_EQUALS, true, false, ExecType.CP);
	}
	
	@Test
	public void testPPredLessEqualsZeroDenseCP() 
	{
		runPPredTest(Type.LESS_EQUALS, true, false, ExecType.CP);
	}

	@Test
	public void testPPredGreaterNonZeroDenseCP() 
	{
		runPPredTest(Type.GREATER, false, false, ExecType.CP);
	}
	
	@Test
	public void testPPredLessNonZeroDenseCP() 
	{
		runPPredTest(Type.LESS, false, false, ExecType.CP);
	}
	
	@Test
	public void testPPredEqualsNonZeroDenseCP() 
	{
		runPPredTest(Type.EQUALS, false, false, ExecType.CP);
	}
	
	@Test
	public void testPPredNotEqualsNonZeroDenseCP() 
	{
		runPPredTest(Type.NOT_EQAULS, false, false, ExecType.CP);
	}
	
	@Test
	public void testPPredGreaterEqualsNonZeroDenseCP() 
	{
		runPPredTest(Type.GREATER_EQUALS, false, false, ExecType.CP);
	}
	
	@Test
	public void testPPredLessEqualsNonZeroDenseCP() 
	{
		runPPredTest(Type.LESS_EQUALS, false, false, ExecType.CP);
	}
	
	@Test
	public void testPPredGreaterZeroSparseCP() 
	{
		runPPredTest(Type.GREATER, true, true, ExecType.CP);
	}
	
	@Test
	public void testPPredLessZeroSparseCP() 
	{
		runPPredTest(Type.LESS, true, true, ExecType.CP);
	}
	
	@Test
	public void testPPredEqualsZeroSparseCP() 
	{
		runPPredTest(Type.EQUALS, true, true, ExecType.CP);
	}
	
	@Test
	public void testPPredNotEqualsZeroSparseCP() 
	{
		runPPredTest(Type.NOT_EQAULS, true, true, ExecType.CP);
	}
	
	@Test
	public void testPPredGreaterEqualsZeroSparseCP() 
	{
		runPPredTest(Type.GREATER_EQUALS, true, true, ExecType.CP);
	}
	
	@Test
	public void testPPredLessEqualsZeroSparseCP() 
	{
		runPPredTest(Type.LESS_EQUALS, true, true, ExecType.CP);
	}

	@Test
	public void testPPredGreaterNonZeroSparseCP() 
	{
		runPPredTest(Type.GREATER, false, true, ExecType.CP);
	}
	
	@Test
	public void testPPredLessNonZeroSparseCP() 
	{
		runPPredTest(Type.LESS, false, true, ExecType.CP);
	}
	
	@Test
	public void testPPredEqualsNonZeroSparseCP() 
	{
		runPPredTest(Type.EQUALS, false, true, ExecType.CP);
	}
	
	@Test
	public void testPPredNotEqualsNonZeroSparseCP() 
	{
		runPPredTest(Type.NOT_EQAULS, false, true, ExecType.CP);
	}
	
	@Test
	public void testPPredGreaterEqualsNonZeroSparseCP() 
	{
		runPPredTest(Type.GREATER_EQUALS, false, true, ExecType.CP);
	}
	
	@Test
	public void testPPredLessEqualsNonZeroSparseCP() 
	{
		runPPredTest(Type.LESS_EQUALS, false, true, ExecType.CP);
	}

	@Test
	public void testPPredGreaterZeroDenseMR() 
	{
		runPPredTest(Type.GREATER, true, false, ExecType.MR);
	}
	
	@Test
	public void testPPredLessZeroDenseMR() 
	{
		runPPredTest(Type.LESS, true, false, ExecType.MR);
	}
	
	@Test
	public void testPPredEqualsZeroDenseMR() 
	{
		runPPredTest(Type.EQUALS, true, false, ExecType.MR);
	}
	
	@Test
	public void testPPredNotEqualsZeroDenseMR() 
	{
		runPPredTest(Type.NOT_EQAULS, true, false, ExecType.MR);
	}
	
	@Test
	public void testPPredGreaterEqualsZeroDenseMR() 
	{
		runPPredTest(Type.GREATER_EQUALS, true, false, ExecType.MR);
	}
	
	@Test
	public void testPPredLessEqualsZeroDenseMR() 
	{
		runPPredTest(Type.LESS_EQUALS, true, false, ExecType.MR);
	}

	@Test
	public void testPPredGreaterNonZeroDenseMR() 
	{
		runPPredTest(Type.GREATER, false, false, ExecType.MR);
	}
	
	@Test
	public void testPPredLessNonZeroDenseMR() 
	{
		runPPredTest(Type.LESS, false, false, ExecType.MR);
	}
	
	@Test
	public void testPPredEqualsNonZeroDenseMR() 
	{
		runPPredTest(Type.EQUALS, false, false, ExecType.MR);
	}
	
	@Test
	public void testPPredNotEqualsNonZeroDenseMR() 
	{
		runPPredTest(Type.NOT_EQAULS, false, false, ExecType.MR);
	}
	
	@Test
	public void testPPredGreaterEqualsNonZeroDenseMR() 
	{
		runPPredTest(Type.GREATER_EQUALS, false, false, ExecType.MR);
	}
	
	@Test
	public void testPPredLessEqualsNonZeroDenseMR() 
	{
		runPPredTest(Type.LESS_EQUALS, false, false, ExecType.MR);
	}
	
	@Test
	public void testPPredGreaterZeroSparseMR() 
	{
		runPPredTest(Type.GREATER, true, true, ExecType.MR);
	}
	
	@Test
	public void testPPredLessZeroSparseMR() 
	{
		runPPredTest(Type.LESS, true, true, ExecType.MR);
	}
	
	@Test
	public void testPPredEqualsZeroSparseMR() 
	{
		runPPredTest(Type.EQUALS, true, true, ExecType.MR);
	}
	
	@Test
	public void testPPredNotEqualsZeroSparseMR() 
	{
		runPPredTest(Type.NOT_EQAULS, true, true, ExecType.MR);
	}
	
	@Test
	public void testPPredGreaterEqualsZeroSparseMR() 
	{
		runPPredTest(Type.GREATER_EQUALS, true, true, ExecType.MR);
	}
	
	@Test
	public void testPPredLessEqualsZeroSparseMR() 
	{
		runPPredTest(Type.LESS_EQUALS, true, true, ExecType.MR);
	}

	@Test
	public void testPPredGreaterNonZeroSparseMR() 
	{
		runPPredTest(Type.GREATER, false, true, ExecType.MR);
	}
	
	@Test
	public void testPPredLessNonZeroSparseMR() 
	{
		runPPredTest(Type.LESS, false, true, ExecType.MR);
	}
	
	@Test
	public void testPPredEqualsNonZeroSparseMR() 
	{
		runPPredTest(Type.EQUALS, false, true, ExecType.MR);
	}
	
	@Test
	public void testPPredNotEqualsNonZeroSparseMR() 
	{
		runPPredTest(Type.NOT_EQAULS, false, true, ExecType.MR);
	}
	
	@Test
	public void testPPredGreaterEqualsNonZeroSparseMR() 
	{
		runPPredTest(Type.GREATER_EQUALS, false, true, ExecType.MR);
	}
	
	@Test
	public void testPPredLessEqualsNonZeroSparseMR() 
	{
		runPPredTest(Type.LESS_EQUALS, false, true, ExecType.MR);
	}
	
	
	/**
	 * 
	 * @param type
	 * @param instType
	 * @param sparse
	 */
	private void runPPredTest( Type type, boolean zero, boolean sparse, ExecType et )
	{
		String TEST_NAME = TEST_NAME1;
		int rows = rows1;
		int cols = cols1;
		double sparsity = sparse ? sparsity2 : sparsity1;
		double constant = zero ? 0 : 0.5;
		
		//rtplatform for MR
		RUNTIME_PLATFORM platformOld = rtplatform;
		rtplatform = (et==ExecType.MR) ? RUNTIME_PLATFORM.HADOOP : RUNTIME_PLATFORM.HYBRID;
	
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", HOME + INPUT_DIR + "A" ,
					                        Integer.toString(rows),
					                        Integer.toString(cols),
					                        Integer.toString(type.ordinal()),
					                        Double.toString(constant),
					                        HOME + OUTPUT_DIR + "B"    };
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       HOME + INPUT_DIR + " " + type.ordinal() + " " + constant + " " + HOME + EXPECTED_DIR;
			
			loadTestConfiguration(config);
	
			//generate actual dataset
			double[][] A = getRandomMatrix(rows, cols, -1, 1, sparsity, 7); 
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