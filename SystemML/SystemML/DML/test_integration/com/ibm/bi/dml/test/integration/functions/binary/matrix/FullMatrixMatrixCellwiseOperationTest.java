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

public class FullMatrixMatrixCellwiseOperationTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME1 = "FullMatrixCellwiseOperation_Addition";
	private final static String TEST_NAME2 = "FullMatrixCellwiseOperation_Substraction";
	private final static String TEST_NAME3 = "FullMatrixCellwiseOperation_Multiplication";
	private final static String TEST_NAME4 = "FullMatrixCellwiseOperation_Division";
	
	private final static String TEST_DIR = "functions/binary/matrix/";
	private final static double eps = 1e-10;
	
	private final static int rows = 1100;
	private final static int cols = 900;
	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.1;
	
	private enum OpType{
		ADDITION,
		SUBSTRACTION,
		MULTIPLICATION,
		DIVISION
	}
	
	private enum SparsityType{
		DENSE,
		SPARSE,
		EMPTY
	}
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME1,new TestConfiguration(TEST_DIR, TEST_NAME1,new String[]{"C"})); 
		addTestConfiguration(TEST_NAME2,new TestConfiguration(TEST_DIR, TEST_NAME2,new String[]{"C"})); 
		addTestConfiguration(TEST_NAME3,new TestConfiguration(TEST_DIR, TEST_NAME3,new String[]{"C"})); 
		addTestConfiguration(TEST_NAME4,new TestConfiguration(TEST_DIR, TEST_NAME4,new String[]{"C"})); 
	}

	
	@Test
	public void testAdditionDenseDenseCP() 
	{
		runMatrixCellwiseOperationTest(OpType.ADDITION, SparsityType.DENSE, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testAdditionDenseSparseCP() 
	{
		runMatrixCellwiseOperationTest(OpType.ADDITION, SparsityType.DENSE, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testAdditionDenseEmptyCP() 
	{
		runMatrixCellwiseOperationTest(OpType.ADDITION, SparsityType.DENSE, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testAdditionSparseDenseCP() 
	{
		runMatrixCellwiseOperationTest(OpType.ADDITION, SparsityType.SPARSE, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testAdditionSparseSparseCP() 
	{
		runMatrixCellwiseOperationTest(OpType.ADDITION, SparsityType.SPARSE, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testAdditionSparseEmptyCP() 
	{
		runMatrixCellwiseOperationTest(OpType.ADDITION, SparsityType.SPARSE, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testAdditionEmptyDenseCP() 
	{
		runMatrixCellwiseOperationTest(OpType.ADDITION, SparsityType.EMPTY, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testAdditionEmptySparseCP() 
	{
		runMatrixCellwiseOperationTest(OpType.ADDITION, SparsityType.EMPTY, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testAdditionEmptyEmptyCP() 
	{
		runMatrixCellwiseOperationTest(OpType.ADDITION, SparsityType.EMPTY, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testAdditionDenseDenseMR() 
	{
		runMatrixCellwiseOperationTest(OpType.ADDITION, SparsityType.DENSE, SparsityType.DENSE, ExecType.MR);
	}
	
	@Test
	public void testAdditionDenseSparseMR() 
	{
		runMatrixCellwiseOperationTest(OpType.ADDITION, SparsityType.DENSE, SparsityType.SPARSE, ExecType.MR);
	}
	
	@Test
	public void testAdditionDenseEmptyMR() 
	{
		runMatrixCellwiseOperationTest(OpType.ADDITION, SparsityType.DENSE, SparsityType.EMPTY, ExecType.MR);
	}
	
	@Test
	public void testAdditionSparseDenseMR() 
	{
		runMatrixCellwiseOperationTest(OpType.ADDITION, SparsityType.SPARSE, SparsityType.DENSE, ExecType.MR);
	}
	
	@Test
	public void testAdditionSparseSparseMR() 
	{
		runMatrixCellwiseOperationTest(OpType.ADDITION, SparsityType.SPARSE, SparsityType.SPARSE, ExecType.MR);
	}
	
	@Test
	public void testAdditionSparseEmptyMR() 
	{
		runMatrixCellwiseOperationTest(OpType.ADDITION, SparsityType.SPARSE, SparsityType.EMPTY, ExecType.MR);
	}
	
	@Test
	public void testAdditionEmptyDenseMR() 
	{
		runMatrixCellwiseOperationTest(OpType.ADDITION, SparsityType.EMPTY, SparsityType.DENSE, ExecType.MR);
	}
	
	@Test
	public void testAdditionEmptySparseMR() 
	{
		runMatrixCellwiseOperationTest(OpType.ADDITION, SparsityType.EMPTY, SparsityType.SPARSE, ExecType.MR);
	}
	
	@Test
	public void testAdditionEmptyEmptyMR() 
	{
		runMatrixCellwiseOperationTest(OpType.ADDITION, SparsityType.EMPTY, SparsityType.EMPTY, ExecType.MR);
	}	
	
	@Test
	public void testSubstractionDenseDenseCP() 
	{
		runMatrixCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.DENSE, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testSubstractionDenseSparseCP() 
	{
		runMatrixCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.DENSE, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testSubstractionDenseEmptyCP() 
	{
		runMatrixCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.DENSE, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testSubstractionSparseDenseCP() 
	{
		runMatrixCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.SPARSE, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testSubstractionSparseSparseCP() 
	{
		runMatrixCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.SPARSE, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testSubstractionSparseEmptyCP() 
	{
		runMatrixCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.SPARSE, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testSubstractionEmptyDenseCP() 
	{
		runMatrixCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.EMPTY, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testSubstractionEmptySparseCP() 
	{
		runMatrixCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.EMPTY, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testSubstractionEmptyEmptyCP() 
	{
		runMatrixCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.EMPTY, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testSubstractionDenseDenseMR() 
	{
		runMatrixCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.DENSE, SparsityType.DENSE, ExecType.MR);
	}
	
	@Test
	public void testSubstractionDenseSparseMR() 
	{
		runMatrixCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.DENSE, SparsityType.SPARSE, ExecType.MR);
	}
	
	@Test
	public void testSubstractionDenseEmptyMR() 
	{
		runMatrixCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.DENSE, SparsityType.EMPTY, ExecType.MR);
	}
	
	@Test
	public void testSubstractionSparseDenseMR() 
	{
		runMatrixCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.SPARSE, SparsityType.DENSE, ExecType.MR);
	}
	
	@Test
	public void testSubstractionSparseSparseMR() 
	{
		runMatrixCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.SPARSE, SparsityType.SPARSE, ExecType.MR);
	}
	
	@Test
	public void testSubstractionSparseEmptyMR() 
	{
		runMatrixCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.SPARSE, SparsityType.EMPTY, ExecType.MR);
	}
	
	@Test
	public void testSubstractionEmptyDenseMR() 
	{
		runMatrixCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.EMPTY, SparsityType.DENSE, ExecType.MR);
	}
	
	@Test
	public void testSubstractionEmptySparseMR() 
	{
		runMatrixCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.EMPTY, SparsityType.SPARSE, ExecType.MR);
	}
	
	@Test
	public void testSubstractionEmptyEmptyMR() 
	{
		runMatrixCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.EMPTY, SparsityType.EMPTY, ExecType.MR);
	}
	
	@Test
	public void testMultiplicationDenseDenseCP() 
	{
		runMatrixCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.DENSE, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testMultiplicationDenseSparseCP() 
	{
		runMatrixCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.DENSE, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testMultiplicationDenseEmptyCP() 
	{
		runMatrixCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.DENSE, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testMultiplicationSparseDenseCP() 
	{
		runMatrixCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.SPARSE, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testMultiplicationSparseSparseCP() 
	{
		runMatrixCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.SPARSE, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testMultiplicationSparseEmptyCP() 
	{
		runMatrixCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.SPARSE, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testMultiplicationEmptyDenseCP() 
	{
		runMatrixCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.EMPTY, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testMultiplicationEmptySparseCP() 
	{
		runMatrixCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.EMPTY, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testMultiplicationEmptyEmptyCP() 
	{
		runMatrixCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.EMPTY, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testMultiplicationDenseDenseMR() 
	{
		runMatrixCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.DENSE, SparsityType.DENSE, ExecType.MR);
	}
	
	@Test
	public void testMultiplicationDenseSparseMR() 
	{
		runMatrixCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.DENSE, SparsityType.SPARSE, ExecType.MR);
	}
	
	@Test
	public void testMultiplicationDenseEmptyMR() 
	{
		runMatrixCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.DENSE, SparsityType.EMPTY, ExecType.MR);
	}
	
	@Test
	public void testMultiplicationSparseDenseMR() 
	{
		runMatrixCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.SPARSE, SparsityType.DENSE, ExecType.MR);
	}
	
	@Test
	public void testMultiplicationSparseSparseMR() 
	{
		runMatrixCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.SPARSE, SparsityType.SPARSE, ExecType.MR);
	}
	
	@Test
	public void testMultiplicationSparseEmptyMR() 
	{
		runMatrixCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.SPARSE, SparsityType.EMPTY, ExecType.MR);
	}
	
	@Test
	public void testMultiplicationEmptyDenseMR() 
	{
		runMatrixCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.EMPTY, SparsityType.DENSE, ExecType.MR);
	}
	
	@Test
	public void testMultiplicationEmptySparseMR() 
	{
		runMatrixCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.EMPTY, SparsityType.SPARSE, ExecType.MR);
	}
	
	@Test
	public void testMultiplicationEmptyEmptyMR() 
	{
		runMatrixCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.EMPTY, SparsityType.EMPTY, ExecType.MR);
	}
	
	@Test
	public void testDivisionDenseDenseCP() 
	{
		runMatrixCellwiseOperationTest(OpType.DIVISION, SparsityType.DENSE, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testDivisionDenseSparseCP() 
	{
		runMatrixCellwiseOperationTest(OpType.DIVISION, SparsityType.DENSE, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testDivisionDenseEmptyCP() 
	{
		runMatrixCellwiseOperationTest(OpType.DIVISION, SparsityType.DENSE, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testDivisionSparseDenseCP() 
	{
		runMatrixCellwiseOperationTest(OpType.DIVISION, SparsityType.SPARSE, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testDivisionSparseSparseCP() 
	{
		runMatrixCellwiseOperationTest(OpType.DIVISION, SparsityType.SPARSE, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testDivisionSparseEmptyCP() 
	{
		runMatrixCellwiseOperationTest(OpType.DIVISION, SparsityType.SPARSE, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testDivisionEmptyDenseCP() 
	{
		runMatrixCellwiseOperationTest(OpType.DIVISION, SparsityType.EMPTY, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testDivisionEmptySparseCP() 
	{
		runMatrixCellwiseOperationTest(OpType.DIVISION, SparsityType.EMPTY, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testDivisionEmptyEmptyCP() 
	{
		runMatrixCellwiseOperationTest(OpType.DIVISION, SparsityType.EMPTY, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testDivisionDenseDenseMR() 
	{
		runMatrixCellwiseOperationTest(OpType.DIVISION, SparsityType.DENSE, SparsityType.DENSE, ExecType.MR);
	}
	
	@Test
	public void testDivisionDenseSparseMR() 
	{
		runMatrixCellwiseOperationTest(OpType.DIVISION, SparsityType.DENSE, SparsityType.SPARSE, ExecType.MR);
	}
	
	@Test
	public void testDivisionDenseEmptyMR() 
	{
		runMatrixCellwiseOperationTest(OpType.DIVISION, SparsityType.DENSE, SparsityType.EMPTY, ExecType.MR);
	}
	
	@Test
	public void testDivisionSparseDenseMR() 
	{
		runMatrixCellwiseOperationTest(OpType.DIVISION, SparsityType.SPARSE, SparsityType.DENSE, ExecType.MR);
	}
	
	@Test
	public void testDivisionSparseSparseMR() 
	{
		runMatrixCellwiseOperationTest(OpType.DIVISION, SparsityType.SPARSE, SparsityType.SPARSE, ExecType.MR);
	}
	
	@Test
	public void testDivisionSparseEmptyMR() 
	{
		runMatrixCellwiseOperationTest(OpType.DIVISION, SparsityType.SPARSE, SparsityType.EMPTY, ExecType.MR);
	}
	
	@Test
	public void testDivisionEmptyDenseMR() 
	{
		runMatrixCellwiseOperationTest(OpType.DIVISION, SparsityType.EMPTY, SparsityType.DENSE, ExecType.MR);
	}
	
	@Test
	public void testDivisionEmptySparseMR() 
	{
		runMatrixCellwiseOperationTest(OpType.DIVISION, SparsityType.EMPTY, SparsityType.SPARSE, ExecType.MR);
	}
	
	@Test
	public void testDivisionEmptyEmptyMR() 
	{
		runMatrixCellwiseOperationTest(OpType.DIVISION, SparsityType.EMPTY, SparsityType.EMPTY, ExecType.MR);
	}
	
	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
	private void runMatrixCellwiseOperationTest( OpType type, SparsityType sparseM1, SparsityType sparseM2, ExecType instType)
	{
		//rtplatform for MR
		RUNTIME_PLATFORM platformOld = rtplatform;
		rtplatform = (instType==ExecType.MR) ? RUNTIME_PLATFORM.HADOOP : RUNTIME_PLATFORM.HYBRID;
	
		try
		{
			String TEST_NAME = null;
			switch( type )
			{
				case ADDITION: TEST_NAME = TEST_NAME1; break;
				case SUBSTRACTION: TEST_NAME = TEST_NAME2; break;
				case MULTIPLICATION: TEST_NAME = TEST_NAME3; break;
				case DIVISION: TEST_NAME = TEST_NAME4; break;
			}
			
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", HOME + INPUT_DIR + "A",
					                        Integer.toString(rows),
					                        Integer.toString(cols),
					                        HOME + INPUT_DIR + "B",
					                        Integer.toString(rows),
					                        Integer.toString(cols),
					                        HOME + OUTPUT_DIR + "C"    };
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR;
			
			loadTestConfiguration(config);
	
			//get sparsity
			double lsparsity1 = 1.0, lsparsity2 = 1.0;
			switch( sparseM1 ){
				case DENSE: lsparsity1 = sparsity1; break;
				case SPARSE: lsparsity1 = sparsity2; break;
				case EMPTY: lsparsity1 = 0.0; break;
			}
			switch( sparseM2 ){
				case DENSE: lsparsity2 = sparsity1; break;
				case SPARSE: lsparsity2 = sparsity2; break;
				case EMPTY: lsparsity2 = 0.0; break;
			}
			
			//generate actual dataset
			double[][] A = getRandomMatrix(rows, cols, 0, (lsparsity1==0)?0:1, lsparsity1, 7); 
			writeInputMatrix("A", A, true);
			double[][] B = getRandomMatrix(rows, cols, 0, (lsparsity2==0)?0:1, lsparsity2, 3); 
			writeInputMatrix("B", B, true);
	
			boolean exceptionExpected = false;
			runTest(true, exceptionExpected, null, -1); 
			
			if( !(type==OpType.DIVISION) )
			{
				runRScript(true); 
			
				//compare matrices 
				HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("C");
				HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("C");
				TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			}
			else
			{
				//For division, IEEE 754 defines x/0.0 as INFINITY and 0.0/0.0 as NaN.
				//Java handles this correctly while R always returns 1.0E308 in those cases.
				//Hence, we directly write the expected results.
				
				double C[][] = new double[rows][cols];
				for( int i=0; i<rows; i++ )
					for( int j=0; j<cols; j++ )
						C[i][j] = A[i][j]/B[i][j];
				writeExpectedMatrix("C", C);
				
				compareResults();
			}
		}
		finally
		{
			rtplatform = platformOld;
		}
	}
	
		
}