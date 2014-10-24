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

public class FullMatrixVectorCellwiseOperationTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME1 = "FullMatrixVectorCellwiseOperation_Addition";
	private final static String TEST_NAME2 = "FullMatrixVectorCellwiseOperation_Substraction";
	private final static String TEST_NAME3 = "FullMatrixVectorCellwiseOperation_Multiplication";
	private final static String TEST_NAME4 = "FullMatrixVectorCellwiseOperation_Division";
	
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
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.DENSE, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testAdditionDenseSparseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.DENSE, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testAdditionDenseEmptyCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.DENSE, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testAdditionSparseDenseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.SPARSE, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testAdditionSparseSparseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.SPARSE, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testAdditionSparseEmptyCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.SPARSE, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testAdditionEmptyDenseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.EMPTY, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testAdditionEmptySparseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.EMPTY, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testAdditionEmptyEmptyCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.EMPTY, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testAdditionDenseDenseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.DENSE, SparsityType.DENSE, ExecType.MR);
	}
	
	@Test
	public void testAdditionDenseSparseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.DENSE, SparsityType.SPARSE, ExecType.MR);
	}
	
	@Test
	public void testAdditionDenseEmptyMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.DENSE, SparsityType.EMPTY, ExecType.MR);
	}
	
	@Test
	public void testAdditionSparseDenseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.SPARSE, SparsityType.DENSE, ExecType.MR);
	}
	
	@Test
	public void testAdditionSparseSparseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.SPARSE, SparsityType.SPARSE, ExecType.MR);
	}
	
	@Test
	public void testAdditionSparseEmptyMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.SPARSE, SparsityType.EMPTY, ExecType.MR);
	}
	
	@Test
	public void testAdditionEmptyDenseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.EMPTY, SparsityType.DENSE, ExecType.MR);
	}
	
	@Test
	public void testAdditionEmptySparseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.EMPTY, SparsityType.SPARSE, ExecType.MR);
	}
	
	@Test
	public void testAdditionEmptyEmptyMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.ADDITION, SparsityType.EMPTY, SparsityType.EMPTY, ExecType.MR);
	}	
	
	@Test
	public void testSubstractionDenseDenseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.DENSE, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testSubstractionDenseSparseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.DENSE, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testSubstractionDenseEmptyCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.DENSE, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testSubstractionSparseDenseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.SPARSE, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testSubstractionSparseSparseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.SPARSE, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testSubstractionSparseEmptyCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.SPARSE, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testSubstractionEmptyDenseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.EMPTY, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testSubstractionEmptySparseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.EMPTY, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testSubstractionEmptyEmptyCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.EMPTY, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testSubstractionDenseDenseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.DENSE, SparsityType.DENSE, ExecType.MR);
	}
	
	@Test
	public void testSubstractionDenseSparseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.DENSE, SparsityType.SPARSE, ExecType.MR);
	}
	
	@Test
	public void testSubstractionDenseEmptyMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.DENSE, SparsityType.EMPTY, ExecType.MR);
	}
	
	@Test
	public void testSubstractionSparseDenseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.SPARSE, SparsityType.DENSE, ExecType.MR);
	}
	
	@Test
	public void testSubstractionSparseSparseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.SPARSE, SparsityType.SPARSE, ExecType.MR);
	}
	
	@Test
	public void testSubstractionSparseEmptyMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.SPARSE, SparsityType.EMPTY, ExecType.MR);
	}
	
	@Test
	public void testSubstractionEmptyDenseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.EMPTY, SparsityType.DENSE, ExecType.MR);
	}
	
	@Test
	public void testSubstractionEmptySparseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.EMPTY, SparsityType.SPARSE, ExecType.MR);
	}
	
	@Test
	public void testSubstractionEmptyEmptyMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.SUBSTRACTION, SparsityType.EMPTY, SparsityType.EMPTY, ExecType.MR);
	}
	
	@Test
	public void testMultiplicationDenseDenseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.DENSE, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testMultiplicationDenseSparseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.DENSE, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testMultiplicationDenseEmptyCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.DENSE, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testMultiplicationSparseDenseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.SPARSE, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testMultiplicationSparseSparseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.SPARSE, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testMultiplicationSparseEmptyCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.SPARSE, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testMultiplicationEmptyDenseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.EMPTY, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testMultiplicationEmptySparseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.EMPTY, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testMultiplicationEmptyEmptyCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.EMPTY, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testMultiplicationDenseDenseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.DENSE, SparsityType.DENSE, ExecType.MR);
	}
	
	@Test
	public void testMultiplicationDenseSparseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.DENSE, SparsityType.SPARSE, ExecType.MR);
	}
	
	@Test
	public void testMultiplicationDenseEmptyMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.DENSE, SparsityType.EMPTY, ExecType.MR);
	}
	
	@Test
	public void testMultiplicationSparseDenseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.SPARSE, SparsityType.DENSE, ExecType.MR);
	}
	
	@Test
	public void testMultiplicationSparseSparseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.SPARSE, SparsityType.SPARSE, ExecType.MR);
	}
	
	@Test
	public void testMultiplicationSparseEmptyMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.SPARSE, SparsityType.EMPTY, ExecType.MR);
	}
	
	@Test
	public void testMultiplicationEmptyDenseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.EMPTY, SparsityType.DENSE, ExecType.MR);
	}
	
	@Test
	public void testMultiplicationEmptySparseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.EMPTY, SparsityType.SPARSE, ExecType.MR);
	}
	
	@Test
	public void testMultiplicationEmptyEmptyMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.MULTIPLICATION, SparsityType.EMPTY, SparsityType.EMPTY, ExecType.MR);
	}
	
	@Test
	public void testDivisionDenseDenseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.DIVISION, SparsityType.DENSE, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testDivisionDenseSparseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.DIVISION, SparsityType.DENSE, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testDivisionDenseEmptyCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.DIVISION, SparsityType.DENSE, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testDivisionSparseDenseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.DIVISION, SparsityType.SPARSE, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testDivisionSparseSparseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.DIVISION, SparsityType.SPARSE, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testDivisionSparseEmptyCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.DIVISION, SparsityType.SPARSE, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testDivisionEmptyDenseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.DIVISION, SparsityType.EMPTY, SparsityType.DENSE, ExecType.CP);
	}
	
	@Test
	public void testDivisionEmptySparseCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.DIVISION, SparsityType.EMPTY, SparsityType.SPARSE, ExecType.CP);
	}
	
	@Test
	public void testDivisionEmptyEmptyCP() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.DIVISION, SparsityType.EMPTY, SparsityType.EMPTY, ExecType.CP);
	}
	
	@Test
	public void testDivisionDenseDenseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.DIVISION, SparsityType.DENSE, SparsityType.DENSE, ExecType.MR);
	}
	
	@Test
	public void testDivisionDenseSparseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.DIVISION, SparsityType.DENSE, SparsityType.SPARSE, ExecType.MR);
	}
	
	@Test
	public void testDivisionDenseEmptyMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.DIVISION, SparsityType.DENSE, SparsityType.EMPTY, ExecType.MR);
	}
	
	@Test
	public void testDivisionSparseDenseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.DIVISION, SparsityType.SPARSE, SparsityType.DENSE, ExecType.MR);
	}
	
	@Test
	public void testDivisionSparseSparseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.DIVISION, SparsityType.SPARSE, SparsityType.SPARSE, ExecType.MR);
	}
	
	@Test
	public void testDivisionSparseEmptyMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.DIVISION, SparsityType.SPARSE, SparsityType.EMPTY, ExecType.MR);
	}
	
	@Test
	public void testDivisionEmptyDenseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.DIVISION, SparsityType.EMPTY, SparsityType.DENSE, ExecType.MR);
	}
	
	@Test
	public void testDivisionEmptySparseMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.DIVISION, SparsityType.EMPTY, SparsityType.SPARSE, ExecType.MR);
	}
	
	@Test
	public void testDivisionEmptyEmptyMR() 
	{
		runMatrixVectorCellwiseOperationTest(OpType.DIVISION, SparsityType.EMPTY, SparsityType.EMPTY, ExecType.MR);
	}
	
	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
	private void runMatrixVectorCellwiseOperationTest( OpType type, SparsityType sparseM1, SparsityType sparseM2, ExecType instType)
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
			programArgs = new String[]{"-explain","recompile_runtime","-args", HOME + INPUT_DIR + "A",
					                        HOME + INPUT_DIR + "B",
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
			writeInputMatrixWithMTD("A", A, true);
			double[][] B = getRandomMatrix(rows, 1, 0, (lsparsity2==0)?0:1, lsparsity2, 3); 
			writeInputMatrixWithMTD("B", B, true);
	
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
						C[i][j] = A[i][j]/B[i][0];
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