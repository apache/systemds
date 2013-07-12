package com.ibm.bi.dml.test.integration.functions.aggregate;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

/**
 * NOTES:
 *  * the R package Matrix does not support colMins and colMaxs; hence, we use the matrixStats package 
 * 
 */
public class FullRowAggregateTest extends AutomatedTestBase 
{
	private final static String TEST_NAME1 = "RowSums";
	private final static String TEST_NAME2 = "RowMeans";
	private final static String TEST_NAME3 = "RowMaxs";
	private final static String TEST_NAME4 = "RowMins";
	
	private final static String TEST_DIR = "functions/aggregate/";
	private final static double eps = 1e-10;
	
	private final static int rows1 = 1;
	private final static int rows2 = 1079;
	private final static int cols = 1005;
	private final static double sparsity1 = 0.1;
	private final static double sparsity2 = 0.7;
	
	private enum OpType{
		ROW_SUMS,
		ROW_MEANS,
		ROW_MAX,
		ROW_MIN
	}
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME1,new TestConfiguration(TEST_DIR, TEST_NAME1,new String[]{"B"})); 
		addTestConfiguration(TEST_NAME2,new TestConfiguration(TEST_DIR, TEST_NAME2,new String[]{"B"})); 
		addTestConfiguration(TEST_NAME3,new TestConfiguration(TEST_DIR, TEST_NAME3,new String[]{"B"})); 
		addTestConfiguration(TEST_NAME4,new TestConfiguration(TEST_DIR, TEST_NAME4,new String[]{"B"})); 
	}

	
	@Test
	public void testRowSumsDenseMatrixCP() 
	{
		runRowAggregateOperationTest(OpType.ROW_SUMS, false, false, ExecType.CP);
	}
	
	@Test
	public void testRowMeansDenseMatrixCP() 
	{
		runRowAggregateOperationTest(OpType.ROW_MEANS, false, false, ExecType.CP);
	}	
	
	@Test
	public void testRowMaxDenseMatrixCP() 
	{
		runRowAggregateOperationTest(OpType.ROW_MAX, false, false, ExecType.CP);
	}
	
	@Test
	public void testRowMinDenseMatrixCP() 
	{
		runRowAggregateOperationTest(OpType.ROW_MIN, false, false, ExecType.CP);
	}
	
	@Test
	public void testRowSumsDenseVectorCP() 
	{
		runRowAggregateOperationTest(OpType.ROW_SUMS, false, true, ExecType.CP);
	}
	
	@Test
	public void testRowMeansDenseVectorCP() 
	{
		runRowAggregateOperationTest(OpType.ROW_MEANS, false, true, ExecType.CP);
	}	
	
	@Test
	public void testRowMaxDenseVectorCP() 
	{
		runRowAggregateOperationTest(OpType.ROW_MAX, false, true, ExecType.CP);
	}
	
	@Test
	public void testRowMinDenseVectorCP() 
	{
		runRowAggregateOperationTest(OpType.ROW_MIN, false, true, ExecType.CP);
	}
	
	@Test
	public void testRowSumsSparseMatrixCP() 
	{
		runRowAggregateOperationTest(OpType.ROW_SUMS, true, false, ExecType.CP);
	}
	
	@Test
	public void testRowMeansSparseMatrixCP() 
	{
		runRowAggregateOperationTest(OpType.ROW_MEANS, true, false, ExecType.CP);
	}	
	
	@Test
	public void testRowMaxSparseMatrixCP() 
	{
		runRowAggregateOperationTest(OpType.ROW_MAX, true, false, ExecType.CP);
	}
	
	@Test
	public void testRowMinSparseMatrixCP() 
	{
		runRowAggregateOperationTest(OpType.ROW_MIN, true, false, ExecType.CP);
	}
	
	@Test
	public void testRowSumsSparseVectorCP() 
	{
		runRowAggregateOperationTest(OpType.ROW_SUMS, true, true, ExecType.CP);
	}
	
	@Test
	public void testRowMeansSparseVectorCP() 
	{
		runRowAggregateOperationTest(OpType.ROW_MEANS, true, true, ExecType.CP);
	}	
	
	@Test
	public void testRowMaxSparseVectorCP() 
	{
		runRowAggregateOperationTest(OpType.ROW_MAX, true, true, ExecType.CP);
	}
	
	@Test
	public void testRowMinSparseVectorCP() 
	{
		runRowAggregateOperationTest(OpType.ROW_MIN, true, true, ExecType.CP);
	}
	
	@Test
	public void testRowSumsDenseMatrixMR() 
	{
		runRowAggregateOperationTest(OpType.ROW_SUMS, false, false, ExecType.MR);
	}
	
	@Test
	public void testRowMeansDenseMatrixMR() 
	{
		runRowAggregateOperationTest(OpType.ROW_MEANS, false, false, ExecType.MR);
	}	
	
	@Test
	public void testRowMaxDenseMatrixMR() 
	{
		runRowAggregateOperationTest(OpType.ROW_MAX, false, false, ExecType.MR);
	}
	
	@Test
	public void testRowMinDenseMatrixMR() 
	{
		runRowAggregateOperationTest(OpType.ROW_MIN, false, false, ExecType.MR);
	}
	
	@Test
	public void testRowSumsDenseVectorMR() 
	{
		runRowAggregateOperationTest(OpType.ROW_SUMS, false, true, ExecType.MR);
	}
	
	@Test
	public void testRowMeansDenseVectorMR() 
	{
		runRowAggregateOperationTest(OpType.ROW_MEANS, false, true, ExecType.MR);
	}	
	
	@Test
	public void testRowMaxDenseVectorMR() 
	{
		runRowAggregateOperationTest(OpType.ROW_MAX, false, true, ExecType.MR);
	}
	
	@Test
	public void testRowMinDenseVectorMR() 
	{
		runRowAggregateOperationTest(OpType.ROW_MIN, false, true, ExecType.MR);
	}
	
	@Test
	public void testRowSumsSparseMatrixMR() 
	{
		runRowAggregateOperationTest(OpType.ROW_SUMS, true, false, ExecType.MR);
	}
	
	@Test
	public void testRowMeansSparseMatrixMR() 
	{
		runRowAggregateOperationTest(OpType.ROW_MEANS, true, false, ExecType.MR);
	}	
	
	@Test
	public void testRowMaxSparseMatrixMR() 
	{
		runRowAggregateOperationTest(OpType.ROW_MAX, true, false, ExecType.MR);
	}
	
	@Test
	public void testRowMinSparseMatrixMR() 
	{
		runRowAggregateOperationTest(OpType.ROW_MIN, true, false, ExecType.MR);
	}
	
	@Test
	public void testRowSumsSparseVectorMR() 
	{
		runRowAggregateOperationTest(OpType.ROW_SUMS, true, true, ExecType.MR);
	}
	
	@Test
	public void testRowMeansSparseVectorMR() 
	{
		runRowAggregateOperationTest(OpType.ROW_MEANS, true, true, ExecType.MR);
	}	
	
	@Test
	public void testRowMaxSparseVectorMR() 
	{
		runRowAggregateOperationTest(OpType.ROW_MAX, true, true, ExecType.MR);
	}
	
	@Test
	public void testRowMinSparseVectorMR() 
	{
		runRowAggregateOperationTest(OpType.ROW_MIN, true, true, ExecType.MR);
	}
	
	
	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
	private void runRowAggregateOperationTest( OpType type, boolean sparse, boolean vector, ExecType instType)
	{
		//rtplatform for MR
		RUNTIME_PLATFORM platformOld = rtplatform;
		rtplatform = (instType==ExecType.MR) ? RUNTIME_PLATFORM.HADOOP : RUNTIME_PLATFORM.HYBRID;
	
		try
		{
			String TEST_NAME = null;
			switch( type )
			{
				case ROW_SUMS: TEST_NAME = TEST_NAME1; break;
				case ROW_MEANS: TEST_NAME = TEST_NAME2; break;
				case ROW_MAX: TEST_NAME = TEST_NAME3; break;
				case ROW_MIN: TEST_NAME = TEST_NAME4; break;
			}
			
			int rows = (vector) ? rows1 : rows2;
			double sparsity = (sparse) ? sparsity1 : sparsity2;
			
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", HOME + INPUT_DIR + "A",
					                        Integer.toString(rows),
					                        Integer.toString(cols),
					                        HOME + OUTPUT_DIR + "B"    };
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR;
			
			loadTestConfiguration(config);
	
			//generate actual dataset
			double[][] A = getRandomMatrix(rows, cols, 0, 1, sparsity, 7); 
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