/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

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
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME1 = "RowSums";
	private final static String TEST_NAME2 = "RowMeans";
	private final static String TEST_NAME3 = "RowMaxs";
	private final static String TEST_NAME4 = "RowMins";
	private final static String TEST_NAME5 = "RowIndexMaxs";
	private final static String TEST_NAME6 = "RowIndexMins";
	
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
		ROW_MIN,
		ROW_INDEXMAX,
		ROW_INDEXMIN
	}
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME1,new TestConfiguration(TEST_DIR, TEST_NAME1,new String[]{"B"})); 
		addTestConfiguration(TEST_NAME2,new TestConfiguration(TEST_DIR, TEST_NAME2,new String[]{"B"})); 
		addTestConfiguration(TEST_NAME3,new TestConfiguration(TEST_DIR, TEST_NAME3,new String[]{"B"})); 
		addTestConfiguration(TEST_NAME4,new TestConfiguration(TEST_DIR, TEST_NAME4,new String[]{"B"}));
		addTestConfiguration(TEST_NAME5,new TestConfiguration(TEST_DIR, TEST_NAME5,new String[]{"B"}));
		addTestConfiguration(TEST_NAME6,new TestConfiguration(TEST_DIR, TEST_NAME6,new String[]{"B"}));
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
	public void testRowIndexMaxDenseMatrixCP() 
	{
		runRowAggregateOperationTest(OpType.ROW_INDEXMAX, false, false, ExecType.CP);
	}
	
	@Test
	public void testRowIndexMinDenseMatrixCP() 
	{
		runRowAggregateOperationTest(OpType.ROW_INDEXMIN, false, false, ExecType.CP);
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
	public void testRowIndexMaxDenseVectorCP() 
	{
		runRowAggregateOperationTest(OpType.ROW_INDEXMAX, false, true, ExecType.CP);
	}
	
	@Test
	public void testRowIndexMinDenseVectorCP() 
	{
		runRowAggregateOperationTest(OpType.ROW_INDEXMIN, false, true, ExecType.CP);
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
	public void testRowIndexMaxSparseMatrixCP() 
	{
		runRowAggregateOperationTest(OpType.ROW_INDEXMAX, true, false, ExecType.CP);
	}
	
	@Test
	public void testRowIndexMinparseMatrixCP() 
	{
		runRowAggregateOperationTest(OpType.ROW_INDEXMIN, true, false, ExecType.CP);
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
	public void testRowIndexMaxSparseVectorCP() 
	{
		runRowAggregateOperationTest(OpType.ROW_INDEXMAX, true, true, ExecType.CP);
	}
	
	@Test
	public void testRowIndexMinSparseVectorCP() 
	{
		runRowAggregateOperationTest(OpType.ROW_INDEXMIN, true, true, ExecType.CP);
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
	public void testRowIndexMaxDenseMatrixMR() 
	{
		runRowAggregateOperationTest(OpType.ROW_INDEXMAX, false, false, ExecType.MR);
	}
	
	@Test
	public void testRowIndexMinDenseMatrixMR() 
	{
		runRowAggregateOperationTest(OpType.ROW_INDEXMIN, false, false, ExecType.MR);
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
	public void testRowIndexMaxDenseVectorMR() 
	{
		runRowAggregateOperationTest(OpType.ROW_INDEXMAX, false, true, ExecType.MR);
	}
	
	@Test
	public void testRowIndexMinDenseVectorMR() 
	{
		runRowAggregateOperationTest(OpType.ROW_INDEXMIN, false, true, ExecType.MR);
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
	public void testRowIndexMaxSparseMatrixMR() 
	{
		runRowAggregateOperationTest(OpType.ROW_INDEXMAX, true, false, ExecType.MR);
	}
	
	@Test
	public void testRowIndexMinSparseMatrixMR() 
	{
		runRowAggregateOperationTest(OpType.ROW_INDEXMIN, true, false, ExecType.MR);
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
	
	@Test
	public void testRowIndexMaxSparseVectorMR() 
	{
		runRowAggregateOperationTest(OpType.ROW_INDEXMAX, true, true, ExecType.MR);
	}
	
	@Test
	public void testRowIndexMinSparseVectorMR() 
	{
		runRowAggregateOperationTest(OpType.ROW_INDEXMIN, true, true, ExecType.MR);
	}
	
	@Test
	public void testRowIndexMaxDenseMatrixNegCP() 
	{
		runRowAggregateOperationTest(OpType.ROW_INDEXMAX, false, false, ExecType.CP, true);
	}
	
	@Test
	public void testRowIndexMaxDenseVectorNegCP() 
	{
		runRowAggregateOperationTest(OpType.ROW_INDEXMAX, false, true, ExecType.CP, true);
	}
	
	@Test
	public void testRowIndexMaxSparseMatrixNegCP() 
	{
		runRowAggregateOperationTest(OpType.ROW_INDEXMAX, true, false, ExecType.CP, true);
	}
	
	@Test
	public void testRowIndexMaxSparseVectorNegCP() 
	{
		runRowAggregateOperationTest(OpType.ROW_INDEXMAX, true, true, ExecType.CP, true);
	}
	
	@Test
	public void testRowIndexMaxDenseMatrixNegMR() 
	{
		runRowAggregateOperationTest(OpType.ROW_INDEXMAX, false, false, ExecType.MR, true);
	}
	
	@Test
	public void testRowIndexMaxDenseVectorNegMR() 
	{
		runRowAggregateOperationTest(OpType.ROW_INDEXMAX, false, true, ExecType.MR, true);
	}
	
	
	@Test
	public void testRowIndexMaxSparseMatrixNegMR() 
	{
		runRowAggregateOperationTest(OpType.ROW_INDEXMAX, true, false, ExecType.MR, true);
	}
	
	
	@Test
	public void testRowIndexMaxSparseVectorNegMR() 
	{
		runRowAggregateOperationTest(OpType.ROW_INDEXMAX, true, true, ExecType.MR, true);
	}
	//----
	
	@Test
	public void testRowIndexMinDenseMatrixNegCP() 
	{
		runRowAggregateOperationTest(OpType.ROW_INDEXMIN, false, false, ExecType.CP, true);
	}
	
	@Test
	public void testRowIndexMinDenseVectorNegCP() 
	{
		runRowAggregateOperationTest(OpType.ROW_INDEXMIN, false, true, ExecType.CP, true);
	}
	
	@Test
	public void testRowIndexMinSparseMatrixNegCP() 
	{
		runRowAggregateOperationTest(OpType.ROW_INDEXMIN, true, false, ExecType.CP, true);
	}
	
	@Test
	public void testRowIndexMinSparseVectorNegCP() 
	{
		runRowAggregateOperationTest(OpType.ROW_INDEXMIN, true, true, ExecType.CP, true);
	}
	
	@Test
	public void testRowIndexMinDenseMatrixNegMR() 
	{
		runRowAggregateOperationTest(OpType.ROW_INDEXMIN, false, false, ExecType.MR, true);
	}
	
	@Test
	public void testRowIndexMinDenseVectorNegMR() 
	{
		runRowAggregateOperationTest(OpType.ROW_INDEXMIN, false, true, ExecType.MR, true);
	}
	
	
	@Test
	public void testRowIndexMinSparseMatrixNegMR() 
	{
		runRowAggregateOperationTest(OpType.ROW_INDEXMIN, true, false, ExecType.MR, true);
	}
	
	
	@Test
	public void testRowIndexMinSparseVectorNegMR() 
	{
		runRowAggregateOperationTest(OpType.ROW_INDEXMIN, true, true, ExecType.MR, true);
	}
	
	/**
	 * 
	 * @param type
	 * @param sparse
	 * @param vector
	 * @param instType
	 */
	private void runRowAggregateOperationTest( OpType type, boolean sparse, boolean vector, ExecType instType)
	{
		runRowAggregateOperationTest(type, sparse, vector, instType, false);
	}
	
	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
	private void runRowAggregateOperationTest( OpType type, boolean sparse, boolean vector, ExecType instType, boolean specialData)
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
				case ROW_INDEXMAX: TEST_NAME = TEST_NAME5; break;
				case ROW_INDEXMIN: TEST_NAME = TEST_NAME6; break;
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
			double min, max;
			
			// In case of ROW_INDEXMAX, generate all negative data 
			// so that the value 0 is the maximum value. Similarly,
			// in case of ROW_INDEXMIN, generate all positive data.
			if ( type == OpType.ROW_INDEXMAX ) {
				min = specialData ? -1 : -0.05;
				max = specialData ? -0.05 : 1;
			}
			else if (type == OpType.ROW_INDEXMIN ){
				min = specialData ? 0 : 0.05;
				max = specialData ? 0.05 : 0;
			} else {
				min = -0.05;
				max = 1;
			}
					
			double[][] A = getRandomMatrix(rows, cols, min, max, sparsity, 7); 
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