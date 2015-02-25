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
import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

/**
 * NOTES:
 *  * the output definition of DML and R differs for col*; R always returns a column vector
 *    while DML returns a row vector.
 *  * the R package Matrix does not support colMins and colMaxs; hence, we use the matrixStats package 
 * 
 */
public class FullColAggregateTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME1 = "ColSums";
	private final static String TEST_NAME2 = "ColMeans";
	private final static String TEST_NAME3 = "ColMaxs";
	private final static String TEST_NAME4 = "ColMins";
	
	private final static String TEST_DIR = "functions/aggregate/";
	private final static double eps = 1e-10;
	
	private final static int rows = 1005;
	private final static int cols1 = 1;
	private final static int cols2 = 1079;
	private final static double sparsity1 = 0.1;
	private final static double sparsity2 = 0.7;
	
	private enum OpType{
		COL_SUMS,
		COL_MEANS,
		COL_MAX,
		COL_MIN
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
	public void testColSumsDenseMatrixCP() 
	{
		runColAggregateOperationTest(OpType.COL_SUMS, false, false, ExecType.CP);
	}
	
	@Test
	public void testColMeansDenseMatrixCP() 
	{
		runColAggregateOperationTest(OpType.COL_MEANS, false, false, ExecType.CP);
	}	
	
	@Test
	public void testColMaxDenseMatrixCP() 
	{
		runColAggregateOperationTest(OpType.COL_MAX, false, false, ExecType.CP);
	}
	
	@Test
	public void testColMinDenseMatrixCP() 
	{
		runColAggregateOperationTest(OpType.COL_MIN, false, false, ExecType.CP);
	}
	
	@Test
	public void testColSumsDenseVectorCP() 
	{
		runColAggregateOperationTest(OpType.COL_SUMS, false, true, ExecType.CP);
	}
	
	@Test
	public void testColMeansDenseVectorCP() 
	{
		runColAggregateOperationTest(OpType.COL_MEANS, false, true, ExecType.CP);
	}	
	
	@Test
	public void testColMaxDenseVectorCP() 
	{
		runColAggregateOperationTest(OpType.COL_MAX, false, true, ExecType.CP);
	}
	
	@Test
	public void testColMinDenseVectorCP() 
	{
		runColAggregateOperationTest(OpType.COL_MIN, false, true, ExecType.CP);
	}
	
	@Test
	public void testColSumsSparseMatrixCP() 
	{
		runColAggregateOperationTest(OpType.COL_SUMS, true, false, ExecType.CP);
	}
	
	@Test
	public void testColMeansSparseMatrixCP() 
	{
		runColAggregateOperationTest(OpType.COL_MEANS, true, false, ExecType.CP);
	}	
	
	@Test
	public void testColMaxSparseMatrixCP() 
	{
		runColAggregateOperationTest(OpType.COL_MAX, true, false, ExecType.CP);
	}
	
	@Test
	public void testColMinSparseMatrixCP() 
	{
		runColAggregateOperationTest(OpType.COL_MIN, true, false, ExecType.CP);
	}
	
	@Test
	public void testColSumsSparseVectorCP() 
	{
		runColAggregateOperationTest(OpType.COL_SUMS, true, true, ExecType.CP);
	}
	
	@Test
	public void testColMeansSparseVectorCP() 
	{
		runColAggregateOperationTest(OpType.COL_MEANS, true, true, ExecType.CP);
	}	
	
	@Test
	public void testColMaxSparseVectorCP() 
	{
		runColAggregateOperationTest(OpType.COL_MAX, true, true, ExecType.CP);
	}
	
	@Test
	public void testColMinSparseVectorCP() 
	{
		runColAggregateOperationTest(OpType.COL_MIN, true, true, ExecType.CP);
	}
	
	@Test
	public void testColSumsDenseMatrixMR() 
	{
		runColAggregateOperationTest(OpType.COL_SUMS, false, false, ExecType.MR);
	}
	
	@Test
	public void testColMeansDenseMatrixMR() 
	{
		runColAggregateOperationTest(OpType.COL_MEANS, false, false, ExecType.MR);
	}	
	
	@Test
	public void testColMaxDenseMatrixMR() 
	{
		runColAggregateOperationTest(OpType.COL_MAX, false, false, ExecType.MR);
	}
	
	@Test
	public void testColMinDenseMatrixMR() 
	{
		runColAggregateOperationTest(OpType.COL_MIN, false, false, ExecType.MR);
	}
	
	@Test
	public void testColSumsDenseVectorMR() 
	{
		runColAggregateOperationTest(OpType.COL_SUMS, false, true, ExecType.MR);
	}
	
	@Test
	public void testColMeansDenseVectorMR() 
	{
		runColAggregateOperationTest(OpType.COL_MEANS, false, true, ExecType.MR);
	}	
	
	@Test
	public void testColMaxDenseVectorMR() 
	{
		runColAggregateOperationTest(OpType.COL_MAX, false, true, ExecType.MR);
	}
	
	@Test
	public void testColMinDenseVectorMR() 
	{
		runColAggregateOperationTest(OpType.COL_MIN, false, true, ExecType.MR);
	}
	
	@Test
	public void testColSumsSparseMatrixMR() 
	{
		runColAggregateOperationTest(OpType.COL_SUMS, true, false, ExecType.MR);
	}
	
	@Test
	public void testColMeansSparseMatrixMR() 
	{
		runColAggregateOperationTest(OpType.COL_MEANS, true, false, ExecType.MR);
	}	
	
	@Test
	public void testColMaxSparseMatrixMR() 
	{
		runColAggregateOperationTest(OpType.COL_MAX, true, false, ExecType.MR);
	}
	
	@Test
	public void testColMinSparseMatrixMR() 
	{
		runColAggregateOperationTest(OpType.COL_MIN, true, false, ExecType.MR);
	}
	
	@Test
	public void testColSumsSparseVectorMR() 
	{
		runColAggregateOperationTest(OpType.COL_SUMS, true, true, ExecType.MR);
	}
	
	@Test
	public void testColMeansSparseVectorMR() 
	{
		runColAggregateOperationTest(OpType.COL_MEANS, true, true, ExecType.MR);
	}	
	
	@Test
	public void testColMaxSparseVectorMR() 
	{
		runColAggregateOperationTest(OpType.COL_MAX, true, true, ExecType.MR);
	}
	
	@Test
	public void testColMinSparseVectorMR() 
	{
		runColAggregateOperationTest(OpType.COL_MIN, true, true, ExecType.MR);
	}

	@Test
	public void testColSumsDenseMatrixNoRewritesCP() 
	{
		runColAggregateOperationTest(OpType.COL_SUMS, false, false, ExecType.CP, false);
	}
	
	@Test
	public void testColMeansDenseMatrixNoRewritesCP() 
	{
		runColAggregateOperationTest(OpType.COL_MEANS, false, false, ExecType.CP, false);
	}	
	
	@Test
	public void testColMaxDenseMatrixNoRewritesCP() 
	{
		runColAggregateOperationTest(OpType.COL_MAX, false, false, ExecType.CP, false);
	}
	
	@Test
	public void testColMinDenseMatrixNoRewritesCP() 
	{
		runColAggregateOperationTest(OpType.COL_MIN, false, false, ExecType.CP, false);
	}
	
	@Test
	public void testColSumsDenseVectorNoRewritesCP() 
	{
		runColAggregateOperationTest(OpType.COL_SUMS, false, true, ExecType.CP, false);
	}
	
	@Test
	public void testColMeansDenseVectorNoRewritesCP() 
	{
		runColAggregateOperationTest(OpType.COL_MEANS, false, true, ExecType.CP, false);
	}	
	
	@Test
	public void testColMaxDenseVectorNoRewritesCP() 
	{
		runColAggregateOperationTest(OpType.COL_MAX, false, true, ExecType.CP, false);
	}
	
	@Test
	public void testColMinDenseVectorNoRewritesCP() 
	{
		runColAggregateOperationTest(OpType.COL_MIN, false, true, ExecType.CP, false);
	}
	
	@Test
	public void testColSumsSparseMatrixNoRewritesCP() 
	{
		runColAggregateOperationTest(OpType.COL_SUMS, true, false, ExecType.CP, false);
	}
	
	@Test
	public void testColMeansSparseMatrixNoRewritesCP() 
	{
		runColAggregateOperationTest(OpType.COL_MEANS, true, false, ExecType.CP, false);
	}	
	
	@Test
	public void testColMaxSparseMatrixNoRewritesCP() 
	{
		runColAggregateOperationTest(OpType.COL_MAX, true, false, ExecType.CP, false);
	}
	
	@Test
	public void testColMinSparseMatrixNoRewritesCP() 
	{
		runColAggregateOperationTest(OpType.COL_MIN, true, false, ExecType.CP, false);
	}
	
	@Test
	public void testColSumsSparseVectorNoRewritesCP() 
	{
		runColAggregateOperationTest(OpType.COL_SUMS, true, true, ExecType.CP, false);
	}
	
	@Test
	public void testColMeansSparseVectorNoRewritesCP() 
	{
		runColAggregateOperationTest(OpType.COL_MEANS, true, true, ExecType.CP, false);
	}	
	
	@Test
	public void testColMaxSparseVectorNoRewritesCP() 
	{
		runColAggregateOperationTest(OpType.COL_MAX, true, true, ExecType.CP, false);
	}
	
	@Test
	public void testColMinSparseVectorNoRewritesCP() 
	{
		runColAggregateOperationTest(OpType.COL_MIN, true, true, ExecType.CP, false);
	}
	
	@Test
	public void testColSumsDenseMatrixNoRewritesMR() 
	{
		runColAggregateOperationTest(OpType.COL_SUMS, false, false, ExecType.MR, false);
	}
	
	@Test
	public void testColMeansDenseMatrixNoRewritesMR() 
	{
		runColAggregateOperationTest(OpType.COL_MEANS, false, false, ExecType.MR, false);
	}	
	
	@Test
	public void testColMaxDenseMatrixNoRewritesMR() 
	{
		runColAggregateOperationTest(OpType.COL_MAX, false, false, ExecType.MR, false);
	}
	
	@Test
	public void testColMinDenseMatrixNoRewritesMR() 
	{
		runColAggregateOperationTest(OpType.COL_MIN, false, false, ExecType.MR, false);
	}
	
	@Test
	public void testColSumsDenseVectorNoRewritesMR() 
	{
		runColAggregateOperationTest(OpType.COL_SUMS, false, true, ExecType.MR, false);
	}
	
	@Test
	public void testColMeansDenseVectorNoRewritesMR() 
	{
		runColAggregateOperationTest(OpType.COL_MEANS, false, true, ExecType.MR, false);
	}	
	
	@Test
	public void testColMaxDenseVectorNoRewritesMR() 
	{
		runColAggregateOperationTest(OpType.COL_MAX, false, true, ExecType.MR, false);
	}
	
	@Test
	public void testColMinDenseVectorNoRewritesMR() 
	{
		runColAggregateOperationTest(OpType.COL_MIN, false, true, ExecType.MR, false);
	}
	
	@Test
	public void testColSumsSparseMatrixNoRewritesMR() 
	{
		runColAggregateOperationTest(OpType.COL_SUMS, true, false, ExecType.MR, false);
	}
	
	@Test
	public void testColMeansSparseMatrixNoRewritesMR() 
	{
		runColAggregateOperationTest(OpType.COL_MEANS, true, false, ExecType.MR, false);
	}	
	
	@Test
	public void testColMaxSparseMatrixNoRewritesMR() 
	{
		runColAggregateOperationTest(OpType.COL_MAX, true, false, ExecType.MR, false);
	}
	
	@Test
	public void testColMinSparseMatrixNoRewritesMR() 
	{
		runColAggregateOperationTest(OpType.COL_MIN, true, false, ExecType.MR, false);
	}
	
	@Test
	public void testColSumsSparseVectorNoRewritesMR() 
	{
		runColAggregateOperationTest(OpType.COL_SUMS, true, true, ExecType.MR, false);
	}
	
	@Test
	public void testColMeansSparseVectorNoRewritesMR() 
	{
		runColAggregateOperationTest(OpType.COL_MEANS, true, true, ExecType.MR, false);
	}	
	
	@Test
	public void testColMaxSparseVectorNoRewritesMR() 
	{
		runColAggregateOperationTest(OpType.COL_MAX, true, true, ExecType.MR, false);
	}
	
	@Test
	public void testColMinSparseVectorNoRewritesMR() 
	{
		runColAggregateOperationTest(OpType.COL_MIN, true, true, ExecType.MR, false);
	}
	
	/**
	 * 
	 * @param type
	 * @param sparse
	 * @param vector
	 * @param instType
	 */
	private void runColAggregateOperationTest( OpType type, boolean sparse, boolean vector, ExecType instType)
	{
		runColAggregateOperationTest(type, sparse, vector, instType, true); //by default apply algebraic simplification
	}
	
	
	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
	private void runColAggregateOperationTest( OpType type, boolean sparse, boolean vector, ExecType instType, boolean rewrites)
	{
		//rtplatform for MR
		RUNTIME_PLATFORM platformOld = rtplatform;
		rtplatform = (instType==ExecType.MR) ? RUNTIME_PLATFORM.HADOOP : RUNTIME_PLATFORM.HYBRID;
	
		boolean oldRewritesFlag = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
		
		try
		{
			String TEST_NAME = null;
			switch( type )
			{
				case COL_SUMS: TEST_NAME = TEST_NAME1; break;
				case COL_MEANS: TEST_NAME = TEST_NAME2; break;
				case COL_MAX: TEST_NAME = TEST_NAME3; break;
				case COL_MIN: TEST_NAME = TEST_NAME4; break;
			}
			
			int cols = (vector) ? cols1 : cols2;
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
			double[][] A = getRandomMatrix(rows, cols, -0.05, 1, sparsity, 7); 
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
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldRewritesFlag;
		}
	}
	
		
}