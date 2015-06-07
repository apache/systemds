/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.aggregate;

import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;
import com.ibm.bi.dml.utils.Statistics;

/**
 * 
 * 
 */
public class FullAggregateTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME1 = "AllSum";
	private final static String TEST_NAME2 = "AllMean";
	private final static String TEST_NAME3 = "AllMax";
	private final static String TEST_NAME4 = "AllMin";
	private final static String TEST_NAME5 = "AllProd";
	private final static String TEST_NAME6 = "DiagSum"; //trace
	

	private final static String TEST_DIR = "functions/aggregate/";
	private final static double eps = 1e-10;
	
	private final static int rows1 = 1005;
	private final static int cols1 = 1;
	private final static int cols2 = 1079;
	private final static double sparsity1 = 0.1;
	private final static double sparsity2 = 0.7;
	
	private enum OpType{
		SUM,
		MEAN,
		MAX,
		MIN,
		PROD,
		TRACE
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
	public void testSumDenseMatrixCP() 
	{
		runColAggregateOperationTest(OpType.SUM, false, false, ExecType.CP);
	}
	
	@Test
	public void testMeanDenseMatrixCP() 
	{
		runColAggregateOperationTest(OpType.MEAN, false, false, ExecType.CP);
	}	
	
	@Test
	public void testMaxDenseMatrixCP() 
	{
		runColAggregateOperationTest(OpType.MAX, false, false, ExecType.CP);
	}
	
	@Test
	public void testMinDenseMatrixCP() 
	{
		runColAggregateOperationTest(OpType.MIN, false, false, ExecType.CP);
	}
	
	@Test
	public void testProdDenseMatrixCP() 
	{
		runColAggregateOperationTest(OpType.PROD, false, false, ExecType.CP);
	}
	
	@Test
	public void testTraceDenseMatrixCP() 
	{
		runColAggregateOperationTest(OpType.TRACE, false, false, ExecType.CP);
	}
	
	@Test
	public void testSumDenseVectorCP() 
	{
		runColAggregateOperationTest(OpType.SUM, false, true, ExecType.CP);
	}
	
	@Test
	public void testMeanDenseVectorCP() 
	{
		runColAggregateOperationTest(OpType.MEAN, false, true, ExecType.CP);
	}	
	
	@Test
	public void testMaxDenseVectorCP() 
	{
		runColAggregateOperationTest(OpType.MAX, false, true, ExecType.CP);
	}
	
	@Test
	public void testMinDenseVectorCP() 
	{
		runColAggregateOperationTest(OpType.MIN, false, true, ExecType.CP);
	}
	
	@Test
	public void testProdDenseVectorCP() 
	{
		runColAggregateOperationTest(OpType.PROD, false, true, ExecType.CP);
	}
	
	@Test
	public void testTraceDenseVectorCP() 
	{
		runColAggregateOperationTest(OpType.TRACE, false, true, ExecType.CP);
	}
	
	@Test
	public void testSumSparseMatrixCP() 
	{
		runColAggregateOperationTest(OpType.SUM, true, false, ExecType.CP);
	}
	
	@Test
	public void testMeanSparseMatrixCP() 
	{
		runColAggregateOperationTest(OpType.MEAN, true, false, ExecType.CP);
	}	
	
	@Test
	public void testMaxSparseMatrixCP() 
	{
		runColAggregateOperationTest(OpType.MAX, true, false, ExecType.CP);
	}
	
	@Test
	public void testMinSparseMatrixCP() 
	{
		runColAggregateOperationTest(OpType.MIN, true, false, ExecType.CP);
	}
	
	@Test
	public void testProdSparseMatrixCP() 
	{
		runColAggregateOperationTest(OpType.PROD, true, false, ExecType.CP);
	}
	
	@Test
	public void testTraceSparseMatrixCP() 
	{
		runColAggregateOperationTest(OpType.TRACE, true, false, ExecType.CP);
	}
	
	@Test
	public void testSumSparseVectorCP() 
	{
		runColAggregateOperationTest(OpType.SUM, true, true, ExecType.CP);
	}
	
	@Test
	public void testMeanSparseVectorCP() 
	{
		runColAggregateOperationTest(OpType.MEAN, true, true, ExecType.CP);
	}	
	
	@Test
	public void testMaxSparseVectorCP() 
	{
		runColAggregateOperationTest(OpType.MAX, true, true, ExecType.CP);
	}
	
	@Test
	public void testMinSparseVectorCP() 
	{
		runColAggregateOperationTest(OpType.MIN, true, true, ExecType.CP);
	}
	
	@Test
	public void testProdSparseVectorCP() 
	{
		runColAggregateOperationTest(OpType.PROD, true, true, ExecType.CP);
	}
	
	@Test
	public void testTraceSparseVectorCP() 
	{
		runColAggregateOperationTest(OpType.TRACE, true, true, ExecType.CP);
	}
	
	@Test
	public void testSumDenseMatrixMR() 
	{
		runColAggregateOperationTest(OpType.SUM, false, false, ExecType.MR);
	}
	
	@Test
	public void testMeanDenseMatrixMR() 
	{
		runColAggregateOperationTest(OpType.MEAN, false, false, ExecType.MR);
	}	
	
	@Test
	public void testMaxDenseMatrixMR() 
	{
		runColAggregateOperationTest(OpType.MAX, false, false, ExecType.MR);
	}
	
	@Test
	public void testMinDenseMatrixMR() 
	{
		runColAggregateOperationTest(OpType.MIN, false, false, ExecType.MR);
	}
	
	@Test
	public void testProdDenseMatrixMR() 
	{
		runColAggregateOperationTest(OpType.PROD, false, false, ExecType.MR);
	}
	
	@Test
	public void testTraceDenseMatrixMR() 
	{
		runColAggregateOperationTest(OpType.TRACE, false, false, ExecType.MR);
	}
	
	@Test
	public void testSumDenseVectorMR() 
	{
		runColAggregateOperationTest(OpType.SUM, false, true, ExecType.MR);
	}
	
	@Test
	public void testMeanDenseVectorMR() 
	{
		runColAggregateOperationTest(OpType.MEAN, false, true, ExecType.MR);
	}	
	
	@Test
	public void testMaxDenseVectorMR() 
	{
		runColAggregateOperationTest(OpType.MAX, false, true, ExecType.MR);
	}
	
	@Test
	public void testMinDenseVectorMR() 
	{
		runColAggregateOperationTest(OpType.MIN, false, true, ExecType.MR);
	}
	
	@Test
	public void testProdDenseVectorMR() 
	{
		runColAggregateOperationTest(OpType.PROD, false, true, ExecType.MR);
	}
	
	@Test
	public void testTraceDenseVectorMR() 
	{
		runColAggregateOperationTest(OpType.TRACE, false, true, ExecType.MR);
	}
	
	@Test
	public void testSumSparseMatrixMR() 
	{
		runColAggregateOperationTest(OpType.SUM, true, false, ExecType.MR);
	}
	
	@Test
	public void testMeanSparseMatrixMR() 
	{
		runColAggregateOperationTest(OpType.MEAN, true, false, ExecType.MR);
	}	
	
	@Test
	public void testMaxSparseMatrixMR() 
	{
		runColAggregateOperationTest(OpType.MAX, true, false, ExecType.MR);
	}
	
	@Test
	public void testMinSparseMatrixMR() 
	{
		runColAggregateOperationTest(OpType.MIN, true, false, ExecType.MR);
	}
	
	@Test
	public void testProdSparseMatrixMR() 
	{
		runColAggregateOperationTest(OpType.PROD, true, false, ExecType.MR);
	}
	
	@Test
	public void testTraceSparseMatrixMR() 
	{
		runColAggregateOperationTest(OpType.TRACE, true, false, ExecType.MR);
	}
	
	@Test
	public void testSumSparseVectorMR() 
	{
		runColAggregateOperationTest(OpType.SUM, true, true, ExecType.MR);
	}
	
	@Test
	public void testMeanSparseVectorMR() 
	{
		runColAggregateOperationTest(OpType.MEAN, true, true, ExecType.MR);
	}	
	
	@Test
	public void testMaxSparseVectorMR() 
	{
		runColAggregateOperationTest(OpType.MAX, true, true, ExecType.MR);
	}
	
	@Test
	public void testMinSparseVectorMR() 
	{
		runColAggregateOperationTest(OpType.MIN, true, true, ExecType.MR);
	}
	
	@Test
	public void testProdSparseVectorMR() 
	{
		runColAggregateOperationTest(OpType.PROD, true, true, ExecType.MR);
	}
	
	@Test
	public void testTraceSparseVectorMR() 
	{
		runColAggregateOperationTest(OpType.TRACE, true, true, ExecType.MR);
	}
	
	
	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 */
	private void runColAggregateOperationTest( OpType type, boolean sparse, boolean vector, ExecType instType)
	{
		//rtplatform for MR
		RUNTIME_PLATFORM platformOld = rtplatform;
		rtplatform = (instType==ExecType.MR) ? RUNTIME_PLATFORM.HADOOP : RUNTIME_PLATFORM.HYBRID;
	
		try
		{
			String TEST_NAME = null;
			switch( type )
			{
				case SUM: TEST_NAME = TEST_NAME1; break;
				case MEAN: TEST_NAME = TEST_NAME2; break;
				case MAX: TEST_NAME = TEST_NAME3; break;
				case MIN: TEST_NAME = TEST_NAME4; break;
				case PROD: TEST_NAME = TEST_NAME5; break;
				case TRACE: TEST_NAME = TEST_NAME6; break;
			}
			
			int cols = (vector) ? cols1 : cols2;
			int rows = (type==OpType.TRACE) ? cols : rows1;
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
	
			runTest(true, false, null, -1); 
			if( instType==ExecType.CP ) //in CP no MR jobs should be executed
				Assert.assertEquals("Unexpected number of executed MR jobs.", 0, Statistics.getNoOfExecutedMRJobs());
		
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