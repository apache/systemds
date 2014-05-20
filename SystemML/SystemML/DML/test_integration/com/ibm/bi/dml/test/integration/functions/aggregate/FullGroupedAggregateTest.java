/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.aggregate;

import java.io.IOException;
import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue.CellIndex;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

/**
 * 
 */
public class FullGroupedAggregateTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME1 = "GroupedAggregate";
	private final static String TEST_NAME2 = "GroupedAggregateWeights";
	
	private final static String TEST_DIR = "functions/aggregate/";
	private final static double eps = 1e-10;
	
	private final static int rows = 17654;
	private final static int cols = 1;
	private final static double sparsity1 = 0.1;
	private final static double sparsity2 = 0.7;
	
	private final static int numGroups = 7;
	private final static int maxWeight = 10;
	
	private enum OpType{
		SUM,
		COUNT,
		MEAN,
		VARIANCE,
		MOMENT3,
		MOMENT4,
	}
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME1, new TestConfiguration(TEST_DIR, TEST_NAME1, new String[]{"C"})); 
		addTestConfiguration(TEST_NAME2, new TestConfiguration(TEST_DIR, TEST_NAME2, new String[]{"D"})); 
		TestUtils.clearAssertionInformation();
	}

	
	@Test
	public void testGroupedAggSumDenseCP() 
	{
		runGroupedAggregateOperationTest(OpType.SUM, false, false, false, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggSumSparseCP() 
	{
		runGroupedAggregateOperationTest(OpType.SUM, true, false, false, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggSumDenseWeightsCP() 
	{
		runGroupedAggregateOperationTest(OpType.SUM, false, true, false, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggSumSparseWeightsCP() 
	{
		runGroupedAggregateOperationTest(OpType.SUM, true, true, false, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggSumDenseTransposeCP() 
	{
		runGroupedAggregateOperationTest(OpType.SUM, false, false, true, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggSumSparseTransposeCP() 
	{
		runGroupedAggregateOperationTest(OpType.SUM, true, false, true, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggSumDenseWeightsTransposeCP() 
	{
		runGroupedAggregateOperationTest(OpType.SUM, false, true, true, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggSumSparseWeightsTransposeCP() 
	{
		runGroupedAggregateOperationTest(OpType.SUM, true, true, true, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggCountDenseCP() 
	{
		runGroupedAggregateOperationTest(OpType.COUNT, false, false, false, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggCountSparseCP() 
	{
		runGroupedAggregateOperationTest(OpType.COUNT, true, false, false, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggCountDenseWeightsCP() 
	{
		runGroupedAggregateOperationTest(OpType.COUNT, false, true, false, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggCountSparseWeightsCP() 
	{
		runGroupedAggregateOperationTest(OpType.COUNT, true, true, false, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggMeanDenseCP() 
	{
		runGroupedAggregateOperationTest(OpType.MEAN, false, false, false, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggMeanSparseCP() 
	{
		runGroupedAggregateOperationTest(OpType.MEAN, true, false, false, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggMeanDenseWeightsCP() 
	{
		runGroupedAggregateOperationTest(OpType.MEAN, false, true, false, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggMeanSparseWeightsCP() 
	{
		runGroupedAggregateOperationTest(OpType.MEAN, true, true, false, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggVarianceDenseCP() 
	{
		runGroupedAggregateOperationTest(OpType.VARIANCE, false, false, false, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggVarianceSparseCP() 
	{
		runGroupedAggregateOperationTest(OpType.VARIANCE, true, false, false, ExecType.CP);
	}
	
	/* TODO weighted variance in R
	@Test
	public void testGroupedAggVarianceDenseWeightsCP() 
	{
		runGroupedAggregateOperationTest(OpType.VARIANCE, false, true, false, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggVarianceSparseWeightsCP() 
	{
		runGroupedAggregateOperationTest(OpType.VARIANCE, true, true, false, ExecType.CP);
	}
	*/
	
	@Test
	public void testGroupedAggMoment3DenseCP() 
	{
		runGroupedAggregateOperationTest(OpType.MOMENT3, false, false, false, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggMoment3SparseCP() 
	{
		runGroupedAggregateOperationTest(OpType.MOMENT3, true, false, false, ExecType.CP);
	}
	
	/* TODO weighted central moment in R
	@Test
	public void testGroupedAggMoment3DenseWeightsCP() 
	{
		runGroupedAggregateOperationTest(OpType.MOMENT3, false, true, false, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggMoment3SparseWeightsCP() 
	{
		runGroupedAggregateOperationTest(OpType.MOMENT3, true, true, false, ExecType.CP);
	}
	*/
	
	@Test
	public void testGroupedAggMoment4DenseCP() 
	{
		runGroupedAggregateOperationTest(OpType.MOMENT4, false, false, false, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggMoment4SparseCP() 
	{
		runGroupedAggregateOperationTest(OpType.MOMENT4, true, false, false, ExecType.CP);
	}
	
	/* TODO weighted central moment in R
	@Test
	public void testGroupedAggMoment4DenseWeightsCP() 
	{
		runGroupedAggregateOperationTest(OpType.MOMENT4, false, true, false, ExecType.CP);
	}
	
	@Test
	public void testGroupedAggMoment4SparseWeightsCP() 
	{
		runGroupedAggregateOperationTest(OpType.MOMENT4, true, true, false, ExecType.CP);
	}
	*/
	
	@Test
	public void testGroupedAggSumDenseMR() 
	{
		runGroupedAggregateOperationTest(OpType.SUM, false, false, false, ExecType.MR);
	}
	
	@Test
	public void testGroupedAggSumSparseMR() 
	{
		runGroupedAggregateOperationTest(OpType.SUM, true, false, false, ExecType.MR);
	}
	
	@Test
	public void testGroupedAggSumDenseWeightsMR() 
	{
		runGroupedAggregateOperationTest(OpType.SUM, false, true, false, ExecType.MR);
	}
	
	@Test
	public void testGroupedAggSumSparseWeightsMR() 
	{
		runGroupedAggregateOperationTest(OpType.SUM, true, true, false, ExecType.MR);
	}
	
	@Test
	public void testGroupedAggCountDenseMR() 
	{
		runGroupedAggregateOperationTest(OpType.COUNT, false, false, false, ExecType.MR);
	}
	
	@Test
	public void testGroupedAggCountSparseMR() 
	{
		runGroupedAggregateOperationTest(OpType.COUNT, true, false, false, ExecType.MR);
	}
	
	@Test
	public void testGroupedAggCountDenseWeightsMR() 
	{
		runGroupedAggregateOperationTest(OpType.COUNT, false, true, false, ExecType.MR);
	}
	
	@Test
	public void testGroupedAggCountSparseWeightsMR() 
	{
		runGroupedAggregateOperationTest(OpType.COUNT, true, true, false, ExecType.MR);
	}
	
	@Test
	public void testGroupedAggMeanDenseMR() 
	{
		runGroupedAggregateOperationTest(OpType.MEAN, false, false, false, ExecType.MR);
	}
	
	@Test
	public void testGroupedAggMeanSparseMR() 
	{
		runGroupedAggregateOperationTest(OpType.MEAN, true, false, false, ExecType.MR);
	}
	
	@Test
	public void testGroupedAggMeanDenseWeightsMR() 
	{
		runGroupedAggregateOperationTest(OpType.MEAN, false, true, false, ExecType.MR);
	}
	
	@Test
	public void testGroupedAggMeanSparseWeightsMR() 
	{
		runGroupedAggregateOperationTest(OpType.MEAN, true, true, false, ExecType.MR);
	}
	
	@Test
	public void testGroupedAggVarianceDenseMR() 
	{
		runGroupedAggregateOperationTest(OpType.VARIANCE, false, false, false, ExecType.MR);
	}
	
	@Test
	public void testGroupedAggVarianceSparseMR() 
	{
		runGroupedAggregateOperationTest(OpType.VARIANCE, true, false, false, ExecType.MR);
	}
	
	/* TODO weighted variance in R
	@Test
	public void testGroupedAggVarianceDenseWeightsMR() 
	{
		runGroupedAggregateOperationTest(OpType.VARIANCE, false, true, false, ExecType.MR);
	}
	
	@Test
	public void testGroupedAggVarianceSparseWeightsMR() 
	{
		runGroupedAggregateOperationTest(OpType.VARIANCE, true, true, false, ExecType.MR);
	}
	*/
	
	@Test
	public void testGroupedAggMoment3DenseMR() 
	{
		runGroupedAggregateOperationTest(OpType.MOMENT3, false, false, false, ExecType.MR);
	}
	
	@Test
	public void testGroupedAggMoment3SparseMR() 
	{
		runGroupedAggregateOperationTest(OpType.MOMENT3, true, false, false, ExecType.MR);
	}
	
	/* TODO weighted central moment in R
	@Test
	public void testGroupedAggMoment3DenseWeightsMR() 
	{
		runGroupedAggregateOperationTest(OpType.MOMENT3, false, true, false, ExecType.MR);
	}
	
	@Test
	public void testGroupedAggMoment3SparseWeightsMR() 
	{
		runGroupedAggregateOperationTest(OpType.MOMENT3, true, true, false, ExecType.MR);
	}
	*/
	
	@Test
	public void testGroupedAggMoment4DenseMR() 
	{
		runGroupedAggregateOperationTest(OpType.MOMENT4, false, false, false, ExecType.MR);
	}
	
	@Test
	public void testGroupedAggMoment4SparseMR() 
	{
		runGroupedAggregateOperationTest(OpType.MOMENT4, true, false, false, ExecType.MR);
	}
	
	/* TODO weighted central moment in R
	@Test
	public void testGroupedAggMoment4DenseWeightsMR() 
	{
		runGroupedAggregateOperationTest(OpType.MOMENT4, false, true, false, ExecType.MR);
	}
	
	@Test
	public void testGroupedAggMoment4SparseWeightsMR() 
	{
		runGroupedAggregateOperationTest(OpType.MOMENT4, true, true, false, ExecType.MR);
	}
	*/
	
	/**
	 * 
	 * @param sparseM1
	 * @param sparseM2
	 * @param instType
	 * @throws IOException 
	 */
	private void runGroupedAggregateOperationTest( OpType type, boolean sparse, boolean weights, boolean transpose, ExecType instType) 
	{
		//rtplatform for MR
		RUNTIME_PLATFORM platformOld = rtplatform;
		rtplatform = (instType==ExecType.MR) ? RUNTIME_PLATFORM.HADOOP : RUNTIME_PLATFORM.HYBRID;
	
		try
		{
			//determine script and function name
			String TEST_NAME = weights ? TEST_NAME2 : TEST_NAME1;
			int fn = type.ordinal();
			
			double sparsity = (sparse) ? sparsity1 : sparsity2;
			
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			// This is for running the junit test the new way, i.e., construct the arguments directly
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			if( !weights ){
				programArgs = new String[]{"-explain","-args", HOME + INPUT_DIR + "A",
												HOME + INPUT_DIR + "B", 
												Integer.toString(fn),
						                        HOME + OUTPUT_DIR + "C"    };
			}
			else{
				programArgs = new String[]{"-args", HOME + INPUT_DIR + "A",
												HOME + INPUT_DIR + "B",
												HOME + INPUT_DIR + "C",
												Integer.toString(fn),
						                        HOME + OUTPUT_DIR + "D"    };
			}
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       HOME + INPUT_DIR + " " + fn + " " + HOME + EXPECTED_DIR;
			
			loadTestConfiguration(config);
	
			//generate actual dataset 
			double[][] A = getRandomMatrix(transpose?cols:rows, transpose?rows:cols, -0.05, 1, sparsity, 7); 
			writeInputMatrix("A", A, true);
			MatrixCharacteristics mc1 = new MatrixCharacteristics(transpose?cols:rows, transpose?rows:cols,1000,1000);
			MapReduceTool.writeMetaDataFile(HOME + INPUT_DIR + "A.mtd", ValueType.DOUBLE, mc1, OutputInfo.TextCellOutputInfo);
			double[][] B = TestUtils.round(getRandomMatrix(rows, cols, 1, numGroups, 1.0, 3)); 
			writeInputMatrix("B", B, true);
			MatrixCharacteristics mc2 = new MatrixCharacteristics(rows,cols,1000,1000);
			MapReduceTool.writeMetaDataFile(HOME + INPUT_DIR + "B.mtd", ValueType.DOUBLE, mc2, OutputInfo.TextCellOutputInfo);
			if( weights ){
				//currently we use integer weights due to our definition of weight as multiplicity
				double[][] C = TestUtils.round(getRandomMatrix(rows, cols, 1, maxWeight, 1.0, 3)); 
				writeInputMatrix("C", C, true);
				MatrixCharacteristics mc3 = new MatrixCharacteristics(rows,cols,1000,1000);
				MapReduceTool.writeMetaDataFile(HOME + INPUT_DIR + "C.mtd", ValueType.DOUBLE, mc3, OutputInfo.TextCellOutputInfo);	
			}
			
			//run tests
			runTest(true, false, null, -1); 
			runRScript(true); 
		
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS(weights?"D":"C");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS(weights?"D":"C");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
		}
		catch(IOException ex)
		{
			ex.printStackTrace();
			throw new RuntimeException(ex);
		}
		finally
		{
			rtplatform = platformOld;
		}
	}
	
		
}