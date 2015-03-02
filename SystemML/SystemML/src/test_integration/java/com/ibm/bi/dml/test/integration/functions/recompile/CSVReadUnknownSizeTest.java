/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.recompile;

import java.util.HashMap;

import junit.framework.Assert;

import org.junit.Test;

import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.CSVFileFormatProperties;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.runtime.util.DataConverter;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;
import com.ibm.bi.dml.utils.Statistics;

public class CSVReadUnknownSizeTest extends AutomatedTestBase {
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n"
			+ "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	private final static String TEST_NAME = "csv_read_unknown";
	private final static String TEST_DIR = "functions/recompile/";

	private final static int rows = 10;
	private final static int cols = 15;

	/** Main method for running one test at a time from Eclipse. */
	public static void main(String[] args) {
		long startMsec = System.currentTimeMillis();

		CSVReadUnknownSizeTest t = new CSVReadUnknownSizeTest();
		t.setUpBase();
		t.setUp();
		t.testCSVReadUnknownSizeSplitRewrites();
		t.tearDown();

		long elapsedMsec = System.currentTimeMillis() - startMsec;
		System.err.printf("Finished in %1.3f sec\n", elapsedMsec / 1000.0);
	}

	@Override
	public void setUp() {
		TestUtils.clearAssertionInformation();
		addTestConfiguration(TEST_NAME, new TestConfiguration(TEST_DIR,
				TEST_NAME, new String[] { "X" }));
		setOutAndExpectedDeletionDisabled(true);
	}

	@Test
	public void testCSVReadUnknownSizeNoSplitNoRewrites() {
		runCSVReadUnknownSizeTest(false, false);
	}

	@Test
	public void testCSVReadUnknownSizeNoSplitRewrites() {
		runCSVReadUnknownSizeTest(false, true);
	}

	@Test
	public void testCSVReadUnknownSizeSplitNoRewrites() {
		runCSVReadUnknownSizeTest(true, false);
	}

	@Test
	public void testCSVReadUnknownSizeSplitRewrites() {
		runCSVReadUnknownSizeTest(true, true);
	}

	/**
	 * 
	 * @param condition
	 * @param branchRemoval
	 * @param IPA
	 */
	private void runCSVReadUnknownSizeTest( boolean splitDags, boolean rewrites )
	{	
		boolean oldFlagSplit = OptimizerUtils.ALLOW_SPLIT_HOP_DAGS;
		boolean oldFlagRewrites = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			/* This is for running the junit test the new way, i.e., construct the arguments directly */
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain","-args",HOME + INPUT_DIR + "X",
					                           HOME + OUTPUT_DIR + "R" };
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR;			
			loadTestConfiguration(config);

			OptimizerUtils.ALLOW_SPLIT_HOP_DAGS = splitDags;
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;

			double[][] X = getRandomMatrix(rows, cols, -1, 1, 1.0d, 7);
			MatrixBlock mb = DataConverter.convertToMatrixBlock(X);
			MatrixCharacteristics mc = new MatrixCharacteristics(rows, cols, 1000, 1000);
			CSVFileFormatProperties fprop = new CSVFileFormatProperties();			
			DataConverter.writeMatrixToHDFS(mb, HOME + INPUT_DIR + "X", OutputInfo.CSVOutputInfo, mc, -1, fprop);
			mc.set(-1, -1, -1, -1);
			MapReduceTool.writeMetaDataFile(HOME + INPUT_DIR + "X.mtd", ValueType.DOUBLE, mc, OutputInfo.CSVOutputInfo, fprop);
			
			runTest(true, false, null, -1); 
	
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			for( int i=0; i<rows; i++ )
				for( int j=0; j<cols; j++ )
				{
					Double tmp = dmlfile.get(new CellIndex(i+1,j+1));
					
					double expectedValue = mb.quickGetValue(i, j);			
					double actualValue =  (tmp==null)?0.0:tmp;
					
					if (expectedValue != actualValue) {
						throw new Exception(String.format("Value of cell (%d,%d) "
								+ "(zero-based indices) in output file %s is %f, "
								+ "but original value was %f",
								i, j, baseDirectory + OUTPUT_DIR + "R",
								actualValue, expectedValue));
					}
				}
			
			
			//check expected number of compiled and executed MR jobs
			//note: with algebraic rewrites - unary op in reducer prevents job-level recompile
			int expectedNumCompiled = (rewrites && !splitDags) ? 2 : 3; //reblock, GMR
			int expectedNumExecuted = splitDags ? 0 : rewrites ? 2 : 2;			
			
			Assert.assertEquals("Unexpected number of compiled MR jobs.", expectedNumCompiled, Statistics.getNoOfCompiledMRJobs()); 
			Assert.assertEquals("Unexpected number of executed MR jobs.", expectedNumExecuted, Statistics.getNoOfExecutedMRJobs()); 
		}
		catch(Exception ex)
		{
			throw new RuntimeException(ex);
		}
		finally
		{
			OptimizerUtils.ALLOW_SPLIT_HOP_DAGS = oldFlagSplit;
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlagRewrites;
		}
	}
}