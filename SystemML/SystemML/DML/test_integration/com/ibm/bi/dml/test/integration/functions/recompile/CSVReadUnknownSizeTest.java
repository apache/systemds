/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.recompile;

import java.util.HashMap;

import junit.framework.Assert;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapred.JobConf;
import org.junit.Test;

import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.io.CSVFileFormatProperties;
import com.ibm.bi.dml.runtime.matrix.io.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.io.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue.CellIndex;
import com.ibm.bi.dml.runtime.util.DataConverter;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.utils.Statistics;

public class CSVReadUnknownSizeTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME = "csv_read_unknown";
	private final static String TEST_DIR = "functions/recompile/";
	
	private final static int rows = 10;
	private final static int cols = 15;    
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration( TEST_NAME, new TestConfiguration(TEST_DIR, TEST_NAME, new String[] { "X" }) );
	}

	@Test
	public void testCSVReadUnknownSizeNoSplitNoRewrites() 
	{
		runCSVReadUnknownSizeTest( false, false );
	}
	
	@Test
	public void testCSVReadUnknownSizeNoSplitRewrites() 
	{
		runCSVReadUnknownSizeTest( false, true );
	}
	
	@Test
	public void testCSVReadUnknownSizeSplitNoRewrites() 
	{
		runCSVReadUnknownSizeTest( true, false );
	}
	
	@Test
	public void testCSVReadUnknownSizeSplitRewrites() 
	{
		runCSVReadUnknownSizeTest( true, true );
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
			DataConverter.writeCSVToHDFS(new Path(HOME + INPUT_DIR + "X"), new JobConf(), mb, 
					                    rows, cols, -1, fprop);
			mc.set(-1, -1, -1, -1);
			MapReduceTool.writeMetaDataFile(HOME + INPUT_DIR + "X.mtd", ValueType.DOUBLE, mc, OutputInfo.CSVOutputInfo, fprop);
			
			runTest(true, false, null, -1); 
			
			
			//compare matrices 
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			for( int i=0; i<rows; i++ )
				for( int j=0; j<cols; j++ )
				{
					Double tmp = dmlfile.get(new CellIndex(i+1,j+1));
					Assert.assertEquals(mb.quickGetValue(i, j), (tmp==null)?0:tmp);
				}
			
			
			//check expected number of compiled and executed MR jobs
			//note: with algebraic rewrites - unary op in reducer prevents job-level recompile
			int expectedNumCompiled = 2; //reblock, GMR
			int expectedNumExecuted = splitDags ? 0 : rewrites ? 2 : 1;			
			
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