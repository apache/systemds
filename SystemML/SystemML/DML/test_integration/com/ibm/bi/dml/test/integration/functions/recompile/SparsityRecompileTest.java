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
import com.ibm.bi.dml.parser.DMLTranslator;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.OutputInfo;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.runtime.util.DataConverter;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;
import com.ibm.bi.dml.utils.Statistics;

public class SparsityRecompileTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_DIR = "functions/recompile/";
	private final static String TEST_NAME1 = "while_recompile_sparse";
	private final static String TEST_NAME2 = "if_recompile_sparse";
	private final static String TEST_NAME3 = "for_recompile_sparse";
	private final static String TEST_NAME4 = "parfor_recompile_sparse";
	
	private final static long rows = 1000;
	private final static long cols = 500000;    
	private final static double sparsity = 0.00001d;    
	private final static double val = 7.0;
	
	
	@Override
	public void setUp() 
	{
		TestUtils.clearAssertionInformation();
		addTestConfiguration(
				TEST_NAME1, 
				new TestConfiguration(TEST_DIR, TEST_NAME1, 
				new String[] { "Rout" })   );
		addTestConfiguration(
				TEST_NAME2, 
				new TestConfiguration(TEST_DIR, TEST_NAME2, 
				new String[] { "Rout" })   );
		addTestConfiguration(
				TEST_NAME3, 
				new TestConfiguration(TEST_DIR, TEST_NAME3, 
				new String[] { "Rout" })   );
		addTestConfiguration(
				TEST_NAME4, 
				new TestConfiguration(TEST_DIR, TEST_NAME4, 
				new String[] { "Rout" })   );
	}

	@Test
	public void testWhileRecompile() 
	{
		runRecompileTest(TEST_NAME1, true);
	}
	
	@Test
	public void testWhileNoRecompile() 
	{
		runRecompileTest(TEST_NAME1, false);
	}
	
	@Test
	public void testIfRecompile() 
	{
		runRecompileTest(TEST_NAME2, true);
	}
	
	@Test
	public void testIfNoRecompile() 
	{
		runRecompileTest(TEST_NAME2, false);
	}
	
	@Test
	public void testForRecompile() 
	{
		runRecompileTest(TEST_NAME3, true);
	}
	
	@Test
	public void testForNoRecompile() 
	{
		runRecompileTest(TEST_NAME3, false);
	}
	
	@Test
	public void testParForRecompile() 
	{
		runRecompileTest(TEST_NAME4, true);
	}
	
	@Test
	public void testParForNoRecompile() 
	{
		runRecompileTest(TEST_NAME4, false);
	}

	
	private void runRecompileTest( String testname, boolean recompile )
	{	
		boolean oldFlagRecompile = OptimizerUtils.ALLOW_DYN_RECOMPILATION;
		
		try
		{
			TestConfiguration config = getTestConfiguration(testname);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{"-args",HOME + INPUT_DIR + "V",
					                           Double.toString(val),
					                           HOME + OUTPUT_DIR + "R" };
			fullRScriptName = HOME + testname + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " + 
			       HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR;			
			loadTestConfiguration(config);

			OptimizerUtils.ALLOW_DYN_RECOMPILATION = recompile;
			
			MatrixBlock mb = MatrixBlock.randOperations((int)rows, (int)cols, sparsity, 0, 1, "uniform", System.currentTimeMillis());
			MatrixCharacteristics mc = new MatrixCharacteristics(rows,cols,DMLTranslator.DMLBlockSize,DMLTranslator.DMLBlockSize,(long)(rows*cols*sparsity));
			DataConverter.writeMatrixToHDFS(mb, HOME + INPUT_DIR + "V", OutputInfo.TextCellOutputInfo, mc);
			MapReduceTool.writeMetaDataFile(HOME + INPUT_DIR + "V.mtd", ValueType.DOUBLE, mc, OutputInfo.TextCellOutputInfo);
			
			boolean exceptionExpected = false;
			runTest(true, exceptionExpected, null, -1); 
			
			//CHECK compiled MR jobs
			int expectNumCompiled = 4 + ((testname.equals(TEST_NAME4))?2:0);//reblock,GMR,GMR,GMR, (+2 resultmerge)
			Assert.assertEquals("Unexpected number of compiled MR jobs.", 
					            expectNumCompiled, Statistics.getNoOfCompiledMRJobs());
		
			//CHECK executed MR jobs
			int expectNumExecuted = -1;
			if( recompile ) expectNumExecuted = 1 + ((testname.equals(TEST_NAME4))?2:0); //GMR (indexing), (+2 resultmerge) 
			else            expectNumExecuted = 4 + ((testname.equals(TEST_NAME4))?2:0); //reblock,GMR,GMR,GMR, (+2 resultmerge) 
			Assert.assertEquals("Unexpected number of executed MR jobs.", 
		                        expectNumExecuted, Statistics.getNoOfExecutedMRJobs());

			
			//compare matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			Assert.assertEquals((double)val, dmlfile.get(new CellIndex(1,1)));
		}
		catch(Exception ex)
		{
			Assert.fail("Failed to run test: "+ex.getMessage());
		}
		finally
		{
			OptimizerUtils.ALLOW_DYN_RECOMPILATION = oldFlagRecompile;
		}
	}
	
}