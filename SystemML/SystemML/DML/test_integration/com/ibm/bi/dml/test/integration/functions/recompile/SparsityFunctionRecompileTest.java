/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
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
import com.ibm.bi.dml.utils.Statistics;

public class SparsityFunctionRecompileTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_DIR = "functions/recompile/";
	private final static String TEST_NAME1 = "while_recompile_func_sparse";
	private final static String TEST_NAME2 = "if_recompile_func_sparse";
	private final static String TEST_NAME3 = "for_recompile_func_sparse";
	private final static String TEST_NAME4 = "parfor_recompile_func_sparse";
	
	private final static long rows = 1000;
	private final static long cols = 500000;    
	private final static double sparsity = 0.00001d;    
	private final static double val = 7.0;
	
	
	@Override
	public void setUp() 
	{
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
	public void testWhileRecompileIPA() 
	{
		runRecompileTest(TEST_NAME1, true, true);
	}
	
	@Test
	public void testWhileNoRecompileIPA() 
	{
		runRecompileTest(TEST_NAME1, false, true);
	}
	
	@Test
	public void testIfRecompileIPA() 
	{
		runRecompileTest(TEST_NAME2, true, true);
	}
	
	@Test
	public void testIfNoRecompileIPA() 
	{
		runRecompileTest(TEST_NAME2, false, true);
	}
	
	@Test
	public void testForRecompileIPA() 
	{
		runRecompileTest(TEST_NAME3, true, true);
	}
	
	@Test
	public void testForNoRecompileIPA() 
	{
		runRecompileTest(TEST_NAME3, false, true);
	}
	
	@Test
	public void testParForRecompileIPA() 
	{
		runRecompileTest(TEST_NAME4, true, true);
	}
	
	@Test
	public void testParForNoRecompileIPA() 
	{
		runRecompileTest(TEST_NAME4, false, true);
	}
	
	@Test
	public void testWhileRecompileNoIPA() 
	{
		runRecompileTest(TEST_NAME1, true, false);
	}
	
	@Test
	public void testWhileNoRecompileNoIPA() 
	{
		runRecompileTest(TEST_NAME1, false, false);
	}
	
	@Test
	public void testIfRecompileNoIPA() 
	{
		runRecompileTest(TEST_NAME2, true, false);
	}
	
	@Test
	public void testIfNoRecompileNoIPA() 
	{
		runRecompileTest(TEST_NAME2, false, false);
	}
	
	@Test
	public void testForRecompileNoIPA() 
	{
		runRecompileTest(TEST_NAME3, true, false);
	}
	
	@Test
	public void testForNoRecompileNoIPA() 
	{
		runRecompileTest(TEST_NAME3, false, false);
	}
	
	@Test
	public void testParForRecompileNoIPA() 
	{
		runRecompileTest(TEST_NAME4, true, false);
	}
	
	@Test
	public void testParForNoRecompileNoIPA() 
	{
		runRecompileTest(TEST_NAME4, false, false);
	}
	
	
	private void runRecompileTest( String testname, boolean recompile, boolean IPA )
	{	
		boolean oldFlagRecompile = OptimizerUtils.ALLOW_DYN_RECOMPILATION;
		boolean oldFlagIPA = OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS;
		boolean oldFlagBranchRemoval = OptimizerUtils.ALLOW_BRANCH_REMOVAL;
		
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
			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = IPA;
			OptimizerUtils.ALLOW_BRANCH_REMOVAL = false;
			
			MatrixBlock mb = MatrixBlock.randOperations((int)rows, (int)cols, sparsity, 0, 1, "uniform", System.currentTimeMillis());
			MatrixCharacteristics mc = new MatrixCharacteristics(rows,cols,DMLTranslator.DMLBlockSize,DMLTranslator.DMLBlockSize,(long)(rows*cols*sparsity));
			DataConverter.writeMatrixToHDFS(mb, HOME + INPUT_DIR + "V", OutputInfo.TextCellOutputInfo, mc);
			MapReduceTool.writeMetaDataFile(HOME + INPUT_DIR + "V.mtd", ValueType.DOUBLE, mc, OutputInfo.TextCellOutputInfo);
			
			boolean exceptionExpected = false;
			runTest(true, exceptionExpected, null, -1); 
			
			//CHECK compiled MR jobs
			int expectNumCompiled = 4 + ((testname.equals(TEST_NAME4))?2:0) //reblock,GMR,GMR,GMR, (+2 resultmerge)
					                  + (IPA ? 0 : (testname.equals(TEST_NAME2)?2:1)); //GMR ua(+), 2x for if
			Assert.assertEquals("Unexpected number of compiled MR jobs.", 
					            expectNumCompiled, Statistics.getNoOfCompiledMRJobs());
		
			//CHECK executed MR jobs
			int expectNumExecuted = -1;
			if( recompile ) expectNumExecuted = 1 + ((testname.equals(TEST_NAME4))?2:0); //GMR (indexing), (+2 resultmerge) 
			else            expectNumExecuted = 4 + ((testname.equals(TEST_NAME4))?2:0) //reblock,GMR,GMR,GMR, (+2 resultmerge) 
					                              + (IPA ? 0 : 1); //GMR ua(+)
			Assert.assertEquals("Unexpected number of executed MR jobs.", 
		                        expectNumExecuted, Statistics.getNoOfExecutedMRJobs());

			
			//compare matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			Assert.assertEquals((double)val, dmlfile.get(new CellIndex(1,1)));
		}
		catch(Exception ex)
		{
			ex.printStackTrace();
			Assert.fail("Failed to run test: "+ex.getMessage());
		}
		finally
		{
			OptimizerUtils.ALLOW_DYN_RECOMPILATION = oldFlagRecompile;
			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = oldFlagIPA;
			OptimizerUtils.ALLOW_BRANCH_REMOVAL = oldFlagBranchRemoval;
		}
	}
	
}