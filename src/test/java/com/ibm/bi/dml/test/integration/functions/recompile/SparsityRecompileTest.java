/**
 * (C) Copyright IBM Corp. 2010, 2015
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 */

package com.ibm.bi.dml.test.integration.functions.recompile;

import java.util.HashMap;

import org.junit.Assert;

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
	
	private final static String TEST_DIR = "functions/recompile/";
	private final static String TEST_NAME1 = "while_recompile_sparse";
	private final static String TEST_NAME2 = "if_recompile_sparse";
	private final static String TEST_NAME3 = "for_recompile_sparse";
	private final static String TEST_NAME4 = "parfor_recompile_sparse";
	
	private final static long rows = 1000;
	private final static long cols = 500000;    
	private final static double sparsity = 0.00001d;    
	private final static double val = 7.0;
	
	/**
	 * Main method for running one test at a time.
	 */
	public static void main(String[] args) {
		long startMsec = System.currentTimeMillis();

		SparsityRecompileTest t = new SparsityRecompileTest();
		t.setUpBase();
		t.setUp();
		t.testParForRecompile();
		t.tearDown();

		long elapsedMsec = System.currentTimeMillis() - startMsec;
		System.err.printf("Finished in %1.3f sec\n", elapsedMsec / 1000.0);

	}
	
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
			programArgs = new String[]{"-explain", "-args",HOME + INPUT_DIR + "V",
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
			
			//CHECK compiled MR jobs (changed 07/2015 due to better sparsity inference, 08/2015 due to better worst-case inference)
			int expectNumCompiled =   (testname.equals(TEST_NAME2)?3:4) //reblock,GMR,GMR,GMR (one GMR less for if) 
	                                + ((testname.equals(TEST_NAME4) && !recompile)?2:0);//(+2 resultmerge)
			Assert.assertEquals("Unexpected number of compiled MR jobs.", 
					            expectNumCompiled, Statistics.getNoOfCompiledMRJobs());
		
			//CHECK executed MR jobs (changed 07/2015 due to better sparsity inference, 08/2015 due to better worst-case inference)
			int expectNumExecuted = -1;
			if( recompile ) expectNumExecuted = 0; //+ ((testname.equals(TEST_NAME4))?2:0); //(+2 resultmerge) 
			else            expectNumExecuted =  (testname.equals(TEST_NAME2)?3:4) //reblock,GMR,GMR,GMR (one GMR less for if)
					                              + ((testname.equals(TEST_NAME4))?2:0); //(+2 resultmerge) 
			Assert.assertEquals("Unexpected number of executed MR jobs.", 
		                        expectNumExecuted, Statistics.getNoOfExecutedMRJobs());

			
			//compare matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			Assert.assertEquals((Double)val, dmlfile.get(new CellIndex(1,1)));
		}
		catch(Exception ex)
		{
			throw new RuntimeException(ex);
			//Assert.fail("Failed to run test: "+ex.getMessage());
		}
		finally
		{
			OptimizerUtils.ALLOW_DYN_RECOMPILATION = oldFlagRecompile;
		}
	}
	
}