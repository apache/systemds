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

package org.apache.sysml.test.integration.functions.recompile;

import java.util.HashMap;

import org.junit.Assert;
import org.junit.Test;

import org.apache.sysml.hops.OptimizerUtils;
import org.apache.sysml.parser.DMLTranslator;
import org.apache.sysml.parser.Expression.ValueType;
import org.apache.sysml.runtime.matrix.MatrixCharacteristics;
import org.apache.sysml.runtime.matrix.data.MatrixBlock;
import org.apache.sysml.runtime.matrix.data.OutputInfo;
import org.apache.sysml.runtime.matrix.data.MatrixValue.CellIndex;
import org.apache.sysml.runtime.util.DataConverter;
import org.apache.sysml.runtime.util.MapReduceTool;
import org.apache.sysml.test.integration.AutomatedTestBase;
import org.apache.sysml.test.integration.TestConfiguration;
import org.apache.sysml.utils.Statistics;

public class SparsityFunctionRecompileTest extends AutomatedTestBase 
{
	
	private final static String TEST_DIR = "functions/recompile/";
	private final static String TEST_NAME1 = "while_recompile_func_sparse";
	private final static String TEST_NAME2 = "if_recompile_func_sparse";
	private final static String TEST_NAME3 = "for_recompile_func_sparse";
	private final static String TEST_NAME4 = "parfor_recompile_func_sparse";
	private final static String TEST_CLASS_DIR = TEST_DIR + 
		SparsityFunctionRecompileTest.class.getSimpleName() + "/";
	
	private final static long rows = 1000;
	private final static long cols = 500000;    
	private final static double sparsity = 0.00001d;    
	private final static double val = 7.0;
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(TEST_NAME1, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME1, new String[] { "Rout" }) );
		addTestConfiguration(TEST_NAME2, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME2, new String[] { "Rout" }) );
		addTestConfiguration(TEST_NAME3, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME3, new String[] { "Rout" }) );
		addTestConfiguration(TEST_NAME4, 
			new TestConfiguration(TEST_CLASS_DIR, TEST_NAME4, new String[] { "Rout" }) );
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
			loadTestConfiguration(config);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + testname + ".dml";
			programArgs = new String[]{"-args",
				input("V"), Double.toString(val), output("R") };

			OptimizerUtils.ALLOW_DYN_RECOMPILATION = recompile;
			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = IPA;
			OptimizerUtils.ALLOW_BRANCH_REMOVAL = false;
			
			MatrixBlock mb = MatrixBlock.randOperations((int)rows, (int)cols, sparsity, 0, 1, "uniform", 732);
			MatrixCharacteristics mc = new MatrixCharacteristics(rows,cols,DMLTranslator.DMLBlockSize,DMLTranslator.DMLBlockSize,(long)(rows*cols*sparsity));
			
			DataConverter.writeMatrixToHDFS(mb, input("V"), OutputInfo.TextCellOutputInfo, mc);
			MapReduceTool.writeMetaDataFile(input("V.mtd"), ValueType.DOUBLE, mc, OutputInfo.TextCellOutputInfo);
			
			boolean exceptionExpected = false;
			runTest(true, exceptionExpected, null, -1); 
			
			//CHECK compiled MR jobs (changed 07/2015 due to better sparsity inference)
			int expectNumCompiled = (testname.equals(TEST_NAME2)?3:4) //reblock,GMR,GMR,GMR (one GMR less for if) 
					                 + ((testname.equals(TEST_NAME4))?2:0) //(+2 resultmerge)
					                 + (IPA ? 0 : (testname.equals(TEST_NAME2)?3:1)); //GMR ua(+), 3x for if
			Assert.assertEquals("Unexpected number of compiled MR jobs.", 
					            expectNumCompiled, Statistics.getNoOfCompiledMRJobs());
		
			//CHECK executed MR jobs (changed 07/2015 due to better sparsity inference)
			int expectNumExecuted = -1;
			if( recompile ) expectNumExecuted = 0 + ((testname.equals(TEST_NAME4))?2:0); //(2x resultmerge) 
			else            expectNumExecuted = (testname.equals(TEST_NAME2)?3:4) //reblock,GMR,GMR,GMR (one GMR less for if) 
					                              + ((testname.equals(TEST_NAME4))?2:0) //(+2 resultmerge) 
					                              + (IPA ? 0 : (testname.equals(TEST_NAME2)?2:1)); //GMR ua(+)
			Assert.assertEquals("Unexpected number of executed MR jobs.", 
		                        expectNumExecuted, Statistics.getNoOfExecutedMRJobs());

			
			//compare matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("R");
			Assert.assertEquals(Double.valueOf(val), dmlfile.get(new CellIndex(1,1)));
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