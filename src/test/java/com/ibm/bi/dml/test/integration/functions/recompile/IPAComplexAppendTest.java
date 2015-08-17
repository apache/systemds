/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.recompile;

import java.io.IOException;

import org.junit.Test;

import com.ibm.bi.dml.hops.OptimizerUtils;
import com.ibm.bi.dml.parser.Expression.ValueType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixBlock;
import com.ibm.bi.dml.runtime.matrix.data.OutputInfo;
import com.ibm.bi.dml.runtime.util.DataConverter;
import com.ibm.bi.dml.runtime.util.MapReduceTool;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;

public class IPAComplexAppendTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME = "append_nnz";
	private final static String TEST_DIR = "functions/recompile/";
	
	private final static int rows = 200000;
	private final static int cols = 1000;
	private final static int nnz  = 700; //ultra-sparse
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration( TEST_NAME, new TestConfiguration(TEST_DIR, TEST_NAME, new String[] { "Y" }) );
	}

	
	
	@Test
	public void testComplexAppendNoIPANoRewrites() 
		throws DMLRuntimeException, IOException 
	{
		runIPAAppendTest(false, false);
	}
	
	@Test
	public void testComplexAppendIPANoRewrites() 
		throws DMLRuntimeException, IOException 
	{
		runIPAAppendTest(true, false);
	}
	
	@Test
	public void testComplexAppendNoIPARewrites() 
		throws DMLRuntimeException, IOException 
	{
		runIPAAppendTest(false, true);
	}
	
	@Test
	public void testComplexAppendIPARewrites() 
		throws DMLRuntimeException, IOException 
	{
		runIPAAppendTest(true, true);
	}

	/**
	 * 
	 * @param condition
	 * @param branchRemoval
	 * @param IPA
	 * @throws DMLRuntimeException 
	 * @throws IOException 
	 */
	private void runIPAAppendTest( boolean IPA, boolean rewrites ) 
		throws DMLRuntimeException, IOException
	{	
		boolean oldFlagIPA = OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS;
		boolean oldFlagRewrites = OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION;
		
		
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-args", 
					                         HOME + OUTPUT_DIR + "X" };
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " +
					 HOME + EXPECTED_DIR + " " + HOME + EXPECTED_DIR;			
			loadTestConfiguration(config);

			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = IPA;
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = rewrites;
			
			
			//generate input data
			MatrixBlock mb = MatrixBlock.randOperations(rows, cols, OptimizerUtils.getSparsity(rows, cols, nnz), -1, 1, "uniform", 7);
			MatrixCharacteristics mc1 = new MatrixCharacteristics(rows,cols,1000,1000,nnz);
			DataConverter.writeMatrixToHDFS(mb, HOME + OUTPUT_DIR + "X", OutputInfo.BinaryBlockOutputInfo, mc1);
			MapReduceTool.writeMetaDataFile(HOME + OUTPUT_DIR + "X.mtd", ValueType.DOUBLE, mc1, OutputInfo.BinaryBlockOutputInfo);
			
			//run test
			runTest(true, false, null, -1); 
			
			//check expected number of compiled and executed MR jobs
			int expectedNumCompiled = (rewrites&&IPA)?2:3; //(GMR mm,) GMR append, GMR sum
			int expectedNumExecuted = rewrites?0:1; //(GMR mm) 			
			
			checkNumCompiledMRJobs(expectedNumCompiled); 
			checkNumExecutedMRJobs(expectedNumExecuted); 
		}
		finally
		{
			OptimizerUtils.ALLOW_INTER_PROCEDURAL_ANALYSIS = oldFlagIPA;
			OptimizerUtils.ALLOW_ALGEBRAIC_SIMPLIFICATION = oldFlagRewrites;
		}
	}
	
}
