/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2014
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.recompile;

import java.io.IOException;
import java.util.HashMap;

import junit.framework.Assert;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.runtime.DMLRuntimeException;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;
import com.ibm.bi.dml.utils.Statistics;

public class RewriteComplexMapMultChainTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2014\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME1 = "rewrite_mapmultchain1";
	private final static String TEST_NAME2 = "rewrite_mapmultchain2";
	private final static String TEST_DIR = "functions/recompile/";
	
	private final static int rows = 1974;
	private final static int cols = 45;
	
	private final static int cols2a = 1;
	private final static int cols2b = 3;
	
	private final static double sparsity = 0.7;
	private final static double eps = 0.0000001;
	
	
	@Override
	public void setUp() 
	{
		addTestConfiguration( TEST_NAME1, new TestConfiguration(TEST_DIR, TEST_NAME1, new String[] { "HV" }) );
		addTestConfiguration( TEST_NAME2, new TestConfiguration(TEST_DIR, TEST_NAME2, new String[] { "HV" }) );
	}

	
	
	@Test
	public void testRewriteExpr1SingleColumnCP() 
		throws DMLRuntimeException, IOException 
	{
		runRewriteMapMultChain(TEST_NAME1, true, ExecType.CP);
	}
	
	@Test
	public void testRewriteExpr1MultiColumnCP() 
		throws DMLRuntimeException, IOException 
	{
		runRewriteMapMultChain(TEST_NAME1, false, ExecType.CP);
	}
	
	@Test
	public void testRewriteExpr1SingleColumnMR() 
		throws DMLRuntimeException, IOException 
	{
		runRewriteMapMultChain(TEST_NAME1, true, ExecType.MR);
	}
	
	@Test
	public void testRewriteExpr1MultiColumnMR() 
		throws DMLRuntimeException, IOException 
	{
		runRewriteMapMultChain(TEST_NAME1, false, ExecType.MR);
	}
	
	@Test
	public void testRewriteExpr2SingleColumnCP() 
		throws DMLRuntimeException, IOException 
	{
		runRewriteMapMultChain(TEST_NAME2, true, ExecType.CP);
	}
	
	@Test
	public void testRewriteExpr2MultiColumnCP() 
		throws DMLRuntimeException, IOException 
	{
		runRewriteMapMultChain(TEST_NAME2, false, ExecType.CP);
	}
	
	@Test
	public void testRewriteExpr2SingleColumnMR() 
		throws DMLRuntimeException, IOException 
	{
		runRewriteMapMultChain(TEST_NAME2, true, ExecType.MR);
	}
	
	@Test
	public void testRewriteExpr2MultiColumnMR() 
		throws DMLRuntimeException, IOException 
	{
		runRewriteMapMultChain(TEST_NAME2, false, ExecType.MR);
	}
	

	/**
	 * 
	 * @param condition
	 * @param branchRemoval
	 * @param IPA
	 * @throws DMLRuntimeException 
	 * @throws IOException 
	 */
	private void runRewriteMapMultChain( String TEST_NAME, boolean singleCol, ExecType et ) 
		throws DMLRuntimeException, IOException
	{	
		RUNTIME_PLATFORM platformOld = rtplatform;
		
		try
		{
			TestConfiguration config = getTestConfiguration(TEST_NAME);
			
			String HOME = SCRIPT_DIR + TEST_DIR;
			fullDMLScriptName = HOME + TEST_NAME + ".dml";
			programArgs = new String[]{"-explain","-args", HOME + INPUT_DIR + "X",
												HOME + INPUT_DIR + "P",
												HOME + INPUT_DIR + "v",
					                            HOME + OUTPUT_DIR + "HV" };
			
			fullRScriptName = HOME + TEST_NAME + ".R";
			rCmd = "Rscript" + " " + fullRScriptName + " " +
					HOME + INPUT_DIR + " " + HOME + EXPECTED_DIR;			
			loadTestConfiguration(config);

			rtplatform = (et==ExecType.MR) ? RUNTIME_PLATFORM.HADOOP : RUNTIME_PLATFORM.HYBRID;
			
			//generate input data
			double[][] X = getRandomMatrix(rows, cols, 0, 1, sparsity, 7);
			writeInputMatrixWithMTD("X", X, true);
			double[][] P = getRandomMatrix(rows, singleCol?cols2a:cols2b, 0, 1, sparsity, 3);
			writeInputMatrixWithMTD("P", P, true);
			double[][] v = getRandomMatrix(cols, singleCol?cols2a:cols2b, 0, 1, 1.0, 21);
			writeInputMatrixWithMTD("v", v, true);
			
			//run test
			runTest(true, false, null, -1); 
			runRScript(true);
			
			//compare matrices
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("HV");
			HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("HV");
			TestUtils.compareMatrices(dmlfile, rfile, eps, "DML", "R");	
			
			
			//check expected number of compiled and executed MR jobs
			int expectedNumCompiled = (et==ExecType.CP)?1:(singleCol?4:6); //mapmultchain if single column
			int expectedNumExecuted = (et==ExecType.CP)?0:(singleCol?4:6); //mapmultchain if single column
			
			Assert.assertEquals("Unexpected number of compiled MR jobs.", expectedNumCompiled, Statistics.getNoOfCompiledMRJobs()); 
			Assert.assertEquals("Unexpected number of executed MR jobs.", expectedNumExecuted, Statistics.getNoOfExecutedMRJobs()); 
		}
		finally
		{
			rtplatform = platformOld;
		}
	}
	
}
