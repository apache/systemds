/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.applications.parfor;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.lops.LopProperties.ExecType;
import com.ibm.bi.dml.runtime.controlprogram.ParForProgramBlock.PExecMode;
import com.ibm.bi.dml.runtime.controlprogram.parfor.stat.InfrastructureAnalyzer;
import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

/**
 * 
 * 
 */
public class ParForNaiveBayesTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME = "parfor_naive-bayes";
	private final static String TEST_DIR = "applications/parfor/";
	private final static double eps = 1e-10; 
	
	private final static int rows = 50000;
	private final static int cols1 = 105;      // # of columns in each vector  
	private final static int cols2 = 15;
	
	private final static double minVal=1;    // minimum value in each vector 
	private final static double maxVal=5; // maximum value in each vector 

	private final static double sparsity1 = 0.7;
	private final static double sparsity2 = 0.05;
	
	
	@Override
	public void setUp() 
	{
		TestUtils.clearAssertionInformation();
		addTestConfiguration(
				TEST_NAME, 
				new TestConfiguration(TEST_DIR, TEST_NAME, 
				new String[] { "class_prior", "class_conditionals" })   );  
	}
	
	@Test
	public void testParForNaiveBayesLocalDenseCP() 
	{
		runParForNaiveBayesTest(PExecMode.LOCAL, ExecType.CP, false, false);
	}
	
	@Test
	public void testParForNaiveBayesLocalSparseCP() 
	{
		runParForNaiveBayesTest(PExecMode.LOCAL, ExecType.CP, false, true);
	}
	
	@Test
	public void testParForNaiveBayesRemoteDenseCP() 
	{
		runParForNaiveBayesTest(PExecMode.REMOTE_MR, ExecType.CP, false, false);
	}
	
	@Test
	public void testParForNaiveBayesRemoteSparseCP() 
	{
		runParForNaiveBayesTest(PExecMode.REMOTE_MR, ExecType.CP, false, true);
	}
	
	/* MB: see small mem test for REMOTE_MR_DP
	@Test
	public void testParForNaiveBayesRemoteDPDenseCP() 
	{
		runParForNaiveBayesTest(PExecMode.REMOTE_MR_DP, ExecType.CP, false, false);
	}
	
	@Test
	public void testParForNaiveBayesRemoteDPSparseCP() 
	{
		runParForNaiveBayesTest(PExecMode.REMOTE_MR_DP, ExecType.CP, false, true);
	}
	*/
	
	@Test
	public void testParForNaiveBayesDefaultDenseCP() 
	{
		runParForNaiveBayesTest(null, ExecType.CP, false, false);
	}
	
	@Test
	public void testParForNaiveBayesDefaultSparseCP() 
	{
		runParForNaiveBayesTest(null, ExecType.CP, false, true);
	}
	
	@Test
	public void testParForNaiveBayesDefaultDenseMR() 
	{
		runParForNaiveBayesTest(null, ExecType.MR, false, false);
	}
	
	@Test
	public void testParForNaiveBayesDefaultSparseMR() 
	{
		runParForNaiveBayesTest(null, ExecType.MR, false, true);
	}

	@Test
	public void testParForNaiveBayesRemoteDPDenseCPSmallMem() 
	{
		runParForNaiveBayesTest(PExecMode.REMOTE_MR_DP, ExecType.CP, true, false);
	}
	
	@Test
	public void testParForNaiveBayesRemoteDPSparseCPSmallMem() 
	{
		runParForNaiveBayesTest(PExecMode.REMOTE_MR_DP, ExecType.CP, true, true);
	}
	
	/**
	 * 
	 * @param outer
	 * @param instType
	 * @param smallMem
	 * @param sparse
	 */
	private void runParForNaiveBayesTest( PExecMode outer, ExecType instType, boolean smallMem, boolean sparse )
	{
		int cols = (instType==ExecType.MR)? cols2 : cols1;
		
		//inst exec type, influenced via rows
		RUNTIME_PLATFORM oldPlatform = rtplatform;
		rtplatform = (instType==ExecType.MR)? RUNTIME_PLATFORM.HADOOP : RUNTIME_PLATFORM.HYBRID;
		
		//determine the script
		int scriptNum = -1;
		if( outer == PExecMode.LOCAL )      scriptNum=1; //constrained opt
		else if( outer == PExecMode.REMOTE_MR ) scriptNum=2; //constrained opt
		else if( outer == PExecMode.REMOTE_MR_DP ) 	scriptNum=3; //constrained opt
		else                                    scriptNum=4; //opt
	
		//invocation arguments
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME +scriptNum + ".dml";
		programArgs = new String[]{"-args", HOME + INPUT_DIR + "D" ,
				                            HOME + INPUT_DIR + "C",
				                            Integer.toString((int)maxVal),
				                            HOME + OUTPUT_DIR + "class_prior",
				                            HOME + OUTPUT_DIR + "class_conditionals"};
		
		fullRScriptName = HOME + TEST_NAME + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + 
		       HOME + INPUT_DIR + " " + Integer.toString((int)maxVal) + " " + HOME + EXPECTED_DIR;
		
		loadTestConfiguration(config);

		//input data
		double[][] D = getRandomMatrix(rows, cols, -1, 1, sparse?sparsity2:sparsity1, 7); 
		double[][] C = TestUtils.round(getRandomMatrix(rows, 1, minVal, maxVal, 1.0, 3)); 
		MatrixCharacteristics mc1 = new MatrixCharacteristics(rows,cols,-1,-1);
		writeInputMatrixWithMTD("D", D, true, mc1);
		MatrixCharacteristics mc2 = new MatrixCharacteristics(rows,1,-1,-1);
		writeInputMatrixWithMTD("C", C, true, mc2);
		
		//set memory budget (to test automatic opt for remote_mr_dp)
		long oldmem = InfrastructureAnalyzer.getLocalMaxMemory();
		if(smallMem) {
			long mem = 1024*1024*8;
			InfrastructureAnalyzer.setLocalMaxMemory(mem);
		}
		
		try
		{
			//run the testcase (DML and R)
			runTest(true, false, null, -1);
			runRScript(true); 
				
			//compare output matrices
			for( String out : new String[]{"class_prior", "class_conditionals" } )
			{
				HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS(out);				
				HashMap<CellIndex, Double> rfile  = readRMatrixFromFS(out);
				TestUtils.compareMatrices(dmlfile, rfile, eps, "Stat-DML", "Stat-R");
			}	
		}
		finally
		{
			rtplatform = oldPlatform;
			InfrastructureAnalyzer.setLocalMaxMemory(oldmem);
		}
	}

}