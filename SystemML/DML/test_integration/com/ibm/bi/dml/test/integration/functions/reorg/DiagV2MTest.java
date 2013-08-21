package com.ibm.bi.dml.test.integration.functions.reorg;

import java.util.HashMap;
import java.util.Random;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.runtime.matrix.io.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

public class DiagV2MTest extends AutomatedTestBase{

	private final static String TEST_DIR = "functions/reorg/";

	private final static double epsilon=0.0000000001;
	private final static int cols = 1059;
	private final static int min=0;
	private final static int max=100;
	
	@Override
	public void setUp() {
		addTestConfiguration("DiagV2MTest", new TestConfiguration(TEST_DIR, "DiagV2MTest", 
				new String[] {"C"}));
	}
	
	public void commonReorgTest(RUNTIME_PLATFORM platform)
	{
		TestConfiguration config = getTestConfiguration("DiagV2MTest");
	    
		RUNTIME_PLATFORM prevPlfm=rtplatform;
		
	    rtplatform = platform;

        config.addVariable("cols", cols);
          
		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String RI_HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = RI_HOME + "DiagV2MTest" + ".dml";
		programArgs = new String[]{"-args",  RI_HOME + INPUT_DIR + "A" , 
				Long.toString(cols), RI_HOME + OUTPUT_DIR + "C" };
		fullRScriptName = RI_HOME + "DiagV2MTest" + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + 
		       RI_HOME + INPUT_DIR + " "+ RI_HOME + EXPECTED_DIR;

		Random rand=new Random(System.currentTimeMillis());
		loadTestConfiguration(config);
		//double sparsity=0.2;
		double sparsity=rand.nextDouble();
		System.out.println("sparsity: "+sparsity);
        double[][] A = getRandomMatrix(1, cols, min, max, sparsity, System.currentTimeMillis());
        writeInputMatrix("A", A, true);
        sparsity=rand.nextDouble();   
		/*
		 * Expected number of jobs:
		 * Reblock - 1 job 
		 * While loop iteration - 10 jobs
		 * Final output write - 1 job
		 */
        //boolean exceptionExpected = false;
		//int expectedNumberOfJobs = 12;
		//runTest(exceptionExpected, null, expectedNumberOfJobs);
        boolean exceptionExpected = false;
		int expectedNumberOfJobs = -1;
		runTest(true, exceptionExpected, null, expectedNumberOfJobs);
		
		runRScript(true);
		//disableOutAndExpectedDeletion();
	
		for(String file: config.getOutputFiles())
		{
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS(file);
			HashMap<CellIndex, Double> rfile = readRMatrixFromFS(file);
		//	System.out.println(file+"-DML: "+dmlfile);
		//	System.out.println(file+"-R: "+rfile);
			TestUtils.compareMatrices(dmlfile, rfile, epsilon, file+"-DML", file+"-R");
		}
		rtplatform = prevPlfm;
	}
	
	@Test
	public void testDiagV2MMR() {
		commonReorgTest(RUNTIME_PLATFORM.HADOOP);
	}   
	
	@Test
	public void testDiagV2MCP() {
		commonReorgTest(RUNTIME_PLATFORM.SINGLE_NODE);
	}   
}

