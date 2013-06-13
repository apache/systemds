package com.ibm.bi.dml.test.integration.applications.descriptivestats;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.runtime.matrix.io.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;


public class BivariateCategoricalCategoricallTest extends AutomatedTestBase {

	private final static String TEST_DIR = "applications/descriptivestats/";
	private final static String TEST_NOMINAL_NOMINAL = "CategoricalCategorical";
	private final static String TEST_NOMINAL_NOMINAL_WEIGHTS = "CategoricalCategoricalWithWeightsTest";
	private final static String TEST_ODDS_RATIO = "OddsRatio";

	private final static double eps = 1e-9;
	private int rows = 10000;  // # of rows in each vector
	private int ncatA = 100;   // # of categories in A
	private int ncatB = 150;   // # of categories in B
	private int maxW = 100;    // maximum weight
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NOMINAL_NOMINAL, new TestConfiguration(TEST_DIR, TEST_NOMINAL_NOMINAL, 
				new String[] { "PValue"+".scalar", "CramersV"+".scalar" }));
		addTestConfiguration(TEST_NOMINAL_NOMINAL_WEIGHTS, new TestConfiguration(TEST_DIR, TEST_NOMINAL_NOMINAL_WEIGHTS, new String[] { "PValue"+".scalar", "CramersV"+".scalar" }));
		addTestConfiguration(TEST_ODDS_RATIO, new TestConfiguration(TEST_DIR, TEST_ODDS_RATIO, new String[] { 	"oddsRatio"+".scalar", 
																												"sigma"+".scalar", 
																												"leftConf"+".scalar", 
																												"rightConf"+".scalar", 
																												"sigmasAway"+".scalar" 
																												//"chiSquared"+".scalar", 
																												//"degFreedom"+".scalar", 
																												//"pValue"+".scalar", 
																												//"cramersV"+".scalar"
																												}));
	}

	@Test
	public void testCategoricalCategorical() {
		
		TestConfiguration config = getTestConfiguration(TEST_NOMINAL_NOMINAL);
		
		config.addVariable("rows", rows);

		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String CC_HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = CC_HOME + TEST_NOMINAL_NOMINAL + ".dml";
		programArgs = new String[]{"-args",  CC_HOME + INPUT_DIR + "A" , 
				                        Integer.toString(rows),
				                        CC_HOME + INPUT_DIR + "B" , 
				                        CC_HOME + OUTPUT_DIR + "PValue" , 
				                        CC_HOME + OUTPUT_DIR + "CramersV" };
		fullRScriptName = CC_HOME + TEST_NOMINAL_NOMINAL + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + 
		       CC_HOME + INPUT_DIR + " " + CC_HOME + EXPECTED_DIR;
		
		loadTestConfiguration(config);

        double[][] A = getRandomMatrix(rows, 1, 1, ncatA, 1, System.currentTimeMillis());
        double[][] B = getRandomMatrix(rows, 1, 1, ncatB, 1, System.currentTimeMillis()+1);
        round(A);
        round(B);
        
		writeInputMatrix("A", A, true);
		writeInputMatrix("B", B, true);

		/*
		 * Expected number of jobs:
		 * Reblock - 1 job 
		 * While loop iteration - 7 jobs
		 * Final output write - 1 job
		 */
		//boolean exceptionExpected = false;
		//int expectedNumberOfJobs = 5;
		runTest(true, false, null, -1);
		
		runRScript(true);
		
		for(String file: config.getOutputFiles())
		{
			/* NOte that some files do not contain matrix, but just a single scalar value inside */
			HashMap<CellIndex, Double> dmlfile;
			HashMap<CellIndex, Double> rfile;
			if (file.endsWith(".scalar")) {
				file = file.replace(".scalar", "");
				dmlfile = readDMLScalarFromHDFS(file);
				rfile = readRScalarFromFS(file);
			}
			else {
				dmlfile = readDMLMatrixFromHDFS(file);
				rfile = readRMatrixFromFS(file);
			}
			TestUtils.compareMatrices(dmlfile, rfile, eps, file+"-DML", file+"-R");
		}
		
	}
	
	private void round(double[][] weight) {
		for(int i=0; i<weight.length; i++)
			weight[i][0]=Math.round(weight[i][0]);
	}

	@Test
	public void testCategoricalCategoricalWithWeights() {

		TestConfiguration config = getTestConfiguration(TEST_NOMINAL_NOMINAL_WEIGHTS);
		
		config.addVariable("rows", rows);

		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String CC_HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = CC_HOME + TEST_NOMINAL_NOMINAL_WEIGHTS + ".dml";
		programArgs = new String[]{"-args",  CC_HOME + INPUT_DIR + "A" , 
				                        Integer.toString(rows),
				                        CC_HOME + INPUT_DIR + "B" , 
				                        CC_HOME + INPUT_DIR + "WM" , 
				                        CC_HOME + OUTPUT_DIR + "PValue" , 
				                        CC_HOME + OUTPUT_DIR + "CramersV" };
		fullRScriptName = CC_HOME + TEST_NOMINAL_NOMINAL_WEIGHTS + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + 
		       CC_HOME + INPUT_DIR + " " + CC_HOME + EXPECTED_DIR;
		
		loadTestConfiguration(config);

        double[][] A = getRandomMatrix(rows, 1, 1, ncatA, 1, System.currentTimeMillis());
        double[][] B = getRandomMatrix(rows, 1, 1, ncatB, 1, System.currentTimeMillis()+1);
        double[][] WM = getRandomMatrix(rows, 1, 1, maxW, 1, System.currentTimeMillis()+2);
        round(A);
        round(B);
        round(WM);
        
		writeInputMatrix("A", A, true);
		writeInputMatrix("B", B, true);
		writeInputMatrix("WM", WM, true);
        createHelperMatrix();
        
		/*
		 * Expected number of jobs:
		 * Mean etc - 2 jobs (reblock & gmr)
		 * Cov etc - 2 jobs
		 * Final output write - 1 job
		 */
		//boolean exceptionExpected = false;
		//int expectedNumberOfJobs = 5;
		//runTest(exceptionExpected, null, expectedNumberOfJobs);
		
        runTest(true, false, null, -1);
		runRScript(true);
		
		for(String file: config.getOutputFiles())
		{
			HashMap<CellIndex, Double> dmlfile;
			HashMap<CellIndex, Double> rfile;
			if (file.endsWith(".scalar")) {
				file = file.replace(".scalar", "");
				dmlfile = readDMLScalarFromHDFS(file);
				rfile = readRScalarFromFS(file);
			}
			else {
				dmlfile = readDMLMatrixFromHDFS(file);
				rfile = readRMatrixFromFS(file);
			}
			TestUtils.compareMatrices(dmlfile, rfile, eps, file+"-DML", file+"-R");
		}
		
	}
	
	@Test
	public void testOddsRatio() {
		
		TestConfiguration config = getTestConfiguration(TEST_ODDS_RATIO);
		
		config.addVariable("rows", rows);

		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String CC_HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = CC_HOME + TEST_ODDS_RATIO + ".dml";
		programArgs = new String[]{"-args",  CC_HOME + INPUT_DIR + "A" , 
				                        Integer.toString(rows),
				                        CC_HOME + INPUT_DIR + "B" , 
				                        CC_HOME + OUTPUT_DIR + "oddsRatio" , 
				                        CC_HOME + OUTPUT_DIR + "sigma", 
				                        CC_HOME + OUTPUT_DIR + "leftConf" , 
				                        CC_HOME + OUTPUT_DIR + "rightConf", 
				                        CC_HOME + OUTPUT_DIR + "sigmasAway" 
				                        //CC_HOME + OUTPUT_DIR + "chiSquared", 
				                        //CC_HOME + OUTPUT_DIR + "degFreedom" , 
				                        //CC_HOME + OUTPUT_DIR + "pValue", 
				                        //CC_HOME + OUTPUT_DIR + "cramersV"
				                        };
		fullRScriptName = CC_HOME + TEST_ODDS_RATIO + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + 
		       CC_HOME + INPUT_DIR + " " + CC_HOME + EXPECTED_DIR;
		
		loadTestConfiguration(config);

		// current test works only for 2x2 contingency tables => #categories must be 2
		int numCat = 2;
        double[][] A = getRandomMatrix(rows, 1, 1, numCat, 1, System.currentTimeMillis());
        double[][] B = getRandomMatrix(rows, 1, 1, numCat, 1, System.currentTimeMillis()+1);
        round(A);
        round(B);
        
		writeInputMatrix("A", A, true);
		writeInputMatrix("B", B, true);

		runTest(true, false, null, -1);
		
		runRScript(true);
		
		for(String file: config.getOutputFiles())
		{
			/* NOte that some files do not contain matrix, but just a single scalar value inside */
			HashMap<CellIndex, Double> dmlfile;
			HashMap<CellIndex, Double> rfile;
			if (file.endsWith(".scalar")) {
				file = file.replace(".scalar", "");
				dmlfile = readDMLScalarFromHDFS(file);
				rfile = readRScalarFromFS(file);
			}
			else {
				dmlfile = readDMLMatrixFromHDFS(file);
				rfile = readRMatrixFromFS(file);
			}
			TestUtils.compareMatrices(dmlfile, rfile, eps, file+"-DML", file+"-R");
		}
		
	}
	


}
