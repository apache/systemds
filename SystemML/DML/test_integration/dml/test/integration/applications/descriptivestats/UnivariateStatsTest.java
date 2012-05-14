package dml.test.integration.applications.descriptivestats;

import java.util.HashMap;

import org.junit.Test;

import dml.runtime.matrix.io.MatrixValue.CellIndex;
import dml.test.integration.AutomatedTestBase;
import dml.test.integration.TestConfiguration;
import dml.test.utils.TestUtils;

public class UnivariateStatsTest extends AutomatedTestBase{
	private final static String TEST_DIR = "applications/descriptivestats/";

	private final static double epsilon=0.0000000001;
	private final static int rows1 = 10000;
	private final static int rows2 = 5;
	private final static int cols = 1;
	private final static int min=0;
	private final static int max=100;

	@Override
	public void setUp() {
		addTestConfiguration("Scale", new TestConfiguration(TEST_DIR, "Scale", 
				new String[] {"mean"+".scalar", "std"+".scalar", "se"+".scalar", "var"+".scalar", "cv"+".scalar", 
				              /*"har", "geom",*/ 
						      "min"+".scalar", "max"+".scalar", "rng"+".scalar", 
						      "g1"+".scalar", "se_g1"+".scalar", "g2"+".scalar", "se_g2"+".scalar", 
						      "out_minus", "out_plus", "median"+".scalar", "quantile", "iqm"+".scalar"}));
		addTestConfiguration("WeightedScaleTest", new TestConfiguration(TEST_DIR, "WeightedScaleTest", 
				new String[] {"mean_weight", "std_weight", "se_weight", "var_weight", "cv_weight", "har_weight", /*"geom_weight",*/ 
				"min_weight", "max_weight", "rng_weight", "g1_weight", "se_g1_weight", "g2_weight", "se_g2_weight", 
				"out_minus_weight", "out_plus_weight", "median_weight", "quantile_weight", "iqm_weight"}));
		addTestConfiguration("Categorical", new TestConfiguration(TEST_DIR, "Categorical", 
				new String[] {"Nc", "R"+".scalar", "Pc", "C", "Mode"})); // Indicate some file is scalar
		addTestConfiguration("WeightedCategoricalTest", new TestConfiguration(TEST_DIR, "WeightedCategoricalTest", 
				new String[] {"Nc_weight", "R_weight", "Pc_weight", "C_weight", "Mode_weight"}));
	}
	
	@Test
	public void testScaleWithR() {
	    TestConfiguration config = getTestConfiguration("Scale");
        config.addVariable("rows1", rows1);
        config.addVariable("rows2", rows2);

		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String S_HOME = SCRIPT_DIR + TEST_DIR;	
		dmlArgs = new String[]{"-f", S_HOME + "Scale" + ".dml",
	               "-args",  S_HOME + INPUT_DIR + "vector" , 
	                        Integer.toString(rows1),
	                         S_HOME + INPUT_DIR + "prob" ,
	                        Integer.toString(rows2),
	                         S_HOME + OUTPUT_DIR + "mean" , 
	                         S_HOME + OUTPUT_DIR + "std" , 
	                         S_HOME + OUTPUT_DIR + "se" ,
	                         S_HOME + OUTPUT_DIR + "var" ,
	                         S_HOME + OUTPUT_DIR + "cv" ,
	                         S_HOME + OUTPUT_DIR + "min" ,
	                         S_HOME + OUTPUT_DIR + "max" ,
	                         S_HOME + OUTPUT_DIR + "rng" ,
	                         S_HOME + OUTPUT_DIR + "g1" ,
	                         S_HOME + OUTPUT_DIR + "se_g1" ,
	                         S_HOME + OUTPUT_DIR + "g2" ,
	                         S_HOME + OUTPUT_DIR + "se_g2" ,
	                         S_HOME + OUTPUT_DIR + "median" ,
	                         S_HOME + OUTPUT_DIR + "iqm" ,
	                         S_HOME + OUTPUT_DIR + "out_minus" ,
	                         S_HOME + OUTPUT_DIR + "out_plus" ,
	                         S_HOME + OUTPUT_DIR + "quantile" };
		dmlArgsDebug = new String[]{"-f", S_HOME + "Scale" + ".dml", "-d",
	               "-args",  S_HOME + INPUT_DIR + "vector" , 
	                        Integer.toString(rows1),
	                         S_HOME + INPUT_DIR + "prob" ,
	                        Integer.toString(rows2),
	                         S_HOME + OUTPUT_DIR + "mean" , 
	                         S_HOME + OUTPUT_DIR + "std" , 
	                         S_HOME + OUTPUT_DIR + "se" ,
	                         S_HOME + OUTPUT_DIR + "var" ,
	                         S_HOME + OUTPUT_DIR + "cv" ,
	                         S_HOME + OUTPUT_DIR + "min" ,
	                         S_HOME + OUTPUT_DIR + "max" ,
	                         S_HOME + OUTPUT_DIR + "rng" ,
	                         S_HOME + OUTPUT_DIR + "g1" ,
	                         S_HOME + OUTPUT_DIR + "se_g1" ,
	                         S_HOME + OUTPUT_DIR + "g2" ,
	                         S_HOME + OUTPUT_DIR + "se_g2" ,
	                         S_HOME + OUTPUT_DIR + "median" ,
	                         S_HOME + OUTPUT_DIR + "iqm" ,
	                         S_HOME + OUTPUT_DIR + "out_minus" ,
	                         S_HOME + OUTPUT_DIR + "out_plus" ,
	                         S_HOME + OUTPUT_DIR + "quantile" };
		rCmd = "Rscript" + " " + S_HOME + "Scale" + ".R" + " " + 
		       S_HOME + INPUT_DIR + " " + S_HOME + EXPECTED_DIR;

		loadTestConfiguration(config);

        double[][] vector = getRandomMatrix(rows1, 1, min, max, 0.4, System.currentTimeMillis());
        double[][] prob = getRandomMatrix(rows2, 1, 0, 1, 1, System.currentTimeMillis());

        writeInputMatrix("vector", vector, true);
        writeInputMatrix("prob", prob, true);

		/*
		 * Expected number of jobs:
		 * Reblock - 1 job 
		 * While loop iteration - 10 jobs
		 * Final output write - 1 job
		 */
        //boolean exceptionExpected = false;
		//int expectedNumberOfJobs = 12;
		//runTest(exceptionExpected, null, expectedNumberOfJobs);
		runTest(true, false, null, -1);
		
		runRScript(true);
		//disableOutAndExpectedDeletion();
	
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
			TestUtils.compareMatrices(dmlfile, rfile, epsilon, file+"-DML", file+"-R");
		}
	}
	
	@Test
	public void testWeightedScaleWithR() {
		
        TestConfiguration config = getTestConfiguration("WeightedScaleTest");
        config.addVariable("rows1", rows1);
        config.addVariable("rows2", rows2);

		loadTestConfiguration(config);

		createHelperMatrix();
        double[][] vector = getRandomMatrix(rows1, 1, min, max, 0.4, 10);//System.currentTimeMillis());
        double[][] weight = getRandomMatrix(rows1, 1, 1, 10, 1, 20);//System.currentTimeMillis());
        OrderStatisticsTest.round(weight);
        double[][] prob = getRandomMatrix(rows2, 1, 0, 1, 1, System.currentTimeMillis());

        writeInputMatrix("vector", vector, true);
        writeInputMatrix("weight", weight, true);
        writeInputMatrix("prob", prob, true);

		/*
		 * Expected number of jobs:
		 * Reblock - 1 job 
		 * While loop iteration - 10 jobs
		 * Final output write - 1 job
		 */
        //boolean exceptionExpected = false;
		//int expectedNumberOfJobs = 12;
		//runTest(exceptionExpected, null, expectedNumberOfJobs);
		runTest();
		
		runRScript();
		//disableOutAndExpectedDeletion();
	
		for(String file: config.getOutputFiles())
		{
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS(file);
			HashMap<CellIndex, Double> rfile = readRMatrixFromFS(file);
			//System.out.println(file+"-DML: "+dmlfile);
			//System.out.println(file+"-R: "+rfile);
			TestUtils.compareMatrices(dmlfile, rfile, epsilon, file+"-DML", file+"-R");
		}
	}
	
	@Test
	public void testCategoricalWithR() {
	
        TestConfiguration config = getTestConfiguration("Categorical");
        config.addVariable("rows1", rows1);
        
		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String C_HOME = SCRIPT_DIR + TEST_DIR;	
		dmlArgs = new String[]{"-f", C_HOME + "Categorical" + ".dml",
	               "-args",  C_HOME + INPUT_DIR + "vector" , 
	                        Integer.toString(rows1),
	                         C_HOME + OUTPUT_DIR + "Nc" , 
	                         C_HOME + OUTPUT_DIR + "R" , 
	                         C_HOME + OUTPUT_DIR + "Pc" ,
	                         C_HOME + OUTPUT_DIR + "C" ,
	                         C_HOME + OUTPUT_DIR + "Mode" };
		dmlArgsDebug = new String[]{"-f", C_HOME + "Categorical" + ".dml", "-d",
	               "-args",  C_HOME + INPUT_DIR + "vector" , 
	                        Integer.toString(rows1),
	                         C_HOME + OUTPUT_DIR + "Nc" , 
	                         C_HOME + OUTPUT_DIR + "R" , 
	                         C_HOME + OUTPUT_DIR + "Pc" ,
	                         C_HOME + OUTPUT_DIR + "C" ,
	                         C_HOME + OUTPUT_DIR + "Mode" };
		rCmd = "Rscript" + " " + C_HOME + "Categorical" + ".R" + " " + 
		       C_HOME + INPUT_DIR + " " + C_HOME + EXPECTED_DIR;

		loadTestConfiguration(config);

        double[][] vector = getRandomMatrix(rows1, 1, 1, 10, 1, System.currentTimeMillis());
        OrderStatisticsTest.round(vector);
        
        writeInputMatrix("vector", vector, true);

		/*
		 * Expected number of jobs:
		 * Reblock - 1 job 
		 * While loop iteration - 10 jobs
		 * Final output write - 1 job
		 */
        boolean exceptionExpected = false;
		int expectedNumberOfJobs = 12;
		runTest(true, exceptionExpected, null, expectedNumberOfJobs);
		
		runRScript(true);
		//disableOutAndExpectedDeletion();
	
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
			TestUtils.compareMatrices(dmlfile, rfile, epsilon, file+"-DML", file+"-R");
		}
	}
	
	@Test
	public void testWeightedCategoricalWithR() {
	
        TestConfiguration config = getTestConfiguration("WeightedCategoricalTest");
        config.addVariable("rows1", rows1);

		loadTestConfiguration(config);

		createHelperMatrix();
        double[][] vector = getRandomMatrix(rows1, 1, 1, 10, 1, System.currentTimeMillis());
        OrderStatisticsTest.round(vector);
        double[][] weight = getRandomMatrix(rows1, 1, 1, 10, 1, System.currentTimeMillis());
        OrderStatisticsTest.round(weight);

        writeInputMatrix("vector", vector, true);
        writeInputMatrix("weight", weight, true);
  
		/*
		 * Expected number of jobs:
		 * Reblock - 1 job 
		 * While loop iteration - 10 jobs
		 * Final output write - 1 job
		 */
        //boolean exceptionExpected = false;
		//int expectedNumberOfJobs = 12;
		//runTest(exceptionExpected, null, expectedNumberOfJobs);
		runTest();
		
		runRScript();
		//disableOutAndExpectedDeletion();
	
		for(String file: config.getOutputFiles())
		{
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS(file);
			HashMap<CellIndex, Double> rfile = readRMatrixFromFS(file);
		//	System.out.println(file+"-DML: "+dmlfile);
		//	System.out.println(file+"-R: "+rfile);
			TestUtils.compareMatrices(dmlfile, rfile, epsilon, file+"-DML", file+"-R");
		}
	}
}
