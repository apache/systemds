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
		addTestConfiguration("ScaleTest", new TestConfiguration(TEST_DIR, "ScaleTest", 
				new String[] {"mean", "std", "se", "var", "cv", "har", /*"geom",*/ "min", "max", "rng", 
				"g1", "se_g1", "g2", "se_g2", "out_minus", "out_plus", "median", "quantile", "iqm"}));
		addTestConfiguration("WeightedScaleTest", new TestConfiguration(TEST_DIR, "WeightedScaleTest", 
				new String[] {"mean_weight", "std_weight", "se_weight", "var_weight", "cv_weight", "har_weight", /*"geom_weight",*/ 
				"min_weight", "max_weight", "rng_weight", "g1_weight", "se_g1_weight", "g2_weight", "se_g2_weight", 
				"out_minus_weight", "out_plus_weight", "median_weight", "quantile_weight", "iqm_weight"}));
		addTestConfiguration("CategoricalTest", new TestConfiguration(TEST_DIR, "CategoricalTest", 
				new String[] {"Nc", "R", "Pc", "C", "Mode"}));
		addTestConfiguration("WeightedCategoricalTest", new TestConfiguration(TEST_DIR, "WeightedCategoricalTest", 
				new String[] {"Nc_weight", "R_weight", "Pc_weight", "C_weight", "Mode_weight"}));
	}
	@Test
	public void testScaleWithR() {
	    TestConfiguration config = getTestConfiguration("ScaleTest");
        config.addVariable("rows1", rows1);
        config.addVariable("rows2", rows2);

		loadTestConfiguration(config);

		createHelperMatrix();
        double[][] vector = getRandomMatrix(rows1, 1, min, max, 0.4, System.currentTimeMillis());
     //   double[][] weight = getRandomMatrix(rows1, 1, 1, 10, 1, System.currentTimeMillis());
      //  OrderStatisticsTest.round(weight);
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
	
        TestConfiguration config = getTestConfiguration("CategoricalTest");
        config.addVariable("rows1", rows1);

		loadTestConfiguration(config);

		createHelperMatrix();
        double[][] vector = getRandomMatrix(rows1, 1, 1, 10, 1, System.currentTimeMillis());
        OrderStatisticsTest.round(vector);
        
        writeInputMatrix("vector", vector, true);

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
