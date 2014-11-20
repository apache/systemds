/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.applications.descriptivestats;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;


public class UnivariateStatsTest extends AutomatedTestBase
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_DIR = "applications/descriptivestats/";

	private final static double epsilon=0.0000000001;
	private final static int rows1 = 10000;
	private final static int rows2 = 5;
	//private final static int cols = 1;
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
				new String[] {"mean"+".scalar", "std"+".scalar", "se"+".scalar", "var"+".scalar", "cv"+".scalar", 
	              			  /*"har", "geom",*/ 
				  			  "min"+".scalar", "max"+".scalar", "rng"+".scalar", 
				  			  "g1"+".scalar", "se_g1"+".scalar", "g2"+".scalar", "se_g2"+".scalar", 
				  			  "out_minus", "out_plus", "median"+".scalar", "quantile", "iqm"+".scalar"}));
		addTestConfiguration("Categorical", new TestConfiguration(TEST_DIR, "Categorical", 
				new String[] {"Nc", "R"+".scalar", "Pc", "C", "Mode"})); // Indicate some file is scalar
		addTestConfiguration("WeightedCategoricalTest", new TestConfiguration(TEST_DIR, "WeightedCategoricalTest", 
				new String[] {"Nc", "R"+".scalar", "Pc", "C", "Mode"}));
	}
	
	@Test
	public void testScaleWithR() {
	    TestConfiguration config = getTestConfiguration("Scale");
        config.addVariable("rows1", rows1);
        config.addVariable("rows2", rows2);

		// This is for running the junit test the new way, i.e., construct the arguments directly 
		String S_HOME = SCRIPT_DIR + TEST_DIR;	
		fullDMLScriptName = S_HOME + "Scale" + ".dml";
		programArgs = new String[]{"-args",  S_HOME + INPUT_DIR + "vector" , 
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
		fullRScriptName = S_HOME + "Scale" + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + 
		       S_HOME + INPUT_DIR + " " + S_HOME + EXPECTED_DIR;


		loadTestConfiguration(config);

        double[][] vector = getRandomMatrix(rows1, 1, min, max, 0.4, System.currentTimeMillis());
        double[][] prob = getRandomMatrix(rows2, 1, 0, 1, 1, System.currentTimeMillis());

        writeInputMatrix("vector", vector, true);
        writeInputMatrix("prob", prob, true);

		// Expected number of jobs:
		// Reblock - 1 job 
		// While loop iteration - 10 jobs
		// Final output write - 1 job
		//
        //boolean exceptionExpected = false;
		//int expectedNumberOfJobs = 12;
		//runTest(exceptionExpected, null, expectedNumberOfJobs);
		runTest(true, false, null, -1);
		
		runRScript(true);
		//disableOutAndExpectedDeletion();
	
		for(String file: config.getOutputFiles())
		{
			// NOte that some files do not contain matrix, but just a single scalar value inside 
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

		// This is for running the junit test the new way, i.e., construct the arguments directly 
		String S_HOME = SCRIPT_DIR + TEST_DIR;	
		fullDMLScriptName = S_HOME + "WeightedScaleTest" + ".dml";
		programArgs = new String[]{"-args",  S_HOME + INPUT_DIR + "vector" , 
	                        Integer.toString(rows1),
	                         S_HOME + INPUT_DIR + "weight" ,
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
		fullRScriptName = S_HOME + "WeightedScaleTest" + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + 
		       S_HOME + INPUT_DIR + " " + S_HOME + EXPECTED_DIR;

		loadTestConfiguration(config);

		createHelperMatrix();
        double[][] vector = getRandomMatrix(rows1, 1, min, max, 0.4, System.currentTimeMillis());
        double[][] weight = getRandomMatrix(rows1, 1, 1, 10, 1, System.currentTimeMillis());
        OrderStatisticsTest.round(weight);
        double[][] prob = getRandomMatrix(rows2, 1, 0, 1, 1, System.currentTimeMillis());

        writeInputMatrix("vector", vector, true);
        writeInputMatrix("weight", weight, true);
        writeInputMatrix("prob", prob, true);

		//
		// Expected number of jobs:
		// Reblock - 1 job 
		// While loop iteration - 10 jobs
		// Final output write - 1 job
		
		runTest(true, false, null, -1);
		
		runRScript(true);
		//disableOutAndExpectedDeletion();
	
		for(String file: config.getOutputFiles())
		{
			// NOte that some files do not contain matrix, but just a single scalar value inside
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
	public void testCategoricalWithR() {
	
        TestConfiguration config = getTestConfiguration("Categorical");
        config.addVariable("rows1", rows1);
        
		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String C_HOME = SCRIPT_DIR + TEST_DIR;	
		fullDMLScriptName = C_HOME + "Categorical" + ".dml";
		programArgs = new String[]{"-args",  C_HOME + INPUT_DIR + "vector" , 
	                        Integer.toString(rows1),
	                         C_HOME + OUTPUT_DIR + "Nc" , 
	                         C_HOME + OUTPUT_DIR + "R" , 
	                         C_HOME + OUTPUT_DIR + "Pc" ,
	                         C_HOME + OUTPUT_DIR + "C" ,
	                         C_HOME + OUTPUT_DIR + "Mode" };
		fullRScriptName = C_HOME + "Categorical" + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + 
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

		// This is for running the junit test the new way, i.e., construct the arguments directly
		String C_HOME = SCRIPT_DIR + TEST_DIR;	
		fullDMLScriptName = C_HOME + "WeightedCategoricalTest" + ".dml";
		programArgs = new String[]{"-args",  C_HOME + INPUT_DIR + "vector" , 
	                        Integer.toString(rows1),
	                         C_HOME + INPUT_DIR + "weight" , 
	                         C_HOME + OUTPUT_DIR + "Nc" , 
	                         C_HOME + OUTPUT_DIR + "R" , 
	                         C_HOME + OUTPUT_DIR + "Pc" ,
	                         C_HOME + OUTPUT_DIR + "C" ,
	                         C_HOME + OUTPUT_DIR + "Mode" };
		fullRScriptName = C_HOME + "WeightedCategoricalTest" + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + 
		       C_HOME + INPUT_DIR + " " + C_HOME + EXPECTED_DIR;
		
		loadTestConfiguration(config);

		createHelperMatrix();
        double[][] vector = getRandomMatrix(rows1, 1, 1, 10, 1, System.currentTimeMillis());
        OrderStatisticsTest.round(vector);
        double[][] weight = getRandomMatrix(rows1, 1, 1, 10, 1, System.currentTimeMillis());
        OrderStatisticsTest.round(weight);

        writeInputMatrix("vector", vector, true);
        writeInputMatrix("weight", weight, true);
  
        boolean exceptionExpected = false;
		runTest(true, exceptionExpected, null, -1);
		runRScript(true);
		//disableOutAndExpectedDeletion();
	
		for(String file: config.getOutputFiles())
		{
			// NOte that some files do not contain matrix, but just a single scalar value inside
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
}
