/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.applications.descriptivestats;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.api.DMLScript.RUNTIME_PLATFORM;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;


public class UnivariateStatsTest extends AutomatedTestBase
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_DIR = "applications/descriptivestats/";

	private final static double epsilon=0.0000000001;
	private final static int rows1 = 2000;
	private final static int rows2 = 5;
	//private final static int cols = 1;
	private final static int min=-100;
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
	
	// DIV4=divisible by 4; DIV4P1=divisible by 4 plus 1
	private enum SIZE {DIV4 (2000), DIV4P1 (2001), DIV4P2 (2002), DIV4P3 (2003);
						int size=-1; SIZE(int s) { size = s;} 
					}; 
	private enum RANGE {NEG (-255,-2), MIXED(-200,200), POS(2,255);
						double min, max; RANGE(double mn, double mx) { min=mn; max=mx;} 
					};
	private enum SPARSITY { SPARSE(0.3), DENSE(0.8);
						double sparsity; SPARSITY(double sp) {sparsity = sp;}
					};

	// -------------------------------------------------------------------------------------
	@Test
	public void testScale1() {
		testScaleWithR(SIZE.DIV4, RANGE.NEG, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testScale2() {
		testScaleWithR(SIZE.DIV4P1, RANGE.NEG, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testScale3() {
		testScaleWithR(SIZE.DIV4P3, RANGE.NEG, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testScale4() {
		testScaleWithR(SIZE.DIV4P3, RANGE.NEG, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testScale5() {
		testScaleWithR(SIZE.DIV4, RANGE.MIXED, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testScale6() {
		testScaleWithR(SIZE.DIV4P1, RANGE.MIXED, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testScale7() {
		testScaleWithR(SIZE.DIV4P3, RANGE.MIXED, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testScale8() {
		testScaleWithR(SIZE.DIV4P3, RANGE.MIXED, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testScale9() {
		testScaleWithR(SIZE.DIV4, RANGE.POS, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testScale10() {
		testScaleWithR(SIZE.DIV4P1, RANGE.POS, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testScale11() {
		testScaleWithR(SIZE.DIV4P3, RANGE.POS, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testScale12() {
		testScaleWithR(SIZE.DIV4P3, RANGE.POS, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
	// -------------------------------------------------------------------------------------

	@Test
	public void testScale13() {
		testScaleWithR(SIZE.DIV4, RANGE.NEG, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testScale14() {
		testScaleWithR(SIZE.DIV4P1, RANGE.NEG, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testScale15() {
		testScaleWithR(SIZE.DIV4P3, RANGE.NEG, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testScale16() {
		testScaleWithR(SIZE.DIV4P3, RANGE.NEG, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testScale17() {
		testScaleWithR(SIZE.DIV4, RANGE.MIXED, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testScale18() {
		testScaleWithR(SIZE.DIV4P1, RANGE.MIXED, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testScale19() {
		testScaleWithR(SIZE.DIV4P3, RANGE.MIXED, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testScale20() {
		testScaleWithR(SIZE.DIV4P3, RANGE.MIXED, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testScale21() {
		testScaleWithR(SIZE.DIV4, RANGE.POS, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testScale22() {
		testScaleWithR(SIZE.DIV4P1, RANGE.POS, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testScale23() {
		testScaleWithR(SIZE.DIV4P3, RANGE.POS, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testScale24() {
		testScaleWithR(SIZE.DIV4P3, RANGE.POS, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
	// -------------------------------------------------------------------------------------
	@Test
	public void testScale25() {
		testScaleWithR(SIZE.DIV4, RANGE.NEG, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testScale26() {
		testScaleWithR(SIZE.DIV4P1, RANGE.NEG, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testScale27() {
		testScaleWithR(SIZE.DIV4P3, RANGE.NEG, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testScale28() {
		testScaleWithR(SIZE.DIV4P3, RANGE.NEG, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testScale29() {
		testScaleWithR(SIZE.DIV4, RANGE.MIXED, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testScale30() {
		testScaleWithR(SIZE.DIV4P1, RANGE.MIXED, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testScale31() {
		testScaleWithR(SIZE.DIV4P3, RANGE.MIXED, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testScale32() {
		testScaleWithR(SIZE.DIV4P3, RANGE.MIXED, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testScale33() {
		testScaleWithR(SIZE.DIV4, RANGE.POS, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testScale34() {
		testScaleWithR(SIZE.DIV4P1, RANGE.POS, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testScale35() {
		testScaleWithR(SIZE.DIV4P3, RANGE.POS, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testScale36() {
		testScaleWithR(SIZE.DIV4P3, RANGE.POS, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
	// -------------------------------------------------------------------------------------

	@Test
	public void testScale37() {
		testScaleWithR(SIZE.DIV4, RANGE.NEG, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testScale38() {
		testScaleWithR(SIZE.DIV4P1, RANGE.NEG, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testScale39() {
		testScaleWithR(SIZE.DIV4P3, RANGE.NEG, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testScale40() {
		testScaleWithR(SIZE.DIV4P3, RANGE.NEG, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testScale41() {
		testScaleWithR(SIZE.DIV4, RANGE.MIXED, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testScale42() {
		testScaleWithR(SIZE.DIV4P1, RANGE.MIXED, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testScale43() {
		testScaleWithR(SIZE.DIV4P3, RANGE.MIXED, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testScale44() {
		testScaleWithR(SIZE.DIV4P3, RANGE.MIXED, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testScale45() {
		testScaleWithR(SIZE.DIV4, RANGE.POS, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testScale46() {
		testScaleWithR(SIZE.DIV4P1, RANGE.POS, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testScale47() {
		testScaleWithR(SIZE.DIV4P3, RANGE.POS, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testScale48() {
		testScaleWithR(SIZE.DIV4P3, RANGE.POS, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
	// -------------------------------------------------------------------------------------
		
	private void testScaleWithR(SIZE sz, RANGE rng, SPARSITY sp, RUNTIME_PLATFORM rt) {
		
		RUNTIME_PLATFORM oldrt = rtplatform;
		rtplatform = rt;
		
	    TestConfiguration config = getTestConfiguration("Scale");
        config.addVariable("rows1", sz.size);
        config.addVariable("rows2", rows2);

		// This is for running the junit test the new way, i.e., construct the arguments directly 
		String S_HOME = SCRIPT_DIR + TEST_DIR;	
		fullDMLScriptName = S_HOME + "Scale" + ".dml";
		programArgs = new String[]{"-args",  S_HOME + INPUT_DIR + "vector" , 
	                        Integer.toString(sz.size),
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

		long seed1 = System.currentTimeMillis(); 
		long seed2 = System.currentTimeMillis(); 
        double[][] vector = getRandomMatrix(sz.size, 1, rng.min, rng.max, sp.sparsity, seed1);
        double[][] prob = getRandomMatrix(rows2, 1, 0, 1, 1, seed2); 
		System.out.println("seeds: " + seed1 + " " + seed2);

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
		
		rtplatform = oldrt;
	}
	
	// -------------------------------------------------------------------------------------------------------
	
	@Test
	public void testWeightedScale1() {
		testWeightedScaleWithR(SIZE.DIV4, RANGE.NEG, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testWeightedScale2() {
		testWeightedScaleWithR(SIZE.DIV4P1, RANGE.NEG, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testWeightedScale3() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.NEG, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testWeightedScale4() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.NEG, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testWeightedScale5() {
		testWeightedScaleWithR(SIZE.DIV4, RANGE.MIXED, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testWeightedScale6() {
		testWeightedScaleWithR(SIZE.DIV4P1, RANGE.MIXED, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testWeightedScale7() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.MIXED, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testWeightedScale8() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.MIXED, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testWeightedScale9() {
		testWeightedScaleWithR(SIZE.DIV4, RANGE.POS, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testWeightedScale10() {
		testWeightedScaleWithR(SIZE.DIV4P1, RANGE.POS, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testWeightedScale11() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.POS, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testWeightedScale12() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.POS, SPARSITY.SPARSE, RUNTIME_PLATFORM.HYBRID);
	}
	// -------------------------------------------------------------------------------------

	@Test
	public void testWeightedScale13() {
		testWeightedScaleWithR(SIZE.DIV4, RANGE.NEG, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testWeightedScale14() {
		testWeightedScaleWithR(SIZE.DIV4P1, RANGE.NEG, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testWeightedScale15() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.NEG, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testWeightedScale16() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.NEG, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testWeightedScale17() {
		testWeightedScaleWithR(SIZE.DIV4, RANGE.MIXED, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testWeightedScale18() {
		testWeightedScaleWithR(SIZE.DIV4P1, RANGE.MIXED, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testWeightedScale19() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.MIXED, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testWeightedScale20() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.MIXED, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testWeightedScale21() {
		testWeightedScaleWithR(SIZE.DIV4, RANGE.POS, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testWeightedScale22() {
		testWeightedScaleWithR(SIZE.DIV4P1, RANGE.POS, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testWeightedScale23() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.POS, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
		
	@Test
	public void testWeightedScale24() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.POS, SPARSITY.DENSE, RUNTIME_PLATFORM.HYBRID);
	}
	// -------------------------------------------------------------------------------------
	@Test
	public void testWeightedScale25() {
		testWeightedScaleWithR(SIZE.DIV4, RANGE.NEG, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testWeightedScale26() {
		testWeightedScaleWithR(SIZE.DIV4P1, RANGE.NEG, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testWeightedScale27() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.NEG, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testWeightedScale28() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.NEG, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testWeightedScale29() {
		testWeightedScaleWithR(SIZE.DIV4, RANGE.MIXED, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testWeightedScale30() {
		testWeightedScaleWithR(SIZE.DIV4P1, RANGE.MIXED, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testWeightedScale31() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.MIXED, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testWeightedScale32() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.MIXED, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testWeightedScale33() {
		testWeightedScaleWithR(SIZE.DIV4, RANGE.POS, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testWeightedScale34() {
		testWeightedScaleWithR(SIZE.DIV4P1, RANGE.POS, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testWeightedScale35() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.POS, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testWeightedScale36() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.POS, SPARSITY.SPARSE, RUNTIME_PLATFORM.HADOOP);
	}
	// -------------------------------------------------------------------------------------

	@Test
	public void testWeightedScale37() {
		testWeightedScaleWithR(SIZE.DIV4, RANGE.NEG, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testWeightedScale38() {
		testWeightedScaleWithR(SIZE.DIV4P1, RANGE.NEG, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testWeightedScale39() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.NEG, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testWeightedScale40() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.NEG, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testWeightedScale41() {
		testWeightedScaleWithR(SIZE.DIV4, RANGE.MIXED, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testWeightedScale42() {
		testWeightedScaleWithR(SIZE.DIV4P1, RANGE.MIXED, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testWeightedScale43() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.MIXED, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testWeightedScale44() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.MIXED, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testWeightedScale45() {
		testWeightedScaleWithR(SIZE.DIV4, RANGE.POS, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testWeightedScale46() {
		testWeightedScaleWithR(SIZE.DIV4P1, RANGE.POS, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testWeightedScale47() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.POS, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
		
	@Test
	public void testWeightedScale48() {
		testWeightedScaleWithR(SIZE.DIV4P3, RANGE.POS, SPARSITY.DENSE, RUNTIME_PLATFORM.HADOOP);
	}
	// -------------------------------------------------------------------------------------
		
	
	
	private void testWeightedScaleWithR(SIZE sz, RANGE rng, SPARSITY sp, RUNTIME_PLATFORM rt) {
		
		RUNTIME_PLATFORM oldrt = rtplatform;
		rtplatform = rt;
		
        TestConfiguration config = getTestConfiguration("WeightedScaleTest");
        config.addVariable("rows1", sz.size);
        config.addVariable("rows2", rows2);

		// This is for running the junit test the new way, i.e., construct the arguments directly 
		String S_HOME = SCRIPT_DIR + TEST_DIR;	
		fullDMLScriptName = S_HOME + "WeightedScaleTest" + ".dml";
		programArgs = new String[]{"-args",  S_HOME + INPUT_DIR + "vector" , 
	                        Integer.toString(sz.size),
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
        double[][] vector = getRandomMatrix(sz.size, 1, rng.min, rng.max, sp.sparsity, System.currentTimeMillis());
        double[][] weight = getRandomMatrix(sz.size, 1, 1, 10, 1, System.currentTimeMillis());
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
		
		//reset runtime platform
		rtplatform = oldrt;
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
