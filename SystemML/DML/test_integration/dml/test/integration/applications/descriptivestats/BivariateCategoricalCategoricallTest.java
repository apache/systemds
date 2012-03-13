package dml.test.integration.applications.descriptivestats;

import java.util.HashMap;

import org.junit.Test;

import dml.runtime.matrix.io.MatrixValue.CellIndex;
import dml.test.integration.AutomatedTestBase;
import dml.test.integration.TestConfiguration;
import dml.test.utils.TestUtils;

public class BivariateCategoricalCategoricallTest extends AutomatedTestBase {

	private final static String TEST_DIR = "applications/descriptivestats/";
	private final static String TEST_NOMINAL_NOMINAL = "CategoricalCategorical";
	private final static String TEST_NOMINAL_NOMINAL_WEIGHTS = "CategoricalCategoricalWithWeightsTest";

	private final static double eps = 1e-9;
	private int rows = 10000;  // # of rows in each vector
	private int ncatA = 100;   // # of categories in A
	private int ncatB = 150;   // # of categories in B
	private int maxW = 100;    // maximum weight
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_NOMINAL_NOMINAL, new TestConfiguration(TEST_DIR, TEST_NOMINAL_NOMINAL, 
				new String[] { "PValue"+".scalar", "CramersV"+".scalar" }));
		addTestConfiguration(TEST_NOMINAL_NOMINAL_WEIGHTS, new TestConfiguration(TEST_DIR, "CategoricalCategoricalWithWeightsTest", new String[] { "outPValue", "outCramersV" }));
	}
	
	@Deprecated 
	class CTable {
		double [][]table;
		int R, S;
		double W;
		
		CTable(int r, int s) {
			R = r;
			S = s;
			W = 0;
			table = new double[r][s];
			for (int i=0; i < R; i++ ) {
				for ( int j=0; j < S; j++ ) {
					table[i][j] = 0;
				}
			}
		}
	}
	
	/*
	 * Method to compute the contingency table
	 */
	CTable computeCTable(double [][]x, double [][]y, double [][]w, int rows) {
		int R=0, S=0;
		
		// compute dimensions of contingency table
		for ( int i=0; i < rows; i++ ) {
			if ( (int)x[i][0] > R )
				R = (int) x[i][0];
			if ( (int)y[i][0] > S )
				S = (int) y[i][0];
		}
		CTable ct = new CTable(R,S);
		
		boolean weights = (w != null);
		// construct the contingency table
		if ( weights ) {
			for ( int i=0; i < rows; i++ ) {
				ct.table[(int)x[i][0]-1][(int)y[i][0]-1] += (int) w[i][0];
				ct.W += w[i][0]; 
			}
		}
		else {
			for ( int i=0; i < rows; i++ ) {
				ct.table[(int)x[i][0]-1][(int)y[i][0]-1]++;
				ct.W ++;
			}
		}
		return ct;
	}
	
	@SuppressWarnings("unused")
	@Deprecated 
	private double computeChiSquared(CTable ct, int rows) {
		
		double[][] E = new double[ct.R][ct.S];
		for (int i=0; i < ct.R; i++ ) {
			for ( int j=0; j < ct.S; j++ ) {
				E[i][j] = 0;
			}
		}
		
		
		// compute row-wise and col-wise sums of the table
		double[] rowSums = new double[ct.R];
		double[] colSums = new double[ct.S];
		for ( int i=0; i < ct.R; i++ ) 
			rowSums[i] = 0;
		for ( int j=0; j < ct.S; j++ )
			colSums[j] = 0;
		
		for ( int i=0; i < ct.R; i++ ) {
			for ( int j=0; j < ct.S; j++ ) {
				rowSums[i] += ct.table[i][j];
				colSums[j] += ct.table[i][j];
			}
		}
		
		double chiSquared = 0.0, Eij=0;
		for ( int i=0; i < ct.R; i++ ) {
			for ( int j=0; j < ct.S; j++ ) {
				Eij = (rowSums[i]*colSums[j])/ct.W; // outer product of two vectors
				chiSquared += ((ct.table[i][j]-Eij)*(ct.table[i][j]-Eij)/Eij);
			}
		}
		
		return chiSquared;
	}
	
	@SuppressWarnings("unused")
	@Deprecated 
	private double computeCramersV(CTable ct, double chiSq) {
		return Math.sqrt(chiSq/(ct.W * (Math.min(ct.R, ct.S)-1) ));
	}
	
	@Test
	public void testCategoricalCategorical() {
		
		TestConfiguration config = getTestConfiguration(TEST_NOMINAL_NOMINAL);
		
		config.addVariable("rows", rows);

		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String CC_HOME = SCRIPT_DIR + TEST_DIR;
		dmlArgs = new String[]{"-f", CC_HOME + TEST_NOMINAL_NOMINAL + ".dml",
				               "-args", "\"" + CC_HOME + INPUT_DIR + "A" + "\"", 
				                        Integer.toString(rows),
				                        "\"" + CC_HOME + INPUT_DIR + "B" + "\"", 
				                        "\"" + CC_HOME + OUTPUT_DIR + "PValue" + "\"", 
				                        "\"" + CC_HOME + OUTPUT_DIR + "CramersV" + "\""};
		dmlArgsDebug = new String[]{"-f", CC_HOME + TEST_NOMINAL_NOMINAL + ".dml", "-d",
	                                "-args", "\"" + CC_HOME + INPUT_DIR + "A" + "\"", 
	                                         Integer.toString(rows),
	                                         "\"" + CC_HOME + INPUT_DIR + "B" + "\"", 
	                                         "\"" + CC_HOME + OUTPUT_DIR + "PValue" + "\"", 
	                                         "\"" + CC_HOME + OUTPUT_DIR + "CramersV" + "\""};
		
		rCmd = "Rscript" + " " + CC_HOME + TEST_NOMINAL_NOMINAL + ".R" + " " + 
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
			weight[i][0]=Math.floor(weight[i][0]);
	}

	@Test
	public void testCategoricalCategoricalWithWeights() {

		TestConfiguration config = getTestConfiguration(TEST_NOMINAL_NOMINAL_WEIGHTS);
		
		config.addVariable("rows", rows);

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
        CTable ct = computeCTable(A,B,WM,rows);
		double chiSquared = computeChiSquared(ct, rows);
		double cramersV = computeCramersV(ct, chiSquared);
		
		writeExpectedHelperMatrix("outChiSquared", chiSquared);
		writeExpectedHelperMatrix("outCramersV", cramersV);
		*/
        
		/*
		 * Expected number of jobs:
		 * Mean etc - 2 jobs (reblock & gmr)
		 * Cov etc - 2 jobs
		 * Final output write - 1 job
		 */
		//boolean exceptionExpected = false;
		//int expectedNumberOfJobs = 5;
		//runTest(exceptionExpected, null, expectedNumberOfJobs);
		runTest();
		runRScript();
		
		for(String file: config.getOutputFiles())
		{
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS(file);
			HashMap<CellIndex, Double> rfile = readRMatrixFromFS(file);
			//System.out.println(file+"-DML: "+dmlfile);
			//System.out.println(file+"-R: "+rfile);
			TestUtils.compareMatrices(dmlfile, rfile, eps, file+"-DML", file+"-R");
		}
		
	}
	

}
