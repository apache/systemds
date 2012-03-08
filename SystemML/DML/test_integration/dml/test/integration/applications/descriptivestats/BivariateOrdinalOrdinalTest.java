package dml.test.integration.applications.descriptivestats;

import org.junit.Test;

import dml.test.integration.AutomatedTestBase;
import dml.test.integration.TestConfiguration;

public class BivariateOrdinalOrdinalTest extends AutomatedTestBase {

	private final static String TEST_DIR = "applications/descriptivestats/";
	private final static String TEST_ORDINAL_ORDINAL = "OrdinalOrdinal";
	private final static String TEST_ORDINAL_ORDINAL_WEIGHTS = "OrdinalOrdinalWithWeightsTest";

	private final static double eps = 1e-9;
	private final static int rows = 10000;
	private final static int ncatA = 100; // # of categories in A
	private final static int ncatB = 200; // # of categories in B
	private int maxW = 100;    // maximum weight
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_ORDINAL_ORDINAL, new TestConfiguration(TEST_DIR, TEST_ORDINAL_ORDINAL, new String[] { "outSpearman" }));
		addTestConfiguration(TEST_ORDINAL_ORDINAL_WEIGHTS, new TestConfiguration(TEST_DIR, "OrdinalOrdinalWithWeightsTest", new String[] { "outSpearman" }));
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
	@Deprecated
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
	
	@Deprecated
	double computeSpearman(double[][] ctable, int rows, int cols) {
		
		double [] rowSums = new double[rows];
		double [] colSums = new double[cols];
		double [] rowScores = new double[rows];
		double [] colScores = new double[cols];
		double totalWeight = 0.0;
		
		for ( int i=0; i < rows; i++ ) {
			rowSums[i] = rowScores[i] = 0.0;
		}
		for ( int j=0; j < cols; j++ ) {
			colSums[j] = colScores[j] = 0;
		}
		
		for ( int i=0; i < rows; i++ ) {
			for ( int j=0; j < cols; j++ ) {
				rowSums[i] += ctable[i][j];
				colSums[j] += ctable[i][j];
				totalWeight += ctable[i][j]; 
			}
		}
		
		double prefix_sum=0.0;
		for ( int i=0; i < rows; i++ ) {
			rowScores[i] = prefix_sum + (rowSums[i]+1)/2;
			prefix_sum += rowSums[i];
		}
		
		prefix_sum=0.0;
		for ( int j=0; j < cols; j++ ) {
			colScores[j] = prefix_sum + (colSums[j]+1)/2;
			prefix_sum += colSums[j];
		}
		
		double Rx = 0.0, Ry = 0.0;
		for ( int i=0; i < rows; i++ ) {
			Rx += rowSums[i]*rowScores[i];
		}
		for ( int j=0; j < cols; j++ ) {
			Ry += colSums[j]*colScores[j];
		}
		Rx = Rx/(double)totalWeight;
		Ry = Ry/(double)totalWeight;
		
		double VRx=0.0, VRy=0.0;
		for ( int i=0; i < rows; i++ ) {
			VRx += rowSums[i] * ((rowScores[i]-Rx)*(rowScores[i]-Rx));
		}
		VRx = VRx/(double)(totalWeight-1);
		
		for ( int j=0; j < cols; j++ ) {
			VRy += colSums[j] * ((colScores[j]-Ry)*(colScores[j]-Ry));
		}
		VRy = VRy/(double)(totalWeight-1);
		
		double CRxRy = 0.0;
		for ( int i=0; i < rows; i++ ) {
			for ( int j=0; j < cols; j++ ) {
				CRxRy = ctable[i][j] * (rowScores[i]-Rx) * (colScores[j]-Ry);
			}
		}
		CRxRy = CRxRy / (double)(totalWeight-1);
		
		double spearman = CRxRy/(Math.sqrt(VRx) * Math.sqrt(VRy));
		
		return spearman;
	}
	

	@Test
	public void testOrdinalOrdinal() {
		TestConfiguration config = getTestConfiguration(TEST_ORDINAL_ORDINAL);
		
		config.addVariable("rows", rows);
		
		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String OO_HOME = SCRIPT_DIR + TEST_DIR;	
		dmlArgs = new String[]{"-f", OO_HOME + TEST_ORDINAL_ORDINAL + ".dml",
	               "-args", "\"" + OO_HOME + INPUT_DIR + "A" + "\"", 
	                        Integer.toString(rows),
	                        "\"" + OO_HOME + INPUT_DIR + "B" + "\"", 
	                        "\"" + OO_HOME + OUTPUT_DIR + "outSpearman" + "\""};
		dmlArgsDebug = new String[]{"-f", OO_HOME + TEST_ORDINAL_ORDINAL + ".dml", "-d", 
	               "-args", "\"" + OO_HOME + INPUT_DIR + "A" + "\"", 
	                        Integer.toString(rows),
	                        "\"" + OO_HOME + INPUT_DIR + "B" + "\"", 
	                        "\"" + OO_HOME + OUTPUT_DIR + "outSpearman" + "\""};
		rCmd = "Rscript" + " " + OO_HOME + TEST_ORDINAL_ORDINAL + ".R" + " " + 
		       OO_HOME + INPUT_DIR + " " + OO_HOME + EXPECTED_DIR;

		loadTestConfiguration(config);

        double[][] A = getRandomMatrix(rows, 1, 1, ncatA, 1, System.currentTimeMillis());
        double[][] B = getRandomMatrix(rows, 1, 1, ncatB, 1, System.currentTimeMillis()+1);
        round(A);
        round(B);
        
		writeInputMatrix("A", A, true);
		writeInputMatrix("B", B, true);
        createHelperMatrix();
        
        CTable ct = computeCTable(A,B,null,rows);
		double spearman = computeSpearman(ct.table, ct.R, ct.S);
		writeExpectedHelperMatrix("outSpearman", spearman);
        
		boolean exceptionExpected = false;
		/*
		 * Expected number of jobs:
		 * Reblock - 1 job 
		 * While loop iteration - 7 jobs
		 * Final output write - 1 job
		 */
		// int expectedNumberOfJobs = 5;
		runTest(true, exceptionExpected, null, -1);
		
		compareResults(eps);

		/*
		runRScript(true);
		for(String file: config.getOutputFiles())
		{
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS(file);
			HashMap<CellIndex, Double> rfile = readRMatrixFromFS(file);
			TestUtils.compareMatrices(dmlfile, rfile, eps, file+"-DML", file+"-R");
		}
		*/
	}
	
	private void round(double[][] weight) {
		for(int i=0; i<weight.length; i++)
			weight[i][0]=Math.floor(weight[i][0]);
	}

	@Test
	public void testOrdinalOrdinalWithWeights() {
		TestConfiguration config = getTestConfiguration(TEST_ORDINAL_ORDINAL_WEIGHTS);
		
		config.addVariable("rows", rows);

		loadTestConfiguration(config);

        double[][] A = getRandomMatrix(rows, 1, 1, ncatA, 1, System.currentTimeMillis());
        double[][] B = getRandomMatrix(rows, 1, 1, ncatB, 1, System.currentTimeMillis()+1);
        double[][] WM = getRandomMatrix(rows, 1, 1, maxW, 1, System.currentTimeMillis());
        round(A);
        round(B);
        round(WM);
        
		writeInputMatrix("A", A, true);
		writeInputMatrix("B", B, true);
		writeInputMatrix("WM", WM, true);
		
        createHelperMatrix();
        
        CTable ct = computeCTable(A,B,WM,rows);
		double spearman = computeSpearman(ct.table, ct.R, ct.S);
		writeExpectedHelperMatrix("outSpearman", spearman);
		
		boolean exceptionExpected = false;
		/*
		 * Expected number of jobs:
		 * Mean etc - 2 jobs (reblock & gmr)
		 * Cov etc - 2 jobs
		 * Final output write - 1 job
		 */
		//int expectedNumberOfJobs = 5;
		//runTest(exceptionExpected, null, expectedNumberOfJobs);
		runTest();
		
		/*
		runRScript();
		for(String file: config.getOutputFiles())
		{
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS(file);
			HashMap<CellIndex, Double> rfile = readRMatrixFromFS(file);
			TestUtils.compareMatrices(dmlfile, rfile, eps, file+"-DML", file+"-R");
		}
		*/
		compareResults(eps);

	}
	

}
