package dml.test.integration.applications.descriptivestats;

import java.util.HashMap;

import org.junit.Test;

import dml.runtime.matrix.io.MatrixValue.CellIndex;
import dml.test.integration.AutomatedTestBase;
import dml.test.integration.TestConfiguration;
import dml.test.utils.TestUtils;

public class BivariateScaleScaleTest extends AutomatedTestBase {

	private final static String TEST_DIR = "applications/descriptivestats/";
	private final static String TEST_SCALE_SCALE = "ScaleScalePearsonRTest";
	private final static String TEST_SCALE_SCALE_WEIGHTS = "ScaleScalePearsonRWithWeightsTest";

	private final static double eps = 1e-10;
	
	private final static int rows = 100000;      // # of rows in each vector
	private final static double minVal=0;       // minimum value in each vector 
	private final static double maxVal=10000;    // maximum value in each vector 
	private int maxW = 1000;    // maximum weight
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_SCALE_SCALE, new TestConfiguration(TEST_DIR, "ScaleScalePearsonRTest", new String[] { "outPearsonR" }));
		addTestConfiguration(TEST_SCALE_SCALE_WEIGHTS, new TestConfiguration(TEST_DIR, "ScaleScalePearsonRWithWeightsTest", new String[] { "outPearsonR" }));
	}
	
	@Deprecated
	double computePearsonR(double [][]x, double [][]y, double [][]w, int rows) {
		double xsum=0.0, ysum=0.0, wsum=0.0;
		
		boolean weights = (w != null);
		
		for (int i=0; i < rows; i++) {
			if ( weights ) {
				xsum += (x[i][0]*w[i][0]);
				ysum += (y[i][0]*w[i][0]);
				wsum += w[i][0];
			}
			else {
				xsum += x[i][0];
				ysum += y[i][0];
				wsum++;
			}
		}
		
		double xbar = xsum/wsum;
		double ybar = ysum/wsum;
		
		double xdiff=0.0, ydiff=0.0, xydiff=0.0;
		
		for ( int i=0; i < rows; i++ ) {
			if ( weights ) {
				xdiff += Math.pow(x[i][0]-xbar, 2)*w[i][0];
				ydiff += Math.pow(y[i][0]-ybar, 2)*w[i][0];
				xydiff += (x[i][0]-xbar)*(y[i][0]-ybar)*w[i][0];
			}
			else {
				xdiff += Math.pow(x[i][0]-xbar, 2);
				ydiff += Math.pow(y[i][0]-ybar, 2);
				xydiff += (x[i][0]-xbar)*(y[i][0]-ybar);
			}
		}
		
		double sigmax = Math.sqrt(xdiff/(wsum-1));
		double sigmay = Math.sqrt(ydiff/(wsum-1));
		double covxy = xydiff/(wsum-1);
		
		double R = covxy / (sigmax * sigmay);
		
		return R;
	}
	
	@Test
	public void testPearsonR() {

		TestConfiguration config = getTestConfiguration(TEST_SCALE_SCALE);
		
		config.addVariable("rows", rows);

		loadTestConfiguration(config);

		long seed = System.currentTimeMillis();
		//System.out.println("Seed = " + seed);
        double[][] X = getRandomMatrix(rows, 1, minVal, maxVal, 0.1, seed);
        double[][] Y = getRandomMatrix(rows, 1, minVal, maxVal, 0.1, seed+1);

		writeInputMatrix("X", X, true);
		writeInputMatrix("Y", Y, true);
        createHelperMatrix();

		/*
        double PearsonR = 0.0;
		PearsonR = computePearsonR(X,Y, null, rows);
        writeExpectedHelperMatrix("outPearsonR", PearsonR);
		*/
        
		boolean exceptionExpected = false;
		/*
		 * Expected number of jobs:
		 */
		int expectedNumberOfJobs = 5;
		runTest(exceptionExpected, null, expectedNumberOfJobs);
		runRScript();
		
		for(String file: config.getOutputFiles())
		{
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS(file);
			HashMap<CellIndex, Double> rfile = readRMatrixFromFS(file);
			TestUtils.compareMatrices(dmlfile, rfile, eps, file+"-DML", file+"-R");
		}
	}
	
	private void round(double[][] weight) {
		for(int i=0; i<weight.length; i++)
			weight[i][0]=Math.floor(weight[i][0]);
	}

	@Test
	public void testPearsonRWithWeights() {

		TestConfiguration config = getTestConfiguration(TEST_SCALE_SCALE_WEIGHTS);
		
		config.addVariable("rows", rows);

		loadTestConfiguration(config);

		//long seed = System.currentTimeMillis();
		//System.out.println("Seed = " + seed);
        double[][] X = getRandomMatrix(rows, 1, minVal, maxVal, 0.1, System.currentTimeMillis());
        double[][] Y = getRandomMatrix(rows, 1, minVal, maxVal, 0.1, System.currentTimeMillis());
        double[][] WM = getRandomMatrix(rows, 1, 1, maxW, 1, System.currentTimeMillis());
        round(WM);
        
		writeInputMatrix("X", X, true);
		writeInputMatrix("Y", Y, true);
		writeInputMatrix("WM", WM, true);
        createHelperMatrix();
		
		/*
        double PearsonR = 0.0;
		PearsonR = computePearsonR(X,Y, WM, rows);
        writeExpectedHelperMatrix("outPearsonR", PearsonR);
		*/
        
		boolean exceptionExpected = false;
		/*
		 * Expected number of jobs:
		 * Mean etc - 2 jobs (reblock & gmr)
		 * Cov etc - 2 jobs
		 * Final output write - 1 job
		 */
		int expectedNumberOfJobs = 6;
		runTest(exceptionExpected, null, expectedNumberOfJobs);
		runRScript();
		
		for(String file: config.getOutputFiles())
		{
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS(file);
			HashMap<CellIndex, Double> rfile = readRMatrixFromFS(file);
			TestUtils.compareMatrices(dmlfile, rfile, eps, file+"-DML", file+"-R");
		}

	}
	

}
