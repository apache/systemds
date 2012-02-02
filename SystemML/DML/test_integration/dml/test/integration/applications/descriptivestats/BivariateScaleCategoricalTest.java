package dml.test.integration.applications.descriptivestats;


import java.util.HashMap;

import org.junit.Test;

import dml.runtime.matrix.io.MatrixValue.CellIndex;
import dml.test.integration.AutomatedTestBase;
import dml.test.integration.TestConfiguration;
import dml.test.utils.TestUtils;

public class BivariateScaleCategoricalTest extends AutomatedTestBase {

	private final static String TEST_DIR = "applications/descriptivestats/";
	private final static String TEST_SCALE_NOMINAL = "ScaleCategoricalTest";
	private final static String TEST_SCALE_NOMINAL_WEIGHTS = "ScaleCategoricalWithWeightsTest";

	private final static double eps = 1e-9;
	private final static int rows = 10000;
	private final static int ncatA = 100; // # of categories in A
	private final static double minVal = 0; // minimum value in Y
	private final static double maxVal = 250; // minimum value in Y
	private int maxW = 10;    // maximum weight
	
	@Override
	public void setUp() {
		addTestConfiguration(TEST_SCALE_NOMINAL, new TestConfiguration(TEST_DIR, "ScaleCategoricalTest", new String[] { "outEta", "outAnovaF", "outVarY", "outMeanY", "outCatFreqs", "outCatMeans", "outCatVars" }));
		addTestConfiguration(TEST_SCALE_NOMINAL_WEIGHTS, new TestConfiguration(TEST_DIR, "ScaleCategoricalWithWeightsTest", new String[] { "outEta", "outAnovaF", "outVarY", "outMeanY", "outCatFreqs", "outCatMeans", "outCatVars" }));
	}
	
	@Deprecated
	class ScaleCategoricalStats {
		double Eta, AnovaF;
		double ybar, yvar;
		double[][] catMeans, catVars, catStdDevs, catFreqs;
		
		ScaleCategoricalStats(double e, double f, double yb, double yv, double[] cm, double[] cv, double[] freq) {
			Eta = e;
			AnovaF = f;
			ybar = yb;
			yvar = yv;
			
			catMeans 	= new double[cm.length][1];
			catVars 	= new double[cv.length][1];
			catFreqs 	= new double[freq.length][1];
			for ( int i=0; i < cm.length; i++ ) {
				catMeans[i][0] = cm[i];
				catVars[i][0] = cv[i];
				catFreqs[i][0] = freq[i];
			}
		}
	}
	
	@Deprecated
	ScaleCategoricalStats computeScaleCategoricalStats ( double[][] x, double[][] y, double[][] w, int rows ) {
		boolean weights = (w != null);
		
		// count the number of categories in x
		int numcat = 0;
		double W = 0.0, ybar=0.0;
		for ( int i=0; i < rows; i++ ) {
			numcat = ( numcat < (int)x[i][0] ? (int)x[i][0] : numcat);
			W = W + (weights ? w[i][0] : 1);
			ybar = ybar + (weights ? y[i][0]*w[i][0] : y[i][0]);
		}
		ybar = ybar/W; // mean of Y
	
		double[] catWeights = new double[numcat];	// counts per category
		double[] catMeans = new double[numcat]; 	// y-sums per category
		double[] catVars = new double[numcat]; 		// y-vars per category
		
		for ( int i=0; i < numcat; i++ ) {
			catWeights[i] = 0.0;
			catMeans[i] = 0.0;
			catVars[i] = 0.0;
		}
		
		int item;
		double yvar = 0.0; // variance in Y
		if ( weights ) {
			for ( int i=0; i < rows; i++ ) {
				item = (int)x[i][0];
				catWeights[item-1] += (int)w[i][0];
				catMeans[item-1] += (y[i][0]*w[i][0]);
				yvar = yvar + (w[i][0] * (y[i][0]-ybar) * (y[i][0]-ybar));
			}			
		}
		else {
			for ( int i=0; i < rows; i++ ) {
				item = (int)x[i][0];
				catWeights[item-1]++;
				catMeans[item-1] += y[i][0];
				yvar = yvar + ((y[i][0]-ybar) * (y[i][0]-ybar));
			}
		}
		yvar = yvar/(W-1);
		for ( int i=0; i < numcat; i++ ) {
			catMeans[i] = catMeans[i]/catWeights[i];
		}
		
		if ( weights ) {
			for ( int i=0; i < rows; i++ ) {
				item = (int)x[i][0];
				catVars[item-1] = catVars[item-1] + (w[i][0] * (y[i][0]-catMeans[item-1]) * (y[i][0]-catMeans[item-1]));
			}
		}
		else {
			for ( int i=0; i < rows; i++ ) {
				item = (int)x[i][0];
				catVars[item-1] = catVars[item-1] + (y[i][0]-catMeans[item-1]) * (y[i][0]-catMeans[item-1]);				
			}
		}
		
		double Eta=0.0, F_num=0.0, F_den=0.0;
		for ( int i=0; i < numcat; i++ ) {
			catVars[i] = catVars[i]/(catWeights[i]-1);
		
			Eta += (catWeights[i]-1)*catVars[i];
			F_num += (catWeights[i]*(catMeans[i]-ybar)*(catMeans[i]-ybar));
			F_den += (catWeights[i]-1)*catVars[i];
		}
		Eta = Eta/((W-1)*yvar);
		Eta = Math.sqrt(1-Eta);
		
		//System.out.println("testcase F_num = " + F_num + "/(" + numcat + "-1) = " + F_num/(numcat-1));
		//System.out.println("testcase F_den = " + F_den + "/(" + W + "-" + numcat + ") = " + F_den/(W-numcat));
		
		
		
		F_num = F_num/(numcat-1);
		F_den = F_den/(W-numcat);
		double AnovaF = F_num/F_den;
		//double AnovaF = Math.exp(Math.log(F_num)-Math.log(F_den));
		
		//System.out.println("exp Eta=" + Eta);
		//System.out.println("testcase AnovaF=" + AnovaF);
		//System.out.println("testcase ybar=" + ybar);
		//System.out.println("testcase yvar=" + yvar);
		
		return new ScaleCategoricalStats(Eta, AnovaF, ybar, yvar, catMeans, catVars, catWeights);
	}
	
	@Deprecated
	ScaleCategoricalStats computeScaleCategoricalStats_IncrementalApproach ( double[][] x, double[][] y, double[][] w, int rows ) {
		boolean weights = (w != null);
		
		if ( weights ) 
			throw new RuntimeException("weights not supported yet for incremental!");
		
		// compute overall mean and variance for y attribute
		double n=0, mean=0, M2 = 0, delta;
		for ( int i=0; i < rows; i++ ) {
			n = n+1;
			delta = y[i][0]-mean;
			mean = mean + (delta/n);
			M2 = M2 + delta*(y[i][0] - mean);
		}
		double ybar = mean;
		double yvar = M2/(n-1); 
		double W = n;
		
		// count the number of categories in x
		int numcat = 0;
		for ( int i=0; i < rows; i++ ) {
			numcat = ( numcat < (int)x[i][0] ? (int)x[i][0] : numcat);
		}
		
		// compute category-wise mean and variance
		double[] catWeights = new double[numcat];	// counts per category
		double[] catMeans = new double[numcat]; 	// y-sums per category
		double[] catVars = new double[numcat]; 		// y-vars per category
		double[] catDelta = new double[numcat]; 		// y-vars per category
		double[] catM2 = new double[numcat]; 		// y-vars per category

		for ( int i=0; i < numcat; i++ ) {
			catWeights[i] = 0.0;
			catMeans[i] = 0.0;
			catVars[i] = 0.0;
			catDelta[i] = 0.0;
			catM2[i] = 0.0;
		}
		
		int item;
		if ( weights ) {
			throw new RuntimeException("weights not supported yet for incremental!");
		}
		else {
			for ( int i=0; i < rows; i++ ) {
				item = (int)x[i][0];
				catWeights[item-1]++; // n = n + 1
				catDelta[item-1] = y[i][0] - catMeans[item-1]; // delta = x - mean
				catMeans[item-1] = catMeans[item-1] + (catDelta[item-1]/catWeights[item-1]); // mean = mean + delta/n
				catM2[item-1] = catM2[item-1] + catDelta[item-1]*(y[i][0]-catMeans[item-1]); // M2 = M2 + delta*(x - mean) 
			}
		}
		for ( int i=0; i < numcat; i++ ) {
			catVars[i] = catM2[i]/(catWeights[i]-1);
		}
		
		double Eta=0.0, F_num=0.0, F_den=0.0;
		for ( int i=0; i < numcat; i++ ) {
			Eta += (catWeights[i]-1)*catVars[i];
			F_num += (catWeights[i]*(catMeans[i]-ybar)*(catMeans[i]-ybar));
			F_den += (catWeights[i]-1)*catVars[i];
		}
		Eta = Eta/((W-1)*yvar);
		Eta = Math.sqrt(1-Eta);
		
		//System.out.println("testcase incr F_num = " + F_num + "/(" + numcat + "-1) = " + F_num/(numcat-1));
		//System.out.println("testcase incr F_den = " + F_den + "/(" + W + "-" + numcat + ") = " + F_den/(W-numcat));
		
		F_num = F_num/(numcat-1);
		F_den = F_den/(W-numcat);
		double AnovaF = F_num/F_den;
		//double AnovaF = Math.exp(Math.log(F_num)-Math.log(F_den));
		
		//System.out.println("exp Eta=" + Eta);
		//System.out.println("testcase incr AnovaF=" + AnovaF);
		//System.out.println("testcar incr ybar=" + ybar);
		//System.out.println("testcar incr yvar=" + yvar);
		
		return new ScaleCategoricalStats(Eta, AnovaF, ybar, yvar, catMeans, catVars, catWeights);
	}
	
	@Test
	public void testScaleCategorical() {
		TestConfiguration config = getTestConfiguration(TEST_SCALE_NOMINAL);
		
		config.addVariable("rows", rows);

		loadTestConfiguration(config);

        double[][] A = getRandomMatrix(rows, 1, 1, ncatA, 1, System.currentTimeMillis()) ; // System.currentTimeMillis());
        round(A);
        double[][] Y = getRandomMatrix(rows, 1, minVal, maxVal, 0.1, System.currentTimeMillis()) ; // System.currentTimeMillis()+1);

		writeInputMatrix("A", A, true);
		writeInputMatrix("Y", Y, true);

        createHelperMatrix();
        
        //ScaleCategoricalStats stats = computeScaleCategoricalStats(A,Y,null,rows);
        ScaleCategoricalStats stats = computeScaleCategoricalStats_IncrementalApproach(A,Y,null,rows);
        
        writeExpectedHelperMatrix("outEta", stats.Eta);
        writeExpectedHelperMatrix("outAnovaF", stats.AnovaF);
        writeExpectedHelperMatrix("outVarY", stats.yvar);
        writeExpectedHelperMatrix("outMeanY", stats.ybar);
        writeExpectedMatrix("outCatMeans", stats.catMeans);
        writeExpectedMatrix("outCatVars", stats.catVars);
        writeExpectedMatrix("outCatFreqs", stats.catFreqs);
		
		// boolean exceptionExpected = false;
		/*
		 * Expected number of jobs:
		 */
		//int expectedNumberOfJobs = 5;
		//runTest(exceptionExpected, null, expectedNumberOfJobs);
		runTest();
		runRScript();
		
		for(String file: config.getOutputFiles())
		{
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS(file);
			HashMap<CellIndex, Double> rfile = readRMatrixFromFS(file);
			TestUtils.compareMatrices(dmlfile, rfile, eps, file+"-DML", file+"-R");
		}
		
		// compareResults(eps);

	}
	
	private void round(double[][] weight) {
		for(int i=0; i<weight.length; i++)
			weight[i][0]=Math.floor(weight[i][0]);
	}

	@Test
	public void testScaleCategoricalWithWeights() {
		TestConfiguration config = getTestConfiguration(TEST_SCALE_NOMINAL_WEIGHTS);
		
		config.addVariable("rows", rows);

		loadTestConfiguration(config);

        double[][] A = getRandomMatrix(rows, 1, 1, ncatA, 1, 98734); // System.currentTimeMillis());
        double[][] Y = getRandomMatrix(rows, 1, minVal, maxVal, 0.1, 7895); // System.currentTimeMillis());
        double[][] WM = getRandomMatrix(rows, 1, 1, maxW, 1, 234); // System.currentTimeMillis());
        round(A);
        round(WM);
        
		writeInputMatrix("A", A, true);
		writeInputMatrix("Y", Y, true);
		writeInputMatrix("WM", WM, true);
		
        createHelperMatrix();
        ScaleCategoricalStats stats = computeScaleCategoricalStats(A,Y,WM,rows);
        
        writeExpectedHelperMatrix("outEta", stats.Eta);
        writeExpectedHelperMatrix("outAnovaF", stats.AnovaF);
        writeExpectedHelperMatrix("outVarY", stats.yvar);
        writeExpectedHelperMatrix("outMeanY", stats.ybar);
        writeExpectedMatrix("outCatMeans", stats.catMeans);
        writeExpectedMatrix("outCatVars", stats.catVars);
        writeExpectedMatrix("outCatFreqs", stats.catFreqs);
		
		// boolean exceptionExpected = false;
		/*
		 * Expected number of jobs:
		 * Mean etc - 2 jobs (reblock & gmr)
		 * Cov etc - 2 jobs
		 * Final output write - 1 job
		 */
		//int expectedNumberOfJobs = 5;
		//runTest(exceptionExpected, null, expectedNumberOfJobs);
		runTest();
		runRScript();
		
		for(String file: config.getOutputFiles())
		{
			HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS(file);
			HashMap<CellIndex, Double> rfile = readRMatrixFromFS(file);
			TestUtils.compareMatrices(dmlfile, rfile, eps, file+"-DML", file+"-R");
		}

	}
	

}
