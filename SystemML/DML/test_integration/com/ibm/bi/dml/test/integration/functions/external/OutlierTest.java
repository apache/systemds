package com.ibm.bi.dml.test.integration.functions.external;

import java.util.PriorityQueue;

import org.junit.Test;
import org.nimble.algorithms.outlier.FindOutliersTask;

import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;


/**
 * 
 * @author Amol Ghoting
 */
public class OutlierTest extends AutomatedTestBase {

	private final static String TEST_OUTLIER = "Outlier"; 

	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/external/";
		availableTestConfigurations.put(TEST_OUTLIER, new TestConfiguration(TEST_OUTLIER, new String[] { "o"}));
	}

	@Test
	public void testOutlierTest() {

		int rows = 100;
		int cols = 10;
		int m = 5;
		int K = 2;

		TestConfiguration config = availableTestConfigurations.get("Outlier");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);

		/* This is for running the junit test by constructing the arguments directly */
		String OUTLIER_HOME = baseDirectory;
		dmlArgs = new String[]{"-f", OUTLIER_HOME + TEST_OUTLIER + ".dml",
				"-args",  OUTLIER_HOME + INPUT_DIR + "M" , 
				Integer.toString(rows), Integer.toString(cols), 
				 OUTLIER_HOME + OUTPUT_DIR + "o" };
		dmlArgsDebug = new String[]{"-f", OUTLIER_HOME + TEST_OUTLIER + ".dml", "-d", 
				"-args",  OUTLIER_HOME + INPUT_DIR + "M" , 
				Integer.toString(rows), Integer.toString(cols), 
				 OUTLIER_HOME + OUTPUT_DIR + "o" };

		double[][] M = getRandomMatrix(rows, cols, -1, 1, 0.05, -1);

		writeInputMatrix("M", M);

		loadTestConfiguration(config);

		// there is no expected number of M/R job calculated, set to default for now
		runTest(true, false, null, -1);


		//holds input
		float[][] bin = new float[rows][rows];
		
		//holds output
		double [][] outMatrix = new double[rows][m];

		for(int i =0; i < rows; i++)
			for(int j=0; j < cols; j++)
				bin[i][j] = (float) M[i][j];

		//determine knn for each point
		PriorityQueue[] nearestNeighbors = new PriorityQueue [rows];
		double [][] dists = new double[rows][rows];
		float [] knn_dist = new float [rows];
		
		//distance to knn to find top m
		PriorityQueue<Float> outlier_knn = new PriorityQueue<Float>();

		for (int i = 0; i < rows; i++) {
			nearestNeighbors[i] = new PriorityQueue<Float>();
			for (int j = 0; j < rows; j++) {
				float f = FindOutliersTask.dist(bin[i], bin[j]);
				dists[i][j] = f;
				if (i != j) {
					nearestNeighbors[i].add(new Float(f));
				}
			}
			
			float knn = 0.0f;
			for (int k = 0; k < K; k++) {
				knn = (Float) nearestNeighbors[i].poll();
			}
			outlier_knn.add(knn);
			knn_dist[i] = knn;
		}
		
		while(outlier_knn.size() > m)
			outlier_knn.poll();
		
		//find top outliers in reverse order 
		while(outlier_knn.size() > 0)
		{
			float threshold = outlier_knn.poll();
			for(int i=0; i < rows; i++)
			{
				if(knn_dist[i] == threshold)
				{
					for(int j=0; i < cols; j++)
						outMatrix[outlier_knn.size() -1][j] = bin[i][j];
				}
			}
		}
		
		writeExpectedMatrix("o", outMatrix);

		compareResults(0.01);
	}
}
