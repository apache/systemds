package com.ibm.bi.dml.test.integration.functions.external;

import java.util.HashSet;
import java.util.PriorityQueue;

import org.junit.Test;
import org.nimble.algorithms.outlier.FindOutliersTask;

import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;

public class OutlierTest extends AutomatedTestBase {

	private final static String TEST_OUTLIER = "Outlier"; 

	private int _rows = 100;
	private int _cols = 10;
	private int _m = 5;
	private int _K = 2;
	private double _sparsity1 = 0.05;
	private double _sparsity2 = 0.5;
	
	
	@Override
	@SuppressWarnings("deprecation")
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/external/";
		availableTestConfigurations.put(TEST_OUTLIER, new TestConfiguration(TEST_OUTLIER, new String[] { "o"}));
	}

	@Test
	public void testOutlierSparse() 
	{
		runOutlierTest(_rows, _cols, _m, _K, _sparsity1);
	}
	
	@Test
	public void testOutlierDense() 
	{
		runOutlierTest(_rows, _cols, _m, _K, _sparsity2);
	}
	
	
	/**
	 * 
	 * @param rows
	 * @param cols
	 * @param m 	num outliers
	 * @param K		num neighbors
	 */
	@SuppressWarnings("unchecked")
	public void runOutlierTest(int rows, int cols, int m, int K, double sparsity) 
	{
		TestConfiguration config = availableTestConfigurations.get("Outlier");
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);

		/* This is for running the junit test by constructing the arguments directly */
		String OUTLIER_HOME = baseDirectory;
		fullDMLScriptName = OUTLIER_HOME + TEST_OUTLIER + ".dml";
		programArgs = new String[]{"-args",  OUTLIER_HOME + INPUT_DIR + "M" , 
				Integer.toString(rows), Integer.toString(cols), 
				 OUTLIER_HOME + OUTPUT_DIR + "o" };

		double[][] M = getRandomMatrix(rows, cols, -1, 1, sparsity, 1471 );

		writeInputMatrix("M", M);

		loadTestConfiguration(config);

		// there is no expected number of M/R job calculated, set to default for now
		runTest(true, false, null, -1);
		
		//holds input
		float[][] bin = new float[rows][cols];
		
		//holds output
		double [][] outMatrix = new double[m][cols]; 

		//copy input
		for(int i =0; i < rows; i++)
			for(int j=0; j < cols; j++)
				bin[i][j] = (float) M[i][j];
		
		//determine knn for each point
		PriorityQueue<Float>[] nearestNeighbors = new PriorityQueue[rows];
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
		
		//remove outliers not in top m
		while( outlier_knn.size()>m )
			outlier_knn.poll();
		
		//find top outliers rowIDs in reverse order
		HashSet<Integer> outliers = new HashSet<Integer>(); 
		while( outliers.size() < m )
		{
			float threshold = outlier_knn.poll();
			for(int i=0; i < rows; i++)
			{
				//find outlier row
				if(knn_dist[i] == threshold)
				{
					//skip rows already processed (if multiple rows have equal threshold)
					if( outliers.contains(i) )
						continue;
					
					//copy row to output
					for(int j=0; j < cols; j++)
						outMatrix[outliers.size()][j] = bin[i][j];
					outliers.add(i);
					
					//finish if we obtained m outliers
					if( outliers.size() == m )
						break;
				}
			}	
		}	
		
		writeExpectedMatrix("o", outMatrix);

		compareResults(0.01);
	}
}
