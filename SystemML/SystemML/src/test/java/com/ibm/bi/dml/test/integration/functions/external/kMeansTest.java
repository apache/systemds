/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.external;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

public class kMeansTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
		
	private final static String TEST_KMEANS = "kMeans"; // The version w/o init kCenters
	
	@Override
	public void setUp() {
		baseDirectory = SCRIPT_DIR + "functions/external/";
		availableTestConfigurations.put(TEST_KMEANS, new TestConfiguration(TEST_KMEANS, new String[] { "kCenters"}));
	}

	@Test
	public void testkMeansTest() {
		
		int rows = 100;
		int cols = 10;
		int centers = 5;

		TestConfiguration config = availableTestConfigurations.get(TEST_KMEANS);
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String KMEANS_HOME = baseDirectory;
		fullDMLScriptName = KMEANS_HOME + TEST_KMEANS + ".dml";
		programArgs = new String[]{"-args",  KMEANS_HOME + INPUT_DIR + "M" , 
				                        Integer.toString(rows), Integer.toString(cols), 
				                         KMEANS_HOME + OUTPUT_DIR + "kCenters" };
		
		double[][] M = getRandomMatrix(rows, cols, -1, 1, 0.05, 10);
		
		writeInputMatrix("M", M);
		
		HashMap<CellIndex, Double> expected_kmeans = TestUtils.readDMLMatrixFromHDFS(baseDirectory + "kMeans/kMeansWrapperOutput");
		
		
		double [][] expected_means_arr = TestUtils.convertHashMapToDoubleArray(expected_kmeans);
		
		writeExpectedMatrix("kCenters", expected_means_arr);
		
				
		loadTestConfiguration(config);

		runTest(true, false, null, -1);

		compareResults(0.0001);
	}
}
