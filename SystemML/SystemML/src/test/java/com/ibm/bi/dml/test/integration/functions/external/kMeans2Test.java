/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2015
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.functions.external;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;



public class kMeans2Test extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2015\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";

	private static final String TEST_DIR = "functions/external/";
	private static final String TEST_KMEANS = "kMeans2";  // The version w/ init kCenters


	@Override
	public void setUp() {
		addTestConfiguration(TEST_KMEANS, new TestConfiguration(TEST_DIR, TEST_KMEANS, new String[] { "kcenters", "kcentersWithInit"}));
	}

	@Test
	@SuppressWarnings("deprecation")
	public void testkMeansTest() 
	{
		int rows = 100;
		int cols = 10;
		int centers = 5;

		TestConfiguration config = availableTestConfigurations.get(TEST_KMEANS);
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		loadTestConfiguration(config);

		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String KMEANS_HOME = baseDirectory;
		fullDMLScriptName = KMEANS_HOME + TEST_KMEANS + ".dml";
		programArgs = new String[]{"-args",  KMEANS_HOME + INPUT_DIR + "M" , 
				Integer.toString(rows), Integer.toString(cols), 
				 KMEANS_HOME + INPUT_DIR + "initialCenters" , 
				 KMEANS_HOME + OUTPUT_DIR + "kcenters" ,
				 KMEANS_HOME + OUTPUT_DIR + "kcentersWithInit" };

		double[][] M = getRandomMatrix(rows, cols, -1, 1, 0.05, 10);
		double[][] initCenters = getRandomMatrix(centers, cols, -1, 1, 0.9, 20);

		writeInputMatrix("M", M);
		writeInputMatrix("initialCenters",initCenters);

		HashMap<CellIndex, Double> expected_kCenters = TestUtils.readDMLMatrixFromHDFS(baseDirectory + "kMeans2/kMeansWrapperOutput1");
		HashMap<CellIndex, Double> expected_kCentersWithInit = TestUtils.readDMLMatrixFromHDFS(baseDirectory + "kMeans2/kMeansWrapperOutput2");

		double [][] kcenters_arr = TestUtils.convertHashMapToDoubleArray(expected_kCenters);
		double [][] kcenters_init_arr = TestUtils.convertHashMapToDoubleArray(expected_kCentersWithInit);

		writeExpectedMatrix("kcenters", kcenters_arr);
		writeExpectedMatrix("kcentersWithInit", kcenters_init_arr);

		runTest(true, false, null, -1);
		
		compareResults(0.0001);
	}
}
