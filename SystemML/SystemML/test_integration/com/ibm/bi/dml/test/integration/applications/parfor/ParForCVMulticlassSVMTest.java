/**
 * IBM Confidential
 * OCO Source Materials
 * (C) Copyright IBM Corp. 2010, 2013
 * The source code for this program is not published or otherwise divested of its trade secrets, irrespective of what has been deposited with the U.S. Copyright Office.
 */

package com.ibm.bi.dml.test.integration.applications.parfor;

import java.util.HashMap;

import org.junit.Test;

import com.ibm.bi.dml.runtime.matrix.MatrixCharacteristics;
import com.ibm.bi.dml.runtime.matrix.data.MatrixValue.CellIndex;
import com.ibm.bi.dml.test.integration.AutomatedTestBase;
import com.ibm.bi.dml.test.integration.TestConfiguration;
import com.ibm.bi.dml.test.utils.TestUtils;

public class ParForCVMulticlassSVMTest extends AutomatedTestBase 
{
	@SuppressWarnings("unused")
	private static final String _COPYRIGHT = "Licensed Materials - Property of IBM\n(C) Copyright IBM Corp. 2010, 2013\n" +
                                             "US Government Users Restricted Rights - Use, duplication  disclosure restricted by GSA ADP Schedule Contract with IBM Corp.";
	
	private final static String TEST_NAME = "parfor_cv_multiclasssvm";
	private final static String TEST_DIR = "applications/parfor/";
	
	private final static double eps = 1e-10; 
		
	//script parameters
	private final static int rows = 1200;
	private final static int cols = 70; 
	
	private final static double sparsity1 = 1.0;
	private final static double sparsity2 = 0.1;
	
	private final static int k = 4;
	private final static int intercept = 0;
	private final static int numclasses = 3;
	private final static double epsilon = 0.001;
	private final static double lambda = 1.0;
	private final static int maxiter = 100;
	
	@Override
	public void setUp() 
	{
		addTestConfiguration(
				TEST_NAME, 
				new TestConfiguration(TEST_DIR, TEST_NAME, 
				new String[] { "stats" })   );  
	}
	
	@Test
	public void testForCVMulticlassSVMSerialDense() 
	{
		runParForMulticlassSVMTest(0, false);
	}
	
	@Test
	public void testForCVMulticlassSVMSerialSparse() 
	{
		runParForMulticlassSVMTest(0, false);
	}

	@Test
	public void testParForCVMulticlassSVMLocalDense() 
	{
		runParForMulticlassSVMTest(1, false);
	}
	
	@Test
	public void testParForCVMulticlassSVMLocalSparse() 
	{
		runParForMulticlassSVMTest(1, true);
	}
	
	@Test
	public void testParForCVMulticlassSVMOptDense() 
	{
		runParForMulticlassSVMTest(4, false);
	}
	
	@Test
	public void testParForCVMulticlassSVMOptSparse() 
	{
		runParForMulticlassSVMTest(4, true);
	}

	/**
	 * 
	 * @param scriptNum
	 * @param sparse
	 */
	private void runParForMulticlassSVMTest( int scriptNum, boolean sparse )
	{	
		TestConfiguration config = getTestConfiguration(TEST_NAME);
		config.addVariable("rows", rows);
		config.addVariable("cols", cols);
		
		/* This is for running the junit test the new way, i.e., construct the arguments directly */
		String HOME = SCRIPT_DIR + TEST_DIR;
		fullDMLScriptName = HOME + TEST_NAME +scriptNum + ".dml";
		programArgs = new String[]{"-args", HOME + INPUT_DIR + "X" ,
				                        HOME + INPUT_DIR + "y",
				                        Integer.toString(rows),
				                        Integer.toString(cols),
				                        Integer.toString(k),
				                        Integer.toString(intercept),
				                        Integer.toString(numclasses),
				                        Double.toString(epsilon),
				                        Double.toString(lambda),
				                        Integer.toString(maxiter),
				                        HOME + OUTPUT_DIR + "stats",
				                        HOME + INPUT_DIR + "P"};
		fullRScriptName = HOME + TEST_NAME + ".R";
		rCmd = "Rscript" + " " + fullRScriptName + " " + 
		       HOME + INPUT_DIR + " " + Integer.toString(k) + " " + Integer.toString(intercept) 
		       + " " + Integer.toString(numclasses) + " " + Double.toString(epsilon) + " " + Double.toString(lambda)  
		       + " " + Integer.toString(maxiter)+ " " + HOME + EXPECTED_DIR;
		
		loadTestConfiguration(config);

		double sparsity = (sparse)? sparsity2 : sparsity1;
		
		//generate actual dataset
		double[][] X = getRandomMatrix(rows, cols, 0, 1, sparsity, 7); 
		double[][] y = round(getRandomMatrix(rows, 1, 0.51, numclasses+0.49, 1.0, 7)); 
		double[][] P = round(getRandomMatrix(rows, 1, 0.51, k+0.49, 1.0, 3)); 

		MatrixCharacteristics mc1 = new MatrixCharacteristics(rows,cols,-1,-1);
		writeInputMatrixWithMTD("X", X, true, mc1);
		MatrixCharacteristics mc2 = new MatrixCharacteristics(rows,1,-1,-1);
		writeInputMatrixWithMTD("y", y, true, mc2);		
		MatrixCharacteristics mc3 = new MatrixCharacteristics(rows,1,-1,-1);
		writeInputMatrixWithMTD("P", P, true, mc3);		
		
		boolean exceptionExpected = false;
		runTest(true, exceptionExpected, null, -1);

		runRScript(true); 
		
		//compare matrices 
		HashMap<CellIndex, Double> dmlfile = readDMLMatrixFromHDFS("stats");
		HashMap<CellIndex, Double> rfile  = readRMatrixFromFS("stats");
		TestUtils.compareMatrices(dmlfile, rfile, eps, "DML", "R");
	}
	
	private double[][] round(double[][] data) {
		for(int i=0; i<data.length; i++)
			data[i][0]=Math.round(data[i][0]);
		return data;
	}
}